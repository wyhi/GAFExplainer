from GNNs import *

from utils import *
from AAExplainer import AAFExplainer

from utils import XCollector
from AAExplainer import PlotUtils

import os
import torch
import numpy as np

from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import random_split, Subset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# -------------------step1 下载数据集---------------------
class MUTAGDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name.upper()
        super(MUTAGDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __len__(self):
        return len(self.slices['x']) - 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return ['MUTAG_A', 'MUTAG_graph_labels', 'MUTAG_graph_indicator', 'MUTAG_node_labels']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder.
        Node labels:
          0  C
          1  N
          2  O
          3  F
          4  I
          5  Cl
          6  Br
        """
        with open(os.path.join(self.raw_dir, 'MUTAG_node_labels.txt'), 'r') as f:
            nodes_all_temp = f.read().splitlines()
            nodes_all = [int(i) for i in nodes_all_temp]  # 将节点的原子类型以整数形式存储在一个列表中

        adj_all = np.zeros((len(nodes_all), len(nodes_all)))
        with open(os.path.join(self.raw_dir, 'MUTAG_A.txt'), 'r') as f:
            adj_list = f.read().splitlines()
        for item in adj_list:
            lr = item.split(', ')
            l = int(lr[0])
            r = int(lr[1])
            adj_all[
                l - 1, r - 1] = 1  # 对应了3DEGN里面的weight_m = get_weight_matrix(g_mask)-->即all_adj_m里的每一个元素，但是这里好像没涉及边的权重信息

        with open(os.path.join(self.raw_dir, 'MUTAG_graph_indicator.txt'), 'r') as f:
            graph_indicator_temp = f.read().splitlines()
            graph_indicator = [int(i) for i in graph_indicator_temp]
            graph_indicator = np.array(graph_indicator)

        with open(os.path.join(self.raw_dir, 'MUTAG_graph_labels.txt'), 'r') as f:
            graph_labels_temp = f.read().splitlines()
            graph_labels = [int(i) for i in graph_labels_temp]

        data_list = []
        for i in range(1, 189):
            idx = np.where(graph_indicator == i)
            graph_len = len(idx[0])
            adj = adj_all[idx[0][0]:idx[0][0] + graph_len, idx[0][0]:idx[0][0] + graph_len]
            label = int(graph_labels[i - 1] == 1)
            feature = nodes_all[idx[0][0]:idx[0][0] + graph_len]  # 特征是原子类型
            nb_clss = 7
            targets = np.array(feature).reshape(-1)
            one_hot_feature = np.eye(nb_clss)[targets]
            data_example = Data(x=torch.from_numpy(one_hot_feature).float(),
                                edge_index=dense_to_sparse(torch.from_numpy(adj))[0],
                                y=label)
            data_list.append(data_example)

        torch.save(self.collate(data_list), self.processed_paths[0])


def get_dataloader(dataset, batch_size, random_split_flag=True, data_split_ratio=[0.7, 0.2, 0.1], seed=1):
    """ data_args.data_split_ratio : list [float, float, float]
        return a dict with three data loaders
    """
    if not random_split_flag and hasattr(dataset, 'supplement'):
        assert 'split_indices' in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement['split_indices']
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test],
                                         generator=torch.Generator().manual_seed(seed))

    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['val'] = DataLoader(eval, batch_size=batch_size, shuffle=False)
    dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=False)
    return dataloader


""" 188 molecules where label = 1 denotes mutagenic effect """
dataset = MUTAGDataset('datasets', 'mutag')

dataloader = get_dataloader(dataset, batch_size=1, random_split_flag=True, data_split_ratio=[0.8, 0.1, 0.1], seed=1)
data_indices = dataloader['test'].dataset.indices
print("test_indices", data_indices)
pgexplainer_trainset = dataloader['train'].dataset


# -------------------step2 训练模型/保存模型/下载模型---------------------
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def evaluate(model, data):
    '''
    定义eval评价函数。
    '''
    with torch.no_grad():
        model.eval()
        output = model(data.x, data.edge_index)
        acc = accuracy(output, data.y)
        return F.nll_loss(output, data.y).detach().item(), acc.detach().item()


def train(model, train_data_loader, valid_data_loader, optimizer, scheduler, epoch_num):
    '''
    模型训练。
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.train()
    train_losses = []
    val_losses = []
    for epoch in range(epoch_num):
        train_loss = 0
        train_acc = 0
        for iter, data in enumerate(train_data_loader):
            model.train()
            log_logits = model(data.x, data.edge_index).to(device)
            loss = F.nll_loss(log_logits, data.y)
            acc = accuracy(log_logits, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
            train_acc += acc
        train_loss /= (iter + 1)
        train_acc /= (iter + 1)
        val_loss = 0
        val_acc = 0
        for iter, data in enumerate(valid_data_loader):
            vl, acc = evaluate(model, data)

            val_loss += vl
            val_acc += acc
        val_loss /= (iter + 1)
        val_acc /= (iter + 1)
        print('Epoch {}, train_loss:{:.4f} train_acc:{:.4f} val_loss:{:.4f} val_acc:{:.4f}'.format(epoch, train_loss,
                                                                                                   train_acc, val_loss,
                                                                                                   val_acc))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()
    draw_epoches(np.array(train_losses), 'train_loss')
    draw_epoches(np.array(val_losses), 'val_loss')
    return model, train_losses, val_losses


def test(model, test_data_loader):
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for iter, data in enumerate(test_data_loader):
            log_logits = model(data.x, data.edge_index)
            loss = F.nll_loss(log_logits, data.y).detach().item()
            acc = accuracy(log_logits, data.y).detach().item()
            test_loss += loss
            test_acc += acc
    test_loss /= (iter + 1)
    test_acc /= (iter + 1)
    return test_loss, test_acc


train_or_not = False
save_or_not = False
# 训练模型

input_dim = dataset.num_node_features
output_dim = dataset.num_classes
# model=GIN_2l(model_level='graph', dim_node=input_dim, dim_hidden=256, num_classes=output_dim).to(device)
# model = GCN_2l(model_level='graph', dim_node=input_dim, dim_hidden=256, num_classes=output_dim).to(device)

model=GAT(model_level='graph',input_dim=input_dim,hidden_dim=8,num_classes=output_dim,heads=8).to(device)
epochs_num = 100
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_num, eta_min=0, last_epoch=-1,
                                                       verbose=False)

if train_or_not:
    train(model, dataloader['train'], dataloader['val'], optimizer, scheduler, epochs_num)

    test_loss, test_acc = test(model, dataloader['test'])
    print("after save", test_loss, test_acc)
# 保存模型
if save_or_not:
    torch.save(model.state_dict(), 'checkpoints/GAT_2l_0720.pth')

# 下载已经训练好的模型
state_dict = torch.load('checkpoints/GAT_2l_0720.pth')
model.load_state_dict(state_dict)
test_loss, test_acc = test(model, dataloader['test'])
print("after save", test_loss, test_acc)

# -------------------step3 开始操作解释器---------------------
"""
实例化解释器的类对象
in_channels:是连接之后的边的表示f12self获得的嵌入的channel大小,隐藏层中给的是256 因此这边是3*256=768
f12self是通过训练好的GNN获得的边嵌入，因此，其权重可以反映边的重要性 wij是f12self通过解释网络后的value
"""
explainer = AAFExplainer(model, in_channels=128,dim_node=input_dim, device=device, explain_graph=True)

# trainPG_or_not = True
# savePG_or_not = True

trainPG_or_not = False
savePG_or_not = False
# 训练、保存、下载解释网络
if trainPG_or_not:
    explainer.train_explanation_network(pgexplainer_trainset)
    if savePG_or_not:
        torch.save(explainer.state_dict(), 'PG_checkpoints/my_PG_nn_GAT_2l_1109.pth')

state_dict = torch.load('PG_checkpoints/my_PG_nn_GAT_2l_1109.pth')
explainer.load_state_dict(state_dict)

# -------------------step4 训练结果可视化---------------------
from torch_geometric.utils import to_networkx

# 实例化plot对象


plotutils = PlotUtils(dataset_name='mutag', is_show=True)
for data_idx in tqdm(range(188)):
    data = dataset[data_idx]
    graph = to_networkx(data)
    with torch.no_grad():
        # GNN和解释器都已经训练好了，输入对应的图和node_idx 去获得edge_mask(注意是子图上的mask)和相关预测值
        walks, masks, related_preds = explainer(data.x, data.edge_index, node_idx=None, y=data.y, top_k=6)

        plotutils.plot_soft_edge_mask(graph, edge_mask=masks[0], un_directed=True, top_k=6,
                                      figname=os.path.join('fig/my_gat', f"example_{data_idx}.png"), x=data.x)
#     # explainer.visualization(data, edge_mask=masks[0], top_k=6, plot_utils=plotutils, node_idx=None)

# -------------------step5 获取训练器相关指标---------------------

# top_k = 6
# undirected_graph = True
# --- Create data collector and explanation processor ---
sparsity_list = [0.45, 0.5, 0.55, 0.60, 0.65, 0.70]
for idx, sparsity in enumerate(sparsity_list):

    x_collector = XCollector()

    ### Run explainer on the given model and dataset
    index = -1
    # 也采用测试集中的边
    undirected_graph = True
    # top_k = top_k if not undirected_graph else top_k * 2

    for data_idx in tqdm(range(188)):
        data = dataset[data_idx]
        index += 1
        data.to(device)
        # 如果data的标签不存在 就跳过
        if torch.isnan(data.y[0].squeeze()):
            continue
        top_k = int((1 - sparsity) * len(data.edge_index[0]))
        with torch.no_grad():
            # 再次获得所需数据
            walks, masks, related_preds = explainer(data.x, data.edge_index, node_idx=None, y=data.y, top_k=top_k)
            masks = [mask.detach() for mask in masks]
        x_collector.collect_data(masks, related_preds)


    # 求得是test集上的平均值
    print(f'Fidelity: {x_collector.fidelity:.4f}\n'
          f'Fidelity_inv: {x_collector.fidelity_inv:.4f}\n'
          f'Sparsity: {x_collector.sparsity:.4f}')
