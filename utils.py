import torch
import torch.nn as nn
from typing import List, Union
from torch import Tensor
import numpy as np
from torch_geometric.data.data import Data
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils.num_nodes import maybe_num_nodes

from tqdm import tqdm
import matplotlib.pyplot as plt

def subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`. when num_hops == -1,
            the whole graph will be returned.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index # edge_index 0 to 1, col: source, row: target

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device, dtype=torch.int64).flatten()
    else:
        node_idx = node_idx.to(row.device)


    inv = None

    if num_hops != -1:
        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]
    else:
        subsets = node_idx
        cur_subsets = node_idx
        while 1:
            node_mask.fill_(False)
            node_mask[subsets] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets = torch.cat([subsets, col[edge_mask]]).unique()
            if not cur_subsets.equal(subsets):
                cur_subsets = subsets
            else:
                subset = subsets
                break



    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

# 重新划分数据集，使得标签中的每一类都均匀分布在训练集中
def split_dataset(dataset):
    indices = []
    num_classes = 4
    train_percent = 0.8
    for i in range(num_classes):  # 根据每一类分别设置索引
        index = (dataset.data.y == i).nonzero().view(-1)  # 显示所有y==i的节点索引的tensor([])列表
        # 将0~n-1（包括0和n-1）随机打乱后获得的数字序列，函数名是random permutation缩写
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:int(len(i) * train_percent)] for i in indices], dim=0)

    rest_index = torch.cat([i[int(len(i) * train_percent):] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    dataset.data.train_mask = index_to_mask(train_index, size=dataset.data.num_nodes)
    dataset.data.val_mask = index_to_mask(rest_index[:len(rest_index) // 2], size=dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(rest_index[len(rest_index) // 2:], size=dataset.data.num_nodes)

    dataset.data, dataset.slices = dataset.collate([dataset.data])

    return dataset


def draw_epoches(epoch_losses, title=''):
    plt.xlabel('epoches', fontsize=10)
    plt.ylabel(title, fontsize=10)
    plt.plot(list(range(1, len(epoch_losses) + 1)), epoch_losses)
    plt.title(title)
    plt.show()

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def evaluate(model, data, mask):
    with torch.no_grad():
        model.eval()
        log_logits = model(data.x, data.edge_index)
        loss = F.nll_loss(log_logits[mask], data.y[mask])
        acc = accuracy(log_logits[mask], data.y[mask])
        return loss.item(), acc.item()

def train(model, data, optimizer, epochs):
    print("Training the model...")
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        optimizer.zero_grad()
        log_logits = model(data.x, data.edge_index)
        # print('log_logits',log_logits)
        loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
        # print('log_logits[data.train_mask]',log_logits[data.train_mask])
        # print('data.y[data.train_mask]',data.y[data.train_mask])
        acc = accuracy(log_logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_loss = loss.detach().item()
        train_acc = acc.detach().item()

        val_loss, val_acc = evaluate(model, data, data.val_mask)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if epoch % 50 == 0:
            # print('Epoch {}, train_loss:{:.4f}? ,val_loss:{:.4f}'.format(epoch, train_loss, val_loss))
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.4f}'.format(train_loss),
                  'acc_train: {:.4f}'.format(train_acc),
                  'loss_val: {:.4f}'.format(val_loss),
                  'acc_val: {:.4f}'.format(val_acc))
    draw_epoches(np.array(train_losses), 'train_loss')
    draw_epoches(np.array(val_losses), 'val_loss')


def control_sparsity(mask: torch.Tensor, sparsity: float = None):
    r"""
    Transform the mask where top 1 - sparsity values are set to inf.
    Args:
        mask (torch.Tensor): Mask that need to transform.
        sparsity (float): Sparsity we need to control i.e. 0.7, 0.5 (Default: :obj:`None`).
    :rtype: torch.Tensor
    """
    if sparsity is None:
        sparsity = 0.7

    # Not apply here, Please refer to specific explainers in other directories
    #
    # if data_args.model_level == 'node':
    #     assert self.hard_edge_mask is not None
    #     mask_indices = torch.where(self.hard_edge_mask)[0]
    #     sub_mask = mask[self.hard_edge_mask]
    #     mask_len = sub_mask.shape[0]
    #     _, sub_indices = torch.sort(sub_mask, descending=True)
    #     split_point = int((1 - sparsity) * mask_len)
    #     important_sub_indices = sub_indices[: split_point]
    #     important_indices = mask_indices[important_sub_indices]
    #     unimportant_sub_indices = sub_indices[split_point:]
    #     unimportant_indices = mask_indices[unimportant_sub_indices]
    #     trans_mask = mask.clone()
    #     trans_mask[:] = - float('inf')
    #     trans_mask[important_indices] = float('inf')
    # else:
    _, indices = torch.sort(mask, descending=True)
    mask_len = mask.shape[0]
    split_point = int((1 - sparsity) * mask_len)
    important_indices = indices[: split_point]
    unimportant_indices = indices[split_point:]
    trans_mask = mask.clone()
    trans_mask[important_indices] = float('inf')
    trans_mask[unimportant_indices] = - float('inf')

    return trans_mask


def fidelity(ori_probs: torch.Tensor, unimportant_probs: torch.Tensor) -> float:
    r"""
    Return the Fidelity+ value according to collected data.

    Args:
        ori_probs (torch.Tensor): It is a vector providing original probabilities for Fidelity+ computation.
        unimportant_probs (torch.Tensor): It is a vector providing probabilities without important features
            for Fidelity+ computation.

    :rtype: float

    .. note::
        Please refer to `Explainability in Graph Neural Networks: A Taxonomic Survey
        <https://arxiv.org/abs/2012.15445>`_ for details.
    """

    drop_probability = ori_probs - unimportant_probs

    return drop_probability.mean().item()


def fidelity_inv(ori_probs: torch.Tensor, important_probs: torch.Tensor) -> float:
    r"""
    Return the Fidelity- value according to collected data.

    Args:
        ori_probs (torch.Tensor): It is a vector providing original probabilities for Fidelity- computation.
        important_probs (torch.Tensor): It is a vector providing probabilities with only important features
            for Fidelity- computation.

    :rtype: float

    .. note::
        Please refer to `Explainability in Graph Neural Networks: A Taxonomic Survey
        <https://arxiv.org/abs/2012.15445>`_ for details.
    """

    drop_probability = ori_probs - important_probs

    return drop_probability.mean().item()


def sparsity(coalition: list, data: Data, subgraph_building_method='zero_filling'):
    if subgraph_building_method == 'zero_filling':
        return 1.0 - len(coalition) / data.num_nodes

    elif subgraph_building_method == 'split':
        row, col = data.edge_index
        node_mask = torch.zeros(data.x.shape[0])
        node_mask[coalition] = 1.0
        edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
        return 1.0 - edge_mask.sum() / edge_mask.shape[0]


class XCollector:
    r"""
    XCollector is a data collector which takes processed related prediction probabilities to calculate Fidelity+
    and Fidelity-.

    Args:
        sparsity (float): The Sparsity is use to transform the soft mask to a hard one.

    .. note::
        For more examples, see `benchmarks/xgraph
        <https://github.com/divelab/DIG/tree/dig/benchmarks/xgraph>`_.

    """

    def __init__(self, sparsity=None):
        self.__related_preds, self.__targets = {'zero': [], 'masked': [], 'maskout': [], 'origin': [],
                                                'sparsity': []}, []
        self.masks: Union[List, List[List[Tensor]]] = []

        self.__sparsity = sparsity
        self.__fidelity, self.__fidelity_inv = None, None
        self.__score = None

    @property
    def targets(self) -> list:
        return self.__targets

    def new(self):
        r"""
        Clear class members.
        """
        self.__related_preds, self.__targets = {'zero': [], 'masked': [], 'maskout': [], 'origin': []}, []
        self.masks: Union[List, List[List[Tensor]]] = []

        self.__fidelity, self.__fidelity_inv = None, None

    def collect_data(self,
                     masks: List[Tensor],
                     related_preds: dir,
                     label: int = 0) -> None:
        r"""
        The function is used to collect related data. After collection, we can call fidelity, fidelity_inv, sparsity
        to calculate their values.

        Args:
            masks (list): It is a list of edge-level explanation for each class.
            related_preds (list): It is a list of dictionary for each class where each dictionary
            includes 4 type predicted probabilities and sparsity.
            label (int): The ground truth label. (default: 0)

        """

        if self.__fidelity or self.__fidelity_inv:
            self.__fidelity, self.__fidelity_inv = None, None
            print(f'#W#Called collect_data() after calculate explainable metrics.')

        if not np.isnan(label):
            for key, value in related_preds[label].items():
                self.__related_preds[key].append(value)
            for key in self.__related_preds.keys():
                if key not in related_preds[0].keys():
                    self.__related_preds[key].append(None)
            self.__targets.append(label)
            self.masks.append(masks)

    @property
    def fidelity(self):
        r"""
        Return the Fidelity+ value according to collected data.

        .. note::
            Please refer to `Explainability in Graph Neural Networks: A Taxonomic Survey
            <https://arxiv.org/abs/2012.15445>`_ for details.
        """
        if self.__fidelity:
            return self.__fidelity
        elif None in self.__related_preds['maskout'] or None in self.__related_preds['origin']:
            return None
        else:
            mask_out_preds, one_mask_preds = \
                torch.tensor(self.__related_preds['maskout']), torch.tensor(self.__related_preds['origin'])

            self.__fidelity = fidelity(one_mask_preds, mask_out_preds)
            return self.__fidelity

    @property
    def fidelity_inv(self):
        r"""
        Return the Fidelity- value according to collected data.

        .. note::
            Please refer to `Explainability in Graph Neural Networks: A Taxonomic Survey
            <https://arxiv.org/abs/2012.15445>`_ for details.
        """
        if self.__fidelity_inv:
            return self.__fidelity_inv
        elif None in self.__related_preds['masked'] or None in self.__related_preds['origin']:
            return None
        else:
            masked_preds, one_mask_preds = \
                torch.tensor(self.__related_preds['masked']), torch.tensor(self.__related_preds['origin'])

            self.__fidelity_inv = fidelity_inv(one_mask_preds, masked_preds)
            return self.__fidelity_inv

    @property
    def sparsity(self):
        r"""
        Return the Sparsity value.
        """
        if self.__sparsity:
            return self.__sparsity
        elif None in self.__related_preds['sparsity']:
            return None
        else:
            return torch.tensor(self.__related_preds['sparsity']).mean().item()


class ExplanationProcessor(nn.Module):
    r"""
    Explanation Processor is edge mask explanation processor which can handle sparsity control and use
    data collector automatically.

    Args:
        model (torch.nn.Module): The target model prepared to explain.
        device (torch.device): Specify running device: CPU or CUDA.

    """

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__()
        self.edge_mask = None
        self.model = model
        self.device = device
        self.mp_layers = [module for module in self.model.modules() if isinstance(module, MessagePassing)]
        self.num_layers = len(self.mp_layers)

    class connect_mask(object):

        def __init__(self, cls):
            self.cls = cls

        def __enter__(self):

            self.cls.edge_mask = [
                nn.Parameter(torch.randn(self.cls.x_batch_size * (self.cls.num_edges + self.cls.num_nodes))) for _ in
                range(self.cls.num_layers)] if hasattr(self.cls, 'x_batch_size') else \
                [nn.Parameter(torch.randn(1 * (self.cls.num_edges + self.cls.num_nodes))) for _ in
                 range(self.cls.num_layers)]

            for idx, module in enumerate(self.cls.mp_layers):
                module.__explain__ = True
                module.__edge_mask__ = self.cls.edge_mask[idx]

        def __exit__(self, *args):
            for idx, module in enumerate(self.cls.mp_layers):
                module.__explain__ = False

    def eval_related_pred(self, x: torch.Tensor, edge_index: torch.Tensor, masks: List[torch.Tensor], **kwargs):

        node_idx = kwargs.get('node_idx')
        node_idx = 0 if node_idx is None else node_idx  # graph level: 0, node level: node_idx

        related_preds = []

        for label, mask in enumerate(masks):
            # origin pred
            for edge_mask in self.edge_mask:
                edge_mask.data = float('inf') * torch.ones(mask.size(), device=self.device)
            ori_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            for edge_mask in self.edge_mask:
                edge_mask.data = mask
            masked_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # mask out important elements for fidelity calculation
            for edge_mask in self.edge_mask:
                edge_mask.data = - mask
            maskout_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # zero_mask
            for edge_mask in self.edge_mask:
                edge_mask.data = - float('inf') * torch.ones(mask.size(), device=self.device)
            zero_mask_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # Store related predictions for further evaluation.
            related_preds.append({'zero': zero_mask_pred[node_idx],
                                  'masked': masked_pred[node_idx],
                                  'maskout': maskout_pred[node_idx],
                                  'origin': ori_pred[node_idx]})

            # Adding proper activation function to the models' outputs.
            related_preds[label] = {key: pred.softmax(0)[label].item()
                                    for key, pred in related_preds[label].items()}

        return related_preds

    def forward(self, data: Data, masks: List[torch.Tensor], x_collector: XCollector, **kwargs):
        r"""
        Please refer to the main function in `metric.py`.
        """

        data.to(self.device)
        node_idx = kwargs.get('node_idx')
        y_idx = 0 if node_idx is None else node_idx

        assert not torch.isnan(data.y[y_idx].squeeze())

        self.num_edges = data.edge_index.shape[1]
        self.num_nodes = data.x.shape[0]

        with torch.no_grad():
            with self.connect_mask(self):
                related_preds = self.eval_related_pred(data.x, data.edge_index, masks, **kwargs)

        x_collector.collect_data(masks,
                                 related_preds,
                                 data.y[y_idx].squeeze().long().item())
