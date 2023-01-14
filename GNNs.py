import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.data.batch import Batch

from typing import Callable, Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import Tensor

from torch_sparse import SparseTensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 由于采用的数据形式是pyg的data类型，因此所有的GNN的数据都要稍微处理一下
class GNNBasic(torch.nn.Module):
    def __init__(self):
        super(GNNBasic, self).__init__()

    def arguments_read(self, *args, **kwargs):

        data: Batch = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index = kwargs['x'], kwargs['edge_index']
                batch = kwargs.get('batch')
                if batch is None:
                    batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=x.device)
            elif len(args) == 2:
                x, edge_index = args[0], args[1]
                batch = torch.zeros(args[0].shape[0], dtype=torch.int64, device=x.device)
            elif len(args):
                x, edge_index, batch = args[0], args[1], args[2]
            else:
                raise ValueError(f"forward's args should take 2 or 3 arguments but got {len(args)}")
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        return x, edge_index, batch


class GNNPool(nn.Module):
    def __init__(self):
        super(GNNPool, self).__init__()


class GlobalMeanPool(GNNPool):
    def __init__(self):
        super(GlobalMeanPool, self).__init__()

    def forward(self, x, batch):
        return gnn.global_mean_pool(x, batch)


class IdenticalPool(GNNPool):
    def __init__(self):
        super(IdenticalPool, self).__init__()

    def forward(self, x, batch):
        return x


# ----------------------------------------------model:2层 GCN
class GCN_2l(GNNBasic):
    def __init__(self, model_level, dim_node, dim_hidden, num_classes):
        super(GCN_2l, self).__init__()
        num_layer = 2

        self.conv1 = GCNConv(dim_node, dim_hidden)
        self.convs = nn.ModuleList(
            [
                GCNConv(dim_hidden, dim_hidden)
                for _ in range(num_layer - 1)
            ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        # *是为了把列表中的值打散
        self.ffn = nn.Sequential(
            *(
                [nn.Linear(dim_hidden, num_classes)]
            )
        )

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.relu1(self.conv1(x, edge_index))

        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))

        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)

        return F.log_softmax(out, dim=1)

    # 没有池化 和最后输出预测的全连接层
    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)

        post_conv = self.relu1(self.conv1(x, edge_index))
        emb = post_conv
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))
            emb += post_conv

        return emb


# GCNConv 继承了 gnn.GCNConv类---------class GCNConv(gnn.GCNConv):
# 而 gnn.GCNConv继承了 MessagePassing--class GCNConv(MessagePassing):
class GCNConv(gnn.GCNConv):
    def __init__(self, *args, **kwargs):
        super(GCNConv, self).__init__(*args, **kwargs)
        self.edge_mask = None
        self.__explain__ = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        # 没有edge_weight，并且要求normalize的情况下 edge_weight会默认全部置1
        if self.normalize and edge_weight is None:
            # 判断edge_index是什么形式 tensor和sparse_tensor是不同的处理方式
            if isinstance(edge_index, Tensor):
                # edge_index,edge_weight 会缓存在self._cached_edge_index中
                cache = self._cached_edge_index
                if cache is None:
                    # edge_weight会默认全部置1，并根据度信息进行nomralize;edge_index增加了自旋边
                    edge_index, edge_weight = gnn.conv.gcn_conv.gcn_norm(edge_index, edge_weight, x.size(self.node_dim),
                                                                         self.improved, self.add_self_loops,
                                                                         dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            # 这个的权重直接记录在edge_index中了
            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gnn.conv.gcn_conv.gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype
                    )

                    if self.cached:
                        self._cached_adj_t = edge_index

                else:
                    edge_index = cache

        # --- add require_grad ---
        # 对邻接矩阵进行考虑了边权重的normalize之后
        edge_weight.requires_grad_(True)

        # x = torch.matmul(x, self.weight)  # 相当于一个线性映射
        x = self.lin(x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        if self.bias is not None:
            out += self.bias

        # --记录edge_weight
        self.edge_weight = edge_weight

        return out

    # 覆写 propagate类
    # size: 邻接节点的数量与中心节点的数量。
    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        # 先确认一下输入的edge_index要么是（2，num_edges）的tensor类型，要么就是sparse_tensor类型
        size = self.__check_input__(edge_index, size)

        if (isinstance(edge_index, SparseTensor) and self.fuse and not self.__explain__):
            # self.message_and_aggregate里是用稀疏张量的形式表示边的
            # self.__collect__返回一个字典，里面是需要的收集到的参数
            # self.__fused_user_args__ 一些fuse里额外的参数
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index, size, kwargs)
            msg_aggr_kwargs = self.inspector.distribute('message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # 其他情况就将message和aggregate分开操作
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)
            msg_kwargs = self.inspector.distribute('message', coll_dict)
            out = self.message(**msg_kwargs)

            # 当要考虑到GNN解释器的时候，要考虑边的掩码edge_mask，因此采用分开的message和aggregate
            if self.__explain__:
                edge_mask = self.edge_mask
                if out.size(self.node_dim) != edge_mask.size(0):  # 说明消息传递的时候增加了self-loop导致边数不一致
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)  # size[0]表示的是中心节点的个数，即节点个数
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)


# ----------------------------------------------model:2层 GIN
class GIN_2l(GNNBasic):

    def __init__(self, model_level, dim_node, dim_hidden, num_classes):
        super().__init__()
        num_layer = 2

        self.conv1 = GINConv(nn.Sequential(nn.Linear(dim_node, dim_hidden), nn.ReLU(),
                                           nn.Linear(dim_hidden, dim_hidden),
                                           nn.ReLU()))  # ,
        # nn.BatchNorm1d(dim_hidden)))
        self.convs = nn.ModuleList(
            [
                GINConv(nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
                                      nn.Linear(dim_hidden, dim_hidden), nn.ReLU()))  # ,
                # nn.BatchNorm1d(dim_hidden)))
                for _ in range(num_layer - 1)
            ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(dim_hidden, dim_hidden)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(dim_hidden, num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)

        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)

        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)
        return F.log_softmax(out, dim=1)

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.conv1(x, edge_index)
        emb = post_conv
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)
            emb += post_conv
        return emb


# 传入待调用的nn是每一层中的mlp，在邻居聚合（propagate）之后调用
class GINConv(gnn.GINConv):

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        super().__init__(nn, eps, train_eps, **kwargs)
        self.edge_weight = None
        self.fc_steps = None
        self.reweight = None
        self.edge_mask = None
        self.__explain__ = None

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, task='explain', **kwargs) -> Tensor:
        """"""
        self.num_nodes = x.shape[0]
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        if edge_weight is not None:
            self.edge_weight = edge_weight
            assert edge_weight.shape[0] == edge_index.shape[1]
            self.reweight = False
        else:
            edge_index, _ = remove_self_loops(edge_index)
            self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
            if self_loop_edge_index.shape[1] != edge_index.shape[1]:
                edge_index = self_loop_edge_index
            self.reweight = True
        out = self.propagate(edge_index, x=x[0], size=None)

        if task == 'explain':
            layer_extractor = []
            hooks = []

            def register_hook(module: nn.Module):
                if not list(module.children()):
                    hooks.append(module.register_forward_hook(forward_hook))

            def forward_hook(module: nn.Module, input: Tuple[Tensor], output: Tensor):
                # input contains x and edge_index
                layer_extractor.append((module, input[0], output))

            # --- register hooks ---
            self.nn.apply(register_hook)

            nn_out = self.nn(out)

            for hook in hooks:
                hook.remove()

            fc_steps = []
            step = {'input': None, 'module': [], 'output': None}
            for layer in layer_extractor:
                if isinstance(layer[0], nn.Linear):
                    if step['module']:
                        fc_steps.append(step)
                    # step = {'input': layer[1], 'module': [], 'output': None}
                    step = {'input': None, 'module': [], 'output': None}
                step['module'].append(layer[0])
                if kwargs.get('probe'):
                    step['output'] = layer[2]
                else:
                    step['output'] = None

            if step['module']:
                fc_steps.append(step)
            self.fc_steps = fc_steps
        else:
            nn_out = self.nn(out)

        return nn_out

    def message(self, x_j: Tensor) -> Tensor:
        if self.reweight:
            edge_weight = torch.ones(x_j.shape[0], device=x_j.device)
            edge_weight.data[-self.num_nodes:] += self.eps
            edge_weight = edge_weight.detach().clone()
            edge_weight.requires_grad_(True)
            self.edge_weight = edge_weight
        return x_j * self.edge_weight.view(-1, 1)

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self.__explain__):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                         kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self.__explain__:
                edge_mask = self.edge_mask
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)


# ----------------------------------------------model:2层 GAT
1


# 2/mine
# class GATConv(gnn.GATConv):
#     def __init__(self, *args, **kwargs):
#         super(GATConv, self).__init__(*args, **kwargs)
#         self.edge_mask = None
#         self.__explain__ = None
#
#     def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
#         size = self.__check_input__(edge_index, size)
#
#         if (isinstance(edge_index, SparseTensor) and self.fuse and not self.__explain__):
#             coll_dict = self.__collect__(self.__fused_user_args__, edge_index, size, kwargs)
#             msg_aggr_kwargs = self.inspector.distribute('message_and_aggregate', coll_dict)
#             out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
#             update_kwargs = self.inspector.distribute('update', coll_dict)
#             return self.update(out, **update_kwargs)
#
#         elif isinstance(edge_index, Tensor) or not self.fuse:
#             coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)
#             msg_kwargs = self.inspector.distribute('message', coll_dict)
#             out = self.message(**msg_kwargs)
#             if self.__explain__:
#                 edge_mask = self.edge_mask
#                 if out.size(self.node_dim) != edge_mask.size(0):  # 说明edge_mask里没有自旋边
#                     loop = edge_mask.new_ones(size[0])  # size[0]==num_nodes
#                     edge_mask = torch.cat([edge_mask, loop], dim=0)
#                 assert out.size(self.node_dim) == edge_mask.size(0)
#                 out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
#
#             aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
#             out = self.aggregate(out, **aggr_kwargs)
#             update_kwargs = self.inspector.distribute('update', coll_dict)
#             return self.update(out, **update_kwargs)

class GATConv(gnn.GATConv):
    def __init__(self, *args, **kwargs):
        super(GATConv, self).__init__(*args, **kwargs)
        self.edge_mask = None
        self.__explain__ = None

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self.__explain__):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                         kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self.__explain__:
                edge_mask = self.edge_mask
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)


# 3/
class GAT(GNNBasic):
    def __init__(self, model_level, input_dim, hidden_dim, num_classes, heads):
        super(GAT, self).__init__()
        num_layers = 2
        dropout = 0.7
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads,
                             dropout=dropout, concat=True)

        self.convs = nn.ModuleList([
            GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True)
            for _ in range(num_layers - 1)
        ])
        self.conv_end = GATConv(hidden_dim * heads, num_classes, dropout=dropout, concat=False)

        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList([
            nn.ReLU()
            for _ in range(num_layers - 1)
        ])
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
            [nn.Linear(hidden_dim * heads, num_classes)]
        ))
        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)

        post_conv = self.relu1(F.normalize(self.conv1(x, edge_index), p=2, dim=-1))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(F.normalize(conv(post_conv, edge_index), p=2, dim=-1))

        out_readout = self.readout(post_conv, batch)
        out = self.ffn(out_readout)
        # out=self.conv_end(out_readout,edge_index)
        return F.log_softmax(out, dim=1)

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.conv1(x, edge_index)
        emb = post_conv
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)
            emb += post_conv
        return emb
