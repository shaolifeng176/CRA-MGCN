import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class NodeAttention(Module):
    """节点注意力层"""

    def __init__(self, in_features):
        super(NodeAttention, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.FloatTensor(in_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x shape: [N, in_features]
        e = torch.matmul(x, self.weight)  # [N, 1]
        attention = F.softmax(e, dim=0)  # 节点级注意力权重
        return attention


class EdgeAttention(Module):
    """边注意力层"""

    def __init__(self, in_features):
        super(EdgeAttention, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.FloatTensor(in_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # x shape: [N, in_features]
        # adj shape: [N, N] (稀疏张量)

        # 将稀疏邻接矩阵转换为密集矩阵
        adj_dense = adj.to_dense() if adj.is_sparse else adj

        h = torch.matmul(x, self.weight)  # [N, in_features]
        e = torch.matmul(h, h.T)  # [N, N] 边注意力分数
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_dense > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)  # 边注意力权重
        return attention


class DualGraphConvolution(Module):
    """带Dual Attention的图卷积层"""

    def __init__(self, in_features, out_features, bias=True):
        super(DualGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        # 添加注意力机制
        self.node_attention = NodeAttention(in_features)
        self.edge_attention = EdgeAttention(in_features)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # 计算节点注意力
        node_att = self.node_attention(input)  # [N, 1]

        # 计算边注意力
        edge_att = self.edge_attention(input, adj)  # [N, N]

        # 组合注意力
        combined_att = node_att * edge_att  # [N, N]

        # 应用注意力
        support = torch.mm(input, self.weight)

        # 处理稀疏邻接矩阵
        if adj.is_sparse:
            # 将注意力权重转换为稀疏格式
            adj_dense = adj.to_dense()
            weighted_adj = adj_dense * combined_att
            # 转换回稀疏格式以提高效率
            weighted_adj_sparse = weighted_adj.to_sparse()
            output = torch.spmm(weighted_adj_sparse, support)
        else:
            output = torch.spmm(adj * combined_att, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'