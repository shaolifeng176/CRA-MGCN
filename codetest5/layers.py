import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class NodeLevelAttention(Module):
    """节点级注意力层"""

    def __init__(self, in_features):
        super(NodeLevelAttention, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.FloatTensor(in_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        e = torch.matmul(x, self.weight)  # [N, 1]
        return torch.sigmoid(e)  # 使用sigmoid限制到[0,1]


class SemanticLevelAttention(Module):
    """语义级注意力层"""

    def __init__(self, in_features):
        super(SemanticLevelAttention, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.FloatTensor(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return torch.softmax(self.weight, dim=0)  # 特征维度的注意力分布


class HierarchicalGraphConvolution(Module):
    """带分层注意力的图卷积层"""

    def __init__(self, in_features, out_features, bias=True):
        super(HierarchicalGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        # 注意力机制
        self.node_att = NodeLevelAttention(in_features)
        self.semantic_att = SemanticLevelAttention(in_features)

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

    def forward(self, x, adj):
        # 节点级注意力 [N,1]
        node_attention = self.node_att(x)

        # 语义级注意力 [in_features]
        semantic_attention = self.semantic_att(x)

        # 组合注意力
        x_weighted = x * node_attention  # 节点加权
        x_weighted = x_weighted * semantic_attention  # 特征维度加权

        # 图卷积
        support = torch.mm(x_weighted, self.weight)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'