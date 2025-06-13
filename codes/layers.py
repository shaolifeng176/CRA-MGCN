import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features#节点特征向量维度
        self.out_features = out_features#输出特征向量维度
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))#节点特征向量维度*输出特征向量维度
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))#输出特征向量维度
        else:
            self.register_parameter('bias', None)#注册参数
        self.reset_parameters()#初始化权重和偏置

    def reset_parameters(self):#初始化权重和偏置
        stdv = 1. / math.sqrt(self.weight.size(1))#计算标准差
        self.weight.data.uniform_(-stdv, stdv)#初始化
        if self.bias is not None:#如果偏置不为空
            self.bias.data.uniform_(-stdv, stdv)#初始化

    def forward(self, input, adj):#前向传播
        support = torch.mm(input, self.weight)#节点特征向量维度*输出特征向量维度
        output = torch.spmm(adj, support)#邻接矩阵*节点特征向量维度*输出特征向量维度
        if self.bias is not None:#如果偏置不为空
            return output + self.bias#返回输出
        else:
            return output#返回输出

    def __repr__(self):#返回类名
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'#返回类名
