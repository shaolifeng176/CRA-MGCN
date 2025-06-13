import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module



class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
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
        input = input.float()
        support = torch.mm(input, self.weight.float())

        if adj.is_sparse:
            adj = adj.float()
            output = torch.spmm(adj, support)
        else:
            output = torch.mm(adj.float(), support)

        if self.bias is not None:
            return output + self.bias.float()
        return output

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_features} -> {self.out_features})'


class MixedOrderGCN(Module):
    def __init__(self, in_features, out_features, orders=[1, 2]):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphConvolution(in_features, out_features)
            for _ in orders
        ])
        self.weights = nn.Parameter(torch.ones(len(orders)))

    def forward(self, x, adj):
        outputs = []
        adj_power = adj
        for i, layer in enumerate(self.layers):
            outputs.append(layer(x, adj_power))
            adj_power = torch.spmm(adj_power, adj)
        weighted = sum(w * out for w, out in zip(self.weights.softmax(dim=0), outputs))
        return weighted