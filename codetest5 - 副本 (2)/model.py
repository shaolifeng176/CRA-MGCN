import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import HierarchicalGraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = HierarchicalGraphConvolution(nfeat, nhid)
        self.gc2 = HierarchicalGraphConvolution(nhid, nclass)
        self.dropout = dropout

    def get_attention_weights(self):
        """安全获取注意力权重的方法"""
        with torch.no_grad():
            weights = {
                'node_layer1': self.gc1.node_att.weight.detach().cpu(),
                'semantic_layer1': self.gc1.semantic_att.weight.detach().cpu(),
                'node_layer2': self.gc2.node_att.weight.detach().cpu(),
                'semantic_layer2': self.gc2.semantic_att.weight.detach().cpu()
            }
            return weights

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)