import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import DualGraphConvolution  # 修改为导入新的DualGraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        # 使用DualGraphConvolution替代原来的GraphConvolution
        self.gc1 = DualGraphConvolution(nfeat, nhid)
        self.gc2 = DualGraphConvolution(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, adj):
        # 第一层带Dual Attention的图卷积
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        # 第二层带Dual Attention的图卷积
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)