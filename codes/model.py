import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):# GCN model
    def __init__(self, nfeat, nhid, nclass, dropout):# nfeat: 输入特征数，nhid: 隐层特征数，nclass: 输出特征数，dropout: dropout概率
        super(GCN, self).__init__()# 调用父类的初始化方法

        self.gc1 = GraphConvolution(nfeat, nhid)# 定义两个图卷积层，分别输入特征数和隐层特征数
        self.gc2 = GraphConvolution(nhid, nclass)#
        #self.gc3 = GraphConvolution(16, nclass)

        self.dropout = dropout# 定义dropout概率

    def forward(self, x, adj):# 定义前向传播函数，输入x和adj
        x = F.relu(self.gc1(x, adj))# 通过第一个图卷积层，然后进行relu激活函数，然后进行dropout
        x = F.dropout(x, self.dropout, training=self.training)# 通过第二个图卷积层，然后进行dropout

        #x = F.relu(self.gc2(x, adj ))
        #x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc2(x, adj) #通过第三个图卷积层
        return F.log_softmax(x, dim=1) #返回log_softmax激活函数后的输出

