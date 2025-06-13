#GAT,GCN串行
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from layers import GraphConvolution, GraphAttentionLayer
#
#
# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return F.log_softmax(x, dim=1)
#
#
# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha=0.2, nheads=2):
#         """Dense version of GAT."""
#         super(GAT, self).__init__()
#         self.dropout = dropout
#
#         # 多头注意力
#         self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
#                            for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#
#         # 输出层
#         self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout,
#                                            alpha=alpha, concat=False)
#
#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return F.log_softmax(x, dim=1)
#
#
# class HybridGCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(HybridGCN, self).__init__()
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gat1 = GraphAttentionLayer(nhid, nhid, dropout)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         # 确保输入是合并后的稀疏张量
#         if adj.is_sparse:
#             adj = adj.coalesce()
#
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#
#         # 将邻接矩阵转换为密集格式供GAT使用
#         adj_dense = adj.to_dense() if adj.is_sparse else adj
#         x = self.gat1(x, adj_dense)
#
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return F.log_softmax(x, dim=1)
#GAT,GAN并行
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphAttentionLayer


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha=0.2, nheads=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        # Multi-head attention
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # Output layer
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout,
                                           alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class HybridGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha=0.2, nheads=2):
        """并行融合GCN和GAT的混合模型

        参数:
            nfeat: 输入特征维度
            nhid: 隐藏层维度
            nclass: 输出类别数
            dropout: dropout概率
            alpha: LeakyReLU的负斜率
            nheads: 多头注意力机制的头数
        """
        super(HybridGCN, self).__init__()
        self.dropout = dropout

        # GCN分支
        self.gcn1 = GraphConvolution(nfeat, nhid)
        self.gcn2 = GraphConvolution(nhid, nclass)

        # GAT分支
        self.gat_attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout,
                                                   alpha=alpha, concat=True)
                               for _ in range(nheads)]
        for i, attention in enumerate(self.gat_attentions):
            self.add_module(f'gat_attention_{i}', attention)
        self.gat_out = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout,
                                           alpha=alpha, concat=False)

        # 输出投影层，融合两个分支的结果
        self.projection = nn.Linear(2 * nclass, nclass)

    def forward(self, x, adj):
        # 确保稀疏邻接矩阵是合并后的
        if adj.is_sparse:
            adj = adj.coalesce()

        # 为GAT准备稠密邻接矩阵
        adj_dense = adj.to_dense() if adj.is_sparse else adj

        # GCN分支
        x_gcn = F.relu(self.gcn1(x, adj))
        x_gcn = F.dropout(x_gcn, self.dropout, training=self.training)
        x_gcn = self.gcn2(x_gcn, adj)

        # GAT分支
        x_gat = F.dropout(x, self.dropout, training=self.training)
        x_gat = torch.cat([att(x_gat, adj_dense) for att in self.gat_attentions], dim=1)
        x_gat = F.dropout(x_gat, self.dropout, training=self.training)
        x_gat = F.elu(self.gat_out(x_gat, adj_dense))

        # 并行融合两个分支的结果
        x_combined = torch.cat([x_gcn, x_gat], dim=1)
        x_combined = self.projection(x_combined)

        return F.log_softmax(x_combined, dim=1)