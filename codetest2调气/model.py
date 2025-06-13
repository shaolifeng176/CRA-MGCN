import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, MixedOrderGCN
from attention import CrossRegionAttentionWrapper


class FeatureExtractor(nn.Module):#  # 特征提取模块
    def __init__(self, in_features, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)


class GatedAttention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x, attention_output):
        gate_score = self.gate(x)
        return gate_score * attention_output + (1 - gate_score) * x

class GCNWithAttention(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, n_heads=4):
        super().__init__()
        # 特征提取模块
        self.feature_extractor = FeatureExtractor(nfeat, nhid)
        self.gated_attn = GatedAttention(nhid)
        # GCN层
        self.gc1 = MixedOrderGCN(nhid, nhid*2, orders=[1, 2])  # 第一层图卷积
        self.gc2 = GraphConvolution(nhid * 2, nhid)  # 第二层图卷积
        self.gc3 = GraphConvolution(nhid, nhid)  # 输出层（这里输出是 nhid）

        # 残差连接
        self.residual = nn.Linear(nhid, nhid)  # 残差连接
        self.aux_classifier = nn.Sequential(
            nn.Linear(nhid, nhid // 2),  # 辅助分类器
            nn.ReLU(),
            nn.Linear(nhid // 2, 2)  # 辅助二分类任务
        )
        # 注意力模块
        self.crmsa = CrossRegionAttentionWrapper(
            nhid,
            n_heads=n_heads,
            dropout=dropout,
            region_size=8
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout * 0.5)

    def forward(self, x, adj):
        x = x.float()

        # 特征提取
        x = self.feature_extractor(x)

        # 保存残差连接
        residual = self.residual(x) if self.residual is not None else None

        # 第一层GCN
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)

        # 第二层GCN
        x = F.relu(self.gc2(x, adj))
        if residual is not None:
            x = x + residual[:x.size(0)]

        # 注意力处理
        x_trans = x.unsqueeze(0).float()
        x_trans, _ = self.crmsa(x_trans)
        x_trans = self.gated_attn(x.squeeze(0), x_trans)  # 在注意力后添加门控
        x = x + self.attention_dropout(x_trans.squeeze(0))

        # 输出层
        x = self.gc3(x, adj)  # 确保输出是 nhid 的大小

        # 确保 aux_classifier 接收的输入是 nhid 的大小
        aux_out = self.aux_classifier(x)  # 辅助输出
        main_out = self.gc3(x, adj)  # 这行可以重复执行，或者根据需要调整

        return F.log_softmax(main_out, dim=1), F.log_softmax(aux_out, dim=1)

