from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from utils import load_data, accuracy, Evaluating_Indicator, write_all_results_to_file, sparse_mx_to_torch_sparse_tensor
from model import GCNWithAttention
import warnings

warnings.filterwarnings("ignore")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.set_default_dtype(torch.float32)

# 参数配置
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='禁用 CUDA')
parser.add_argument('--epochs', type=int, default=5000, help='训练轮次')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
parser.add_argument('--hidden', type=int, default=256, help='隐藏层维度')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout 概率')#0.5
parser.add_argument('--n_heads', type=int, default=4, help='注意力头数')
parser.add_argument('--gamma', type=float, default=5.0, help='Focal Loss的gamma参数')#2
parser.add_argument('--alpha', type=float, default=0.25, help='Focal Loss的alpha参数')#0.25
parser.add_argument('--label_smoothing', type=float, default=0.1, help='标签平滑系数')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # 标签平滑处理
        n_classes = inputs.size(1)
        targets_onehot = torch.zeros_like(inputs)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
        targets_onehot = targets_onehot * (1 - self.label_smoothing) + self.label_smoothing / n_classes

        # 计算Focal Loss
        log_prob = F.log_softmax(inputs, dim=-1)
        loss = -targets_onehot * log_prob
        loss = loss.sum(dim=-1)

        # Focal Loss权重调整
        pt = torch.exp(-loss)
        loss = self.alpha * (1 - pt) ** self.gamma * loss
        return loss.mean()


def augment_adj(adj, drop_rate=0.1, add_rate=0.05):
    """图数据增强：随机添加/删除边"""
    adj = adj.to_dense()  # 转换为稠密矩阵
    indices = adj.nonzero()# 非零元素的索引

    # 随机删除边
    if drop_rate > 0:
        drop_mask = torch.rand(len(indices[0])) > drop_rate  # 创建一个布尔 mask
        adj[indices[0][~drop_mask], indices[1][~drop_mask]] = 0 # 根据mask删除边

    # 随机添加边
    if add_rate > 0:
        n_nodes = adj.shape[0]
        n_add = int(n_nodes * add_rate)
        new_edges = torch.randint(0, n_nodes, (2, n_add))
        adj[new_edges[0], new_edges[1]] = 1
        adj[new_edges[1], new_edges[0]] = 1  # 无向图

    # 将增强后的邻接矩阵转换为稀疏矩阵，并返回
    adj_sparse = adj.to_sparse()
    return adj_sparse



def train(epoch, model, optimizer, features, adj, labels, idx_train):
    model.train()# 训练模式
    optimizer.zero_grad()# 梯度清零

    # 50%概率使用数据增强
    adj_aug = augment_adj(adj) if random.random() < 0.5 else adj# 数据增强

    # 前向传播
    output = model(features, adj_aug)# 前向传播

    # 如果是多任务学习，取主任务输出
    if isinstance(output, tuple):
        output = output[0]

    # 计算损失
    criterion = FocalLoss(
        alpha=args.alpha,
        gamma=args.gamma,
        label_smoothing=args.label_smoothing
    )
    loss_train = criterion(output[idx_train], labels[idx_train])

    # 计算准确率
    acc_train = accuracy(output[idx_train], labels[idx_train])

    # 反向传播
    loss_train.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 梯度裁剪
    optimizer.step()# 更新参数

    # 打印训练信息
    if epoch % 100 == 0:
        print(f'Epoch: {epoch + 1:04d} | '
              f'Loss Train: {loss_train.item():.4f} | '
              f'Acc Train: {acc_train.item():.4f}')# 打印训练信息

    # 返回训练集指标
    train_pred = output[idx_train].max(1)[1].cpu().numpy()# 获取预测结果
    train_true = labels[idx_train].cpu().numpy()# 获取真实标签
    return Evaluating_Indicator(train_true, train_pred)# 计算指标


def test(model, features, adj, labels, idx_test):
    model.eval()
    output = model(features, adj)

    # 如果是多任务学习，取主任务输出
    if isinstance(output, tuple):
        output = output[0]

    test_pred = output[idx_test].max(1)[1].cpu().numpy()
    test_true = labels[idx_test].cpu().numpy()

    test_metrics = Evaluating_Indicator(test_true, test_pred)

    print("\nTest Results:")
    if len(test_metrics) == 9:  # 二分类问题
        print(f"Accuracy: {test_metrics[4]:.4f} | Recall: {test_metrics[5]:.4f} | "
              f"Precision: {test_metrics[6]:.4f} | F1: {test_metrics[7]:.4f} | AUC: {test_metrics[8]:.4f}")
    else:  # 多分类问题
        print(f"Accuracy: {test_metrics[0]:.4f} | Recall: {test_metrics[1]:.4f} | "
              f"Precision: {test_metrics[2]:.4f} | F1: {test_metrics[3]:.4f} | AUC: {test_metrics[4]:.4f}")

    return test_metrics



if __name__ == "__main__":
    # 10折交叉验证
    for fold in range(10):
        print(f"\n==================== Fold {fold} ====================")

        # 加载数据
        print("Loading data...")
        adj, features, labels, idx_train, idx_test = load_data(
            path=r'C:\project\python\pythonProject\re\keti\codetest2活血\training data\\',
            dataset='HCGCN-blood-activating herb pairs',
            fold=fold
        )
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        # 初始化模型
        model = GCNWithAttention(
            nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            n_heads=args.n_heads
        )
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)

        # 设备转移
        device = torch.device('cuda' if args.cuda else 'cpu')
        model = model.to(device)
        features = features.to(device)
        adj = adj.to(device)
        labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_test = idx_test.to(device)

        # 训练循环
        print("Starting training...")
        best_auc = 0
        for epoch in range(args.epochs):
            train_metrics = train(epoch, model, optimizer, features, adj, labels, idx_train)

            # 每隔100轮验证一次
            if epoch % 100 == 0:
                test_metrics = test(model, features, adj, labels, idx_test)
                if test_metrics[-1] > best_auc:  # 保存最佳AUC模型
                    best_auc = test_metrics[-1]
                    torch.save(model.state_dict(), f'model_fold{fold}_best.pth')

        # 最终测试并保存结果
        model.load_state_dict(torch.load(f'model_fold{fold}_best.pth'))
        test_metrics = test(model, features, adj, labels, idx_test)
        write_all_results_to_file(train_metrics, test_metrics, fold)

        print(f"Fold {fold} completed. Best AUC: {best_auc:.4f}")