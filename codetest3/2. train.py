from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import random as rd
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data, accuracy, Evaluating_Indicator, write_result_to_file, write_matrix_to_file
from model import GCN, GAT, HybridGCN

# 设置随机种子
seed_rand = rd.randint(1, 200)
print("Random seed:", seed_rand)

# 训练参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=seed_rand, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--model', type=str, default='hybrid',
                    help='Model type: gcn, gat or hybrid')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='Alpha for the leaky_relu in attention.')
parser.add_argument('--nheads', type=int, default=2,
                    help='Number of attention heads for GAT.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 设置随机种子
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    if epoch == args.epochs - 1:
        labels_train = labels[idx_train]
        output_train = output[idx_train]
        tn, fp, fn, tp, acc, recall, precision, F1, auc = Evaluating_Indicator(
            labels_train.tolist(),
            output_train.max(1)[1].type_as(labels).tolist()
        )
        return tn, fp, fn, tp, acc, recall, precision, F1, auc


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    labels_test = labels[idx_test]
    output_test = output[idx_test]

    print(output_test)  # 输出输出
    print(output_test.max(1)[1])  # 输出输出

    tn, fp, fn, tp, acc, recall, precision, F1, auc = Evaluating_Indicator(
        labels_test.tolist(),
        output_test.max(1)[1].type_as(labels).tolist()
    )
    return tn, fp, fn, tp, acc, recall, precision, F1, auc


# 训练主循环
t_total = time.time()
for file_par in range(2, 3):
    acc_train_total, recall_train_total, precision_train_total, F1_train_total, auc_train_total = 0, 0, 0, 0, 0
    acc_test_total, recall_test_total, precision_test_total, F1_test_total, auc_test_total = 0, 0, 0, 0, 0

    for parti in range(0, 5):
        # 加载数据
        adj, features, labels, idx_train, idx_val, idx_test = load_data(file_par, parti)

        # 初始化模型
        if args.model == 'gcn':
            model = GCN(nfeat=features.shape[1],
                        nhid=args.hidden,
                        nclass=labels.max().item() + 1,
                        dropout=args.dropout)
        elif args.model == 'gat':
            model = GAT(nfeat=features.shape[1],
                        nhid=args.hidden,
                        nclass=labels.max().item() + 1,
                        dropout=args.dropout,
                        alpha=args.alpha,
                        nheads=args.nheads)
        elif args.model == 'hybrid':
            model = HybridGCN(nfeat=features.shape[1],
                              nhid=args.hidden,
                              nclass=labels.max().item() + 1,
                              dropout=args.dropout)

        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)

        if args.cuda:
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()

        # 训练模型
        for epoch in range(args.epochs):
            rs = train(epoch)
            if rs is not None:
                tn, fp, fn, tp, acc_train, recall_train, precision_train, F1_train, auc_train = rs
                write_result_to_file(acc_train, recall_train, precision_train, F1_train, auc_train)
                write_matrix_to_file(tn, fp, fn, tp)

        # 测试模型
        tn, fp, fn, tp, acc_test, recall_test, precision_test, F1_test, auc_test = test()
        write_result_to_file(acc_test, recall_test, precision_test, F1_test, auc_test)
        write_matrix_to_file(tn, fp, fn, tp)

# 保存模型
torch.save(model.state_dict(), f'model_{args.model}.pth')
print(f"Model saved to model_{args.model}.pth")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))