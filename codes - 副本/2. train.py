from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random as rd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve
from utils import load_data, accuracy, Evaluating_Indicator, write_result_to_file, write_matrix_to_file, write_roc_data_to_file
from model import GCN

seed_rand = rd.randint(1, 200)  # 随机种子
print(seed_rand)  # 随机种子

# Training settings
parser = argparse.ArgumentParser()  # 参数解析器
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=seed_rand, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0006,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,  # 30 256 2n
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()  # 解析参数
args.cuda = not args.no_cuda and torch.cuda.is_available()  # 是否使用 GPU

np.random.seed(args.seed)  # 设置随机种子
torch.manual_seed(args.seed)  # 设置随机种子
if args.cuda:  # 如果使用 GPU
    torch.cuda.manual_seed(args.seed)  # 设置随机种子


def train(epoch):  # 训练
    t = time.time()  # 记录开始时间
    model.train()  # 将模型设置为训练模式
    optimizer.zero_grad()  # 清空梯度
    output = model(features, adj)  # 前向传播
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])  # 计算训练集上的损失
    acc_train = accuracy(output[idx_train], labels[idx_train])  # 计算训练集上的准确率
    loss_train.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if not args.fastmode:  # 如果不使用快速模式
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()  # 将模型设置为评估模式
        output = model(features, adj)  # 前向传播
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])  # 计算验证集上的损失
    acc_val = accuracy(output[idx_val], labels[idx_val])  # 计算验证集上的准确率

    '''
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    '''
    if epoch == 499:  # 如果使用快速模式
        labels_train = labels[idx_train]  # 获取训练集的标签
        output_train = output[idx_train]  # 获取训练集的输出
        tn, fp, fn, tp, acc, recall, precision, F1, auc = Evaluating_Indicator(labels_train.tolist(),
                                                                               output_train.max(1)[1].type_as(
                                                                                   labels).tolist())
        # write_result_to_file(acc,recall,precision,F1,auc)
        return tn, fp, fn, tp, acc, recall, precision, F1, auc


def test():  # 测试
    model.eval()  # 将模型设置为评估模式
    output = model(features, adj)  # 前向传播
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])  # 计算测试
    acc_test = accuracy(output[idx_test], labels[idx_test])  # 计算测试集上的准确率
    labels_test = labels[idx_test]  # 获取测试集的标签
    output_test = output[idx_test]  # 获取测试集的输出
    print(output_test)  # 输出输出
    print(output_test.max(1)[1])  # 输出输出
    tn, fp, fn, tp, acc, recall, precision, F1, auc = Evaluating_Indicator(labels_test.tolist(),
                                                                           output_test.max(1)[1].type_as(
                                                                               labels).tolist())  # 计算指标
    # write_result_to_file(acc,recall,precision,F1,auc)
    '''
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    '''
    y_score = F.softmax(output_test, dim=1)[:, 1].detach().cpu().numpy()
    y_true = labels_test.detach().cpu().numpy()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return tn, fp, fn, tp, acc, recall, precision, F1, auc, fpr, tpr


t_total = time.time()  # 记录开始时间
all_fpr = []
all_tpr = []
for file_par in range(2, 3):  # 循环文件
    acc_train_total, recall_train_total, precision_train_total, F1_train_total, auc_train_total = 0, 0, 0, 0, 0  # 初始化指标
    acc_test_total, recall_test_total, precision_test_total, F1_test_total, auc_test_total = 0, 0, 0, 0, 0  # 初始化

    for parti in range(0, 5):  # 循环部分
        # Load data
        adj, features, labels, idx_train, idx_val, idx_test = load_data(file_par, parti)  # 加载数据

        # Train model
        # Model and optimizer
        model = GCN(nfeat=features.shape[1],  # 特征维度
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,  # 类别数
                    dropout=args.dropout)  # 模型
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)  # 优化器
        if args.cuda:  # 如果使用 GPU
            model.cuda()  # 将模型加载到 GPU
            features = features.cuda()  # 将特征加载到 GPU
            adj = adj.cuda()  # 将邻接矩阵加载到 GPU
            labels = labels.cuda()  # 将标签加载到 GPU
            idx_train = idx_train.cuda()  # 将训练集索引加载到 GPU
            idx_val = idx_val.cuda()  # 将验证集索引加载到 GPU
            idx_test = idx_test.cuda()  # 将测试集索引加载到 GPU

        for epoch in range(args.epochs):  # 循环训练次数
            rs = train(epoch)  # 训练
            if rs is not None:  # 如果使用快速模式
                tn, fp, fn, tp, acc_train, recall_train, precision_train, F1_train, auc_train = rs  # 获取指标
                write_result_to_file(acc_train, recall_train, precision_train, F1_train, auc_train)  # 写入指标
                write_matrix_to_file(tn, fp, fn, tp)  # 写入矩阵
                '''
                acc_train_total += acc_train
                recall_train_total += recall_train
                precision_train_total += precision_train
                F1_train_total += F1_train
                auc_train_total += auc_train
                '''
        '''
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        '''
        # Testing
        tn, fp, fn, tp, acc_test, recall_test, precision_test, F1_test, auc_test, fpr, tpr = test()  # 测试
        write_result_to_file(acc_test, recall_test, precision_test, F1_test, auc_test)  # 写入指标
        write_matrix_to_file(tn, fp, fn, tp)  # 写入矩阵
        all_fpr.append(fpr)
        all_tpr.append(tpr)

# 计算平均 ROC 曲线
mean_fpr = np.linspace(0, 1, 200)
mean_tpr = np.zeros_like(mean_fpr)
for i in range(len(all_fpr)):
    mean_tpr += np.interp(mean_fpr, all_fpr[i], all_tpr[i])
mean_tpr /= len(all_fpr)

# 保存平均 ROC 曲线数据到 CSV 文件
write_roc_data_to_file("GCN", mean_fpr, mean_tpr)

# Save the model after training
torch.save(model.state_dict(), 'model.pth')  # 保存模型
print("Model saved to model.pth")  # 打印信息