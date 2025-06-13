from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random as rd
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, Evaluating_Indicator,write_result_to_file,write_matrix_to_file

from model import GCN
seed_rand = rd.randint(1,200)
print(seed_rand)
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=seed_rand, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0006,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,#30 256 2n
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

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
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    '''
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    '''
    if epoch == 499:
        labels_train = labels[idx_train]
        output_train = output[idx_train]
        tn, fp, fn, tp, acc,recall,precision,F1,auc = Evaluating_Indicator(labels_train.tolist(), output_train.max(1)[1].type_as(labels).tolist())
        #write_result_to_file(acc,recall,precision,F1,auc)
        return  tn, fp, fn, tp, acc,recall,precision,F1,auc


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    labels_test = labels[idx_test]
    output_test = output[idx_test]
    print(output_test)
    print(output_test.max(1)[1])
    tn, fp, fn, tp,acc,recall,precision,F1,auc = Evaluating_Indicator(labels_test.tolist(), output_test.max(1)[1].type_as(labels).tolist())
    #write_result_to_file(acc,recall,precision,F1,auc)
    '''
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    '''
    return tn, fp, fn, tp, acc,recall,precision,F1,auc

t_total = time.time()
for file_par in range(2,3):
    acc_train_total, recall_train_total, precision_train_total, F1_train_total, auc_train_total = 0, 0, 0, 0, 0
    acc_test_total, recall_test_total, precision_test_total, F1_test_total, auc_test_total = 0, 0, 0, 0, 0

    for parti in range(0,5):
        # Load data
        adj, features, labels, idx_train, idx_val, idx_test = load_data(file_par, parti)

        # Train model
        # Model and optimizer
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
        if args.cuda:
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()

        for epoch in range(args.epochs):
            rs = train(epoch)
            if rs !=None:
                tn, fp, fn, tp, acc_train, recall_train, precision_train, F1_train, auc_train = rs
                write_result_to_file(acc_train, recall_train, precision_train, F1_train, auc_train)
                write_matrix_to_file(tn, fp, fn, tp)
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
        tn, fp, fn, tp, acc_test, recall_test, precision_test, F1_test, auc_test = test()
        write_result_to_file(acc_test, recall_test, precision_test, F1_test, auc_test)
        write_matrix_to_file(tn, fp, fn, tp)

# Save the model after training
torch.save(model.state_dict(), 'model.pth')
print("Model saved to model.pth")