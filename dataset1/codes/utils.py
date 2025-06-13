import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def Evaluating_Indicator(y_true,y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = metrics.accuracy_score(y_true,y_pred)
    recall = metrics.recall_score(y_true,y_pred)
    precision = metrics.precision_score(y_true,y_pred)
    F1 = metrics.f1_score(y_true,y_pred)
    auc = metrics.roc_auc_score(y_true,y_pred)
    return  tn, fp, fn, tp,acc,recall,precision,F1,auc

def write_result_to_file(acc,recall,precision,F1,auc):
    with open('rs.csv','a') as fw:
        fw.write(str(acc))
        fw.write(',')
        fw.write(str(recall))
        fw.write(',')
        fw.write(str(precision))
        fw.write(',')
        fw.write(str(F1))
        fw.write(',')
        fw.write(str(auc))
        fw.write('\n')

def write_matrix_to_file(tn, fp, fn, tp):
    with open('matrix.csv','a') as fw:
        fw.write(str(tn))
        fw.write(',')
        fw.write(str(fp))
        fw.write(',')
        fw.write(str(fn))
        fw.write(',')
        fw.write(str(tp))
        fw.write('\n')


def encode_onehot(labels):
    classes = list(set(labels))
    classes.sort(key = list(labels).index,reverse=True)
    #print(classes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(file_par, parti, path="D:\\Project\\pythonProject\\re\\keti\\dataset1\\dataset\\training data\\", dataset="HCGCN-all herb pairs"):

    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str), encoding='utf-8')
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    #dataset = dataset + str(file_par)
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    '''
    part1 = list(range(0,64))
    part2 = list(range(64,128))
    part3 = list(range(128,192))
    part4 = list(range(192,318))
    part5 = list(range(255,318))
    '''
    #part5 = list(range(318,568))


    part1 = list(range(0,32))
    part2 = list(range(32,64))
    part3 = list(range(64,86))
    part4 = list(range(96,132))
    part5 = list(range(132,164))

    partlist = [part1 , part2 , part3 , part4 , part5]

    test_part = partlist[parti]
    idx_train = []
    for t in partlist:
        if t!= test_part:
            idx_train = idx_train + t
    idx_val = range(0, 0)
    idx_test = test_part

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum + 1e-10, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    #print('output',output.max(1)[1])
    #print('labels',labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)