import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn import metrics
from sklearn.model_selection import KFold
import random
import torch.nn.functional as F
from torch_geometric.utils import degree


def Evaluating_Indicator(y_true, y_pred):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 如果是二分类问题（混淆矩阵是 2x2）
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()  # 解包为四个值
        acc = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        F1 = 2 * (precision * recall) / (precision + recall)
        auc = roc_auc_score(y_true, y_pred)
        return tn, fp, fn, tp, acc, recall, precision, F1, auc

    # 如果是多分类问题（混淆矩阵是 n_classes x n_classes）
    else:
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        auc = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')  # AUC for multi-class
        return acc, recall, precision, F1, auc


def write_all_results_to_file(train_metrics, test_metrics, fold):
    """保存训练集、测试集指标到文件"""
    with open('results_all.csv', 'a') as fw:
        fw.write(f"Fold {fold},Type,TN,FP,FN,TP,Accuracy,Recall,Precision,F1,AUC\n")
        fw.write(f"Fold {fold},Train,{','.join(map(str, train_metrics))}\n")
        fw.write(f"Fold {fold},Test,{','.join(map(str, test_metrics))}\n")


def encode_onehot(labels):
    """标签编码为 one-hot"""
    classes = list(set(labels))
    classes.sort(key=list(labels).index, reverse=True)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path, dataset, fold=0):
    """加载数据并使用10折交叉验证划分数据集"""
    # 加载特征和标签
    content_file = f"{path}{dataset}.content"
    cites_file = f"{path}{dataset}.cites"

    idx_features_labels = np.genfromtxt(content_file, dtype=np.dtype(str), encoding='gbk')
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # 构建图结构
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.genfromtxt(cites_file, dtype=np.int32, encoding='gbk')
    edges = np.array(list(map(idx_map.get, edges.flatten())), dtype=np.int32).reshape(edges.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 添加拓扑特征
    features = add_topological_features(features, adj)

    # 归一化
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # 10折交叉验证划分
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_indices = list(kf.split(np.arange(features.shape[0])))

    # 获取当前fold的索引
    train_indices, test_indices = all_indices[fold]

    # 转换为 PyTorch 张量
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    idx_train = torch.LongTensor(train_indices)
    idx_test = torch.LongTensor(test_indices)

    return adj, features, labels, idx_train, idx_test


def add_topological_features(features, adj):
    """添加图拓扑特征"""
    # 计算节点度
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj)
    deg = degree(adj_tensor._indices()[0], adj_tensor.size(0))

    # 计算聚类系数 (简化版)
    adj_dense = adj_tensor.to_dense()
    triangles = torch.diag(adj_dense @ adj_dense @ adj_dense)
    clustering = triangles / (deg * (deg - 1) + 1e-10)

    # 拼接原始特征和拓扑特征
    topo_feats = torch.stack([deg, clustering], dim=1).numpy()
    features = sp.hstack([features, topo_feats], format='csr')

    return features


def normalize(mx):
    """行归一化稀疏矩阵"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum + 1e-10, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将 scipy 稀疏矩阵转换为 PyTorch 稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data).float()
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def accuracy(output, labels):
    """计算准确率"""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


def graph_data_augment(adj, features, p_drop=0.2, p_add=0.1):
    """图数据增强：随机边操作+特征掩码"""
    # 边丢弃
    if p_drop > 0:
        adj = edge_dropout(adj, p_drop)

    # 边添加
    if p_add > 0:
        adj = edge_add(adj, p_add)

    # 特征掩码
    features = feature_masking(features, p=0.2)

    return adj, features


def edge_dropout(adj, p):
    """随机边丢弃"""
    adj = adj.tocoo()
    mask = np.random.binomial(1, 1 - p, size=adj.nnz)
    new_data = adj.data * mask
    return sp.coo_matrix((new_data, (adj.row, adj.col)), shape=adj.shape)


def edge_add(adj, p):
    """随机边添加"""
    n_nodes = adj.shape[0]
    n_add = int(n_nodes * p)
    new_edges = np.random.randint(0, n_nodes, size=(2, n_add))
    adj = adj.tolil()
    for i in range(n_add):
        adj[new_edges[0, i], new_edges[1, i]] = 1
        adj[new_edges[1, i], new_edges[0, i]] = 1
    return adj.tocoo()


def feature_masking(features, p=0.2):
    """随机特征掩码"""
    mask = (torch.rand(features.size()) > p).float()
    return features * mask