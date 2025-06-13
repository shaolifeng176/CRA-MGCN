import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def encode_onehot(labels):
    """将标签转换为one-hot编码"""
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """行归一化稀疏矩阵"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum + 1e-10, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将scipy稀疏矩阵转换为torch稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def load_data(file_par, parti, path="C:\\project\\python\\pythonProject\\re\\keti\\dataset\\training data\\", dataset="HCGCN-all herb pairs"):

            np.random.seed(42)
            torch.manual_seed(42)

            # 加载原始数据
            idx_features_labels = np.genfromtxt(f"{path}{dataset}.content",
                                                dtype=np.dtype(str),
                                                encoding='utf-8')

            # 构建特征矩阵
            features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
            labels = encode_onehot(idx_features_labels[:, -1])

            # 构建图结构
            idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
            idx_map = {j: i for i, j in enumerate(idx)}
            edges = np.genfromtxt(f"{path}{dataset}.cites", dtype=np.int32)
            edges = np.array(list(map(idx_map.get, edges.flatten())),
                             dtype=np.int32).reshape(edges.shape)

            # 构建对称邻接矩阵
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(labels.shape[0], labels.shape[0]),
                                dtype=np.float32)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

            # 归一化处理
            features = normalize(features)
            adj = normalize(adj + sp.eye(adj.shape[0]))

            # 分层划分训练测试集
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            labels_raw = np.where(labels)[1]  # 转换回原始标签
            for train_idx, test_idx in sss.split(features, labels_raw):
                idx_train = torch.LongTensor(train_idx)
                idx_test = torch.LongTensor(test_idx)

            # 为验证集 idx_val 进行赋值
            idx_val = torch.LongTensor([])  # 可以为空，或者按照需要填充

            # 转换为PyTorch张量
            features = torch.FloatTensor(np.array(features.todense()))
            labels = torch.LongTensor(np.where(labels)[1])
            adj = sparse_mx_to_torch_sparse_tensor(adj)

            # 数据完整性验证
            assert len(idx_train) + len(idx_test) == len(features), "数据划分不完整!"
            print_data_stats(features, labels, idx_train, idx_test, adj)

            # 返回 adj, features, labels, idx_train, idx_val, idx_test
            return adj, features, labels, idx_train, idx_val, idx_test


def print_data_stats(features, labels, idx_train, idx_test, adj):
    """打印数据统计信息"""
    print("\n=== 数据统计 ===")
    print(f"总样本量: {len(features)}")
    print(f"特征维度: {features.shape[1]}")
    print(f"训练集: {len(idx_train)} (负类={sum(labels[idx_train] == 0)}, 正类={sum(labels[idx_train] == 1)})")
    print(f"测试集: {len(idx_test)} (负类={sum(labels[idx_test] == 0)}, 正类={sum(labels[idx_test] == 1)})")
    print(f"邻接矩阵密度: {(adj._nnz() / (adj.size(0) * adj.size(1))):.4f}")



def Evaluating_Indicator(y_true, y_pred):
    """计算评估指标"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 处理除零情况
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    auc = roc_auc_score(y_true, y_pred)

    return tn, fp, fn, tp, acc, recall, precision, F1, auc


def write_result_to_file(acc, recall, precision, F1, auc, filename='rs.csv'):
    """写入结果到文件"""
    with open(filename, 'a') as f:
        f.write(f"{acc:.4f},{recall:.4f},{precision:.4f},{F1:.4f},{auc:.4f}\n")


def write_matrix_to_file(tn, fp, fn, tp, filename='matrix.csv'):
    """写入混淆矩阵到文件"""
    with open(filename, 'a') as f:
        f.write(f"{tn},{fp},{fn},{tp}\n")


def plot_attention_weights(weights_dict, epoch):
    """可视化注意力权重"""
    plt.figure(figsize=(12, 6))

    # 节点注意力权重
    plt.subplot(121)
    plt.hist(weights_dict['node'].numpy(), bins=20)
    plt.title(f'Epoch {epoch}: Node Attention')

    # 语义注意力权重
    plt.subplot(122)
    plt.bar(range(len(weights_dict['semantic'])), weights_dict['semantic'].numpy())
    plt.title(f'Epoch {epoch}: Semantic Attention')

    plt.tight_layout()
    plt.savefig(f'attention_epoch_{epoch}.png')
    plt.close()


def accuracy(output, labels):
    """计算准确率"""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


def check_data_consistency(features, labels):
    """检查数据一致性"""
    assert not torch.isnan(features).any(), "特征包含NaN值!"
    assert torch.all(labels >= 0), "标签包含负值!"
    assert features.shape[0] == labels.shape[0], "特征与标签数量不匹配!"
    print("数据一致性检查通过")
