import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def Evaluating_Indicator(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  # 混淆矩阵
    acc = metrics.accuracy_score(y_true, y_pred)  # 准确率
    recall = metrics.recall_score(y_true, y_pred)  # 召回率
    precision = metrics.precision_score(y_true, y_pred)  # 精度
    F1 = metrics.f1_score(y_true, y_pred)  # F1 分数
    auc = metrics.roc_auc_score(y_true, y_pred)  # AUC
    return tn, fp, fn, tp, acc, recall, precision, F1, auc


def write_result_to_file(acc, recall, precision, F1, auc):  # 写入指标
    with open('rs.csv', 'a') as fw:  # 写入指标
        fw.write(str(acc))  # 写入指标
        fw.write(',')  # 写入指标
        fw.write(str(recall))  # 写入指标
        fw.write(',')  # 写入指标
        fw.write(str(precision))  # 写入指标
        fw.write(',')  # 写入指标
        fw.write(str(F1))  # 写入指标
        fw.write(',')
        fw.write(str(auc))
        fw.write('\n')


def write_matrix_to_file(tn, fp, fn, tp):
    with open('matrix.csv', 'a') as fw:
        fw.write(str(tn))
        fw.write(',')
        fw.write(str(fp))
        fw.write(',')
        fw.write(str(fn))
        fw.write(',')
        fw.write(str(tp))
        fw.write('\n')


def encode_onehot(labels):  # 将标签编码为 one-hot 形式
    classes = list(set(labels))  # 获取标签的种类
    classes.sort(key=list(labels).index, reverse=True)  # 按标签的索引进行排序
    # print(classes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}  # 将标签编码为 one-hot 形式
    labels_onehot = np.array(list(map(classes_dict.get, labels)),  # 将标签转换为 one-hot 形式
                             dtype=np.int32)
    return labels_onehot  # 返回 one-hot 编码的标签


def load_data(file_par, parti, path="C:\\project\\python\\pythonProject\\re\\keti\\dataset\\training data\\",
              dataset="HCGCN-all herb pairs"):  # 加载数据集
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))  # 打印正在加载的数据集

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str), encoding='utf-8')  # 读取数据集的特征和标签
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 第 2 列到倒数第 2 列，将特征转换为稀疏矩阵
    labels = encode_onehot(idx_features_labels[:, -1])  # 将所有行的最后一列标签转换为 one-hot 形式

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  # 将第一列特征标签的索引转换为整数类型
    idx_map = {j: i for i, j in enumerate(idx)}  # 将生成字典的元素映射到索引中
    # dataset = dataset + str(file_par)
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)  # 读取数据集的边
    '''
    np.genfromtxt：
    这是 NumPy 提供的一个函数，用于从文本文件中加载数据并生成一个 NumPy 数组。
    它可以处理缺失值，并支持多种数据类型。
    path 和 dataset 拼接成一个完整的文件路径，依次对应两个{}
    {} 是占位符，format 方法会将 path 和 dataset 的值依次填充到占位符中
    '''
    # 将 edges_unordered 中的节点编号映射为新的编号（基于 idx_map），并将结果重新整理为一个与 edges_unordered 形状相同的 NumPy 数组

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    '''
    map 是 Python 的内置函数，用于将一个函数应用到可迭代对象的每个元素上。
    idx_map.get 是一个字典的 get 方法，用于根据键（节点编号）查找对应的值（新的编号）。
    例如：
    idx_map = {35: 0, 40: 1, 50: 2}
    list(map(idx_map.get, [35, 40, 40, 50, 50, 35]))  # 结果为 [0, 1, 1, 2, 2, 0]
    .reshape(edges_unordered.shape)将一维数组重新调整为与 edges_unordered 相同的形状。
    例如：
    np.array([0, 1, 1, 2, 2, 0], dtype=np.int32).reshape((3, 2)) # 结果为 [[0, 1], [1, 2], [2, 0]]
    '''
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    '''
    scipy.sparse.coo_matrix 是 SciPy 提供的一种稀疏矩阵格式，称为 坐标格式（COO）。
    它通过三个数组来定义矩阵：
    data：非零元素的值（这里是 np.ones(edges.shape[0])）。
    row：非零元素的行索引（这里是 edges[:, 0]）。
    col：非零元素的列索引（这里是 edges[:, 1]）。
    '''

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    '''
    adj.T > adj
    这是一个布尔矩阵，表示 adj.T 中每个元素是否大于 adj 中对应位置的元素。
    例如：
    adj = [[0, 1], [0, 0]]
    adj.T = [[0, 0], [1, 0]]
    adj.T > adj = [[False, False], [True, False]]
    adj.T.multiply(adj.T > adj)
    multiply 是逐元素乘法（哈达玛积）。
    这里将 adj.T 与布尔矩阵 (adj.T > adj) 相乘，结果是一个矩阵，其中：
    如果 adj.T[i, j] > adj[i, j]，则保留 adj.T[i, j] 的值。否则为 0。
    例如：
    adj.T.multiply(adj.T > adj) = [[0, 0], [1, 0]]
    '''

    features = normalize(features)  # 将特征矩阵归一化

    adj = normalize(adj + sp.eye(adj.shape[0]))  # 对邻接矩阵 adj 添加自环并进行归一化。

    '''
    part1 = list(range(0,64))
    part2 = list(range(64,128))
    part3 = list(range(128,192))
    part4 = list(range(192,318))
    part5 = list(range(255,318))
    '''
    # part5 = list(range(318,568))

    part1 = list(range(0, 32))
    part2 = list(range(32, 64))
    part3 = list(range(64, 86))
    part4 = list(range(96, 132))
    part5 = list(range(132, 164))
    '''
    list(range(start, end))
    range(start, end) 生成一个从 start 到 end-1 的整数序列。
    list() 将 range 对象转换为列表。
    2. part1, part2, part3, part4, part5
    每个 part 列表包含一个连续的整数范围：
    part1 = list(range(0, 32))：生成 [0, 1, 2, ..., 31]。
    part2 = list(range(32, 64))：生成 [32, 33, 34, ..., 63]。
    part3 = list(range(64, 86))：生成 [64, 65, 66, ..., 85]。
    part4 = list(range(96, 132))：生成 [96, 97, 98, ..., 131]。
    part5 = list(range(132, 164))：生成 [132, 133, 134, ..., 163]。
    '''
    partlist = [part1, part2, part3, part4, part5]
    '''
    这行代码将 part1, part2, part3, part4, part5 组合成一个嵌套列表 partlist。
    partlist 是一个列表的列表，便于统一管理和操作多个部分的数据。
    这种结构常用于数据划分、批量处理或数据传递。
    '''
    test_part = partlist[parti]  # 从 partlist 中提取特定的部分（子列表），用于后续处理或分析。
    idx_train = []  # 创建一个空列表，用于存储训练数据的索引。
    for t in partlist:  # 遍历 partlist 中的每个子列表，并判断是否与 test_part 相等。
        if t != test_part:  # 如果不相等，则将子列表中的元素添加到 idx_train 中。
            idx_train = idx_train + t  # 将子列表中的元素添加到 idx_train 中。
    idx_val = range(0, 0)  # 创建一个空列表，用于存储验证数据的索引。
    idx_test = test_part  # 将 test_part 赋值给 idx_test，用于后续处理或分析。

    features = torch.FloatTensor(np.array(features.todense()))  # 将特征矩阵转换为 PyTorch 的 FloatTensor 类型
    labels = torch.LongTensor(np.where(labels)[1])  # 将标签转换为 PyTorch 的 LongTensor 类型
    adj = sparse_mx_to_torch_sparse_tensor(adj)  # 将邻接矩阵转换为 PyTorch 的稀疏张量类型

    idx_train = torch.LongTensor(idx_train)  # 将训练数据的索引转换为 PyTorch 的 LongTensor 类型
    idx_val = torch.LongTensor(idx_val)  # 将验证数据的索引转换为 PyTorch 的 LongTensor 类型
    idx_test = torch.LongTensor(idx_test)  # 将测试数据的索引转换为 PyTorch 的 LongTensor 类型

    return adj, features, labels, idx_train, idx_val, idx_test  # 返回邻接矩阵、特征矩阵、标签矩阵、训练数据的索引、验证数据的索引和测试数据的索引


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 计算每一行的和
    r_inv = np.power(rowsum + 1e-10, -1).flatten()  # 计算每一行的逆
    # rowsum + 1e-10：为了避免除以零的情况，给每一行的和加上一个很小的数（1e-10）
    r_inv[np.isinf(r_inv)] = 0.  # 将无穷大的值设置为 0
    '''
    np.isinf(r_inv)：检查 r_inv 中是否有无穷大的值（例如，当某一行和为 0 时，倒数会变成无穷大）。
    将无穷大的值设置为 0。
    '''
    r_mat_inv = sp.diags(r_inv)  # 将数组转换为对角矩阵
    mx = r_mat_inv.dot(mx)  # r_mat_inv.dot(mx)：将对角矩阵 r_mat_inv 与稀疏矩阵 mx 相乘，实现行归一化。
    return mx  # 返回归一化后的矩阵


'''
output.max(1)[1]
output 是模型的输出，通常是一个二维张量，形状为 (N, C)，其中：
N 是样本数量。
C 是类别数量。
output.max(1)：对 output 的每一行取最大值，返回一个元组 (max_values, max_indices)。
max_values：每一行的最大值。
max_indices：每一行最大值的索引（即预测的类别）。
output.max(1)[1]：提取最大值的索引，即模型的预测类别。
'''


def accuracy(output, labels):
    # print('output',output.max(1)[1])
    # print('labels',labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    '''
    preds.eq(labels)
    eq 是 PyTorch 张量的方法，用于逐元素比较两个张量是否相等。     
    返回一个布尔张量，表示每个预测是否与真实标签匹配。
    '''
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):  # 将 SciPy 稀疏矩阵转换为 PyTorch 稀疏张量
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    '''
    sparse_mx.tocoo()：将稀疏矩阵转换为 COO（Coordinate Format）格式。
    COO 格式是一种稀疏矩阵存储方式，通过三个数组表示：
    row：非零元素的行索引。
    col：非零元素的列索引。
    data：非零元素的值。
    astype(np.float32)：将稀疏矩阵的数据类型转换为 float32，以便与 PyTorch 张量兼容。
    '''
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  # np.vstack((sparse_mx.row, sparse_mx.col))：将 row 和 col 数组垂直堆叠，形成一个二维数组。
    '''
    torch.from_numpy(...)：将 NumPy 数组转换为 PyTorch 张量。
    astype(np.int64)：确保索引数据类型为 int64，这是 PyTorch 稀疏张量的要求
    '''
    values = torch.from_numpy(sparse_mx.data)
    '''
    values = torch.from_numpy(sparse_mx.data)
    sparse_mx.data：COO 格式稀疏矩阵的非零元素值。
    torch.from_numpy(...)：将 NumPy 数组转换为 PyTorch 张量。
    '''
    shape = torch.Size(sparse_mx.shape)
    '''
    shape = torch.Size(sparse_mx.shape)
    sparse_mx.shape：稀疏矩阵的形状（行数，列数）。
    torch.Size(...)：将形状转换为 PyTorch 的 Size 对象
    '''
    return torch.sparse_coo_tensor(indices, values, shape)


def write_roc_data_to_file(model_name, fpr, tpr):
    with open('roc_data.csv', 'w') as fw:
        fw.write('Model,FPR,TPR\n')
        for i in range(len(fpr)):
            fw.write(f'{model_name},{fpr[i]},{tpr[i]}\n')
