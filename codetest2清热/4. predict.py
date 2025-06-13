import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from model import GCNWithAttention
from utils import load_data, sparse_mx_to_torch_sparse_tensor, normalize, encode_onehot, add_topological_features

# 参数配置
hidden = 256
dropout = 0.5
n_heads = 4

# 加载模型，注意输入特征维度修改为48
model = GCNWithAttention(
    nfeat=48,  # 因为添加了两个拓扑特征，所以特征维度变为48
    nhid=hidden,
    nclass=2,  # 二分类问题
    dropout=dropout,
    n_heads=n_heads
)
model.load_state_dict(torch.load(r'C:\project\python\pythonProject\re\keti\codetest2\model_fold1_best.pth'))
model.eval()

# 设备转移
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def save_predictions(predictions, probabilities, output_path):
    """保存预测结果，包含索引、预测类别和预测概率"""
    results = pd.DataFrame({
        'Index': range(len(predictions)),  # 索引列
        'Predicted_Class': predictions,    # 预测类别 (0或1)
        # 'Probability_Class_0': probabilities[:, 0],  # 类别0的概率
        # 'Probability_Class_1': probabilities[:, 1],  # 类别1的概率
        'Probability': probabilities.max(axis=1)  # 最大概率值
    })
    results.to_csv(output_path, index=False)

# 循环处理643个数据文件
for i in range(1, 644):
    try:
        content_file = fr'C:\project\python\pythonProject\re\keti\dataset\HCGCN-prediction set\prediction set-{i}-.content'
        cites_file = fr'C:\project\python\pythonProject\re\keti\dataset\HCGCN-prediction set\prediction set-{i}.cites'
        output_file = fr'C:\project\python\pythonProject\re\keti\codetest2\output\predictions{i}.csv'

        # 加载特征和标签
        idx_features_labels = np.genfromtxt(content_file, dtype=np.dtype(str), delimiter='\t', encoding='gbk')
        features = sp.csr_matrix(idx_features_labels[:, 1:], dtype=np.float32)

        # 构建图结构
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.genfromtxt(cites_file, dtype=np.int32, delimiter='\t', encoding='gbk')
        edges = np.array(list(map(idx_map.get, edges.flatten())), dtype=np.int32).reshape(edges.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(features.shape[0], features.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # 添加拓扑特征
        features = add_topological_features(features, adj)

        # 归一化
        features = normalize(features)
        adj = normalize(adj + sp.eye(adj.shape[0]))

        # 转换为 PyTorch 张量
        features = torch.FloatTensor(np.array(features.todense()))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        # 设备转移
        features = features.to(device)
        adj = adj.to(device)

        # 进行预测
        with torch.no_grad():
            output = model(features, adj)
            if isinstance(output, tuple):
                output = output[0]  # 只取主任务输出
            probabilities = F.softmax(output, dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)

        # 保存预测结果
        save_predictions(predictions, probabilities, output_file)
        print(f"Successfully processed file {i}: predictions saved to {output_file}")

    except Exception as e:
        print(f"Error processing file {i}: {str(e)}")
        continue