import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from model import GCNWithAttention
from utils import load_data, sparse_mx_to_torch_sparse_tensor, normalize, encode_onehot, add_topological_features
import warnings
warnings.filterwarnings("ignore")

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
model.load_state_dict(torch.load(r'C:\project\python\pythonProject\re\keti\codetest2 - 副本\model_fold4_best.pth'))
model.eval()

# 设备转移
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def save_predictions(predictions, probabilities, output_path):
    """保存预测结果，包含索引、预测类别和预测概率"""
    results = pd.DataFrame({
        'Index': range(len(predictions)),  # 索引列
        'Predicted_Class': predictions,    # 预测类别 (0或1)
        'Probability': probabilities.max(axis=1)  # 最大概率值
    })
    results.to_csv(output_path, index=False)

# 初始化一个空的DataFrame用于存储合并后的数据
merged_data = pd.DataFrame(columns=['Source_File', 'Original_Index', 'Predicted_Class', 'Probability'])

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

        # 读取预测结果文件
        df = pd.read_csv(output_file)

        # 添加来源文件列
        df['Source_File'] = f'predictions{i}.csv'

        # 重命名索引列为Original_Index
        df = df.rename(columns={'Index': 'Original_Index'})

        # 选择需要的列
        df = df[['Source_File', 'Original_Index', 'Predicted_Class', 'Probability']]

        # 合并数据
        merged_data = pd.concat([merged_data, df], ignore_index=True)

    except Exception as e:
        print(f"Error processing file {i}: {str(e)}")
        continue

# 保存合并后的数据
merged_output_file = r'C:\project\python\pythonProject\re\keti\codetest2 - 副本\output\merged_predictions4.csv'
merged_data.to_csv(merged_output_file, index=False)
print(f"Successfully merged data saved to {merged_output_file}")