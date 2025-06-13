import torch
import numpy as np
import pandas as pd
from model import GCN

# 定义一个函数，用于加载特征
def load_features(features_path):
    # 使用pandas库读取特征文件，不读取表头，以制表符为分隔符
    df = pd.read_csv(features_path, header=None, sep='\t')
    # 将特征文件中的数据转换为numpy数组，并转换为浮点型
    features = torch.FloatTensor(np.array(df.iloc[:, 1:], dtype=np.float32))
    # 返回特征
    return features

def load_adjacency(adj_path, num_nodes):
    # 读取邻接矩阵文件
    edges = pd.read_csv(adj_path, header=None, sep='\t', dtype=np.int64)
    # 初始化邻接矩阵，大小为num_nodes*num_nodes，数据类型为float32
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    # 遍历邻接矩阵中的边
    for i, j in edges.values:
        # 将邻接矩阵中的对应位置置为1
        adj[i, j] = 1
        adj[j, i] = 1
    # 将邻接矩阵转换为torch张量
    adj = torch.FloatTensor(adj)
    # 返回邻接矩阵
    return adj

def predict(model_path, features, adj, num_classes, threshold=0.5, temperature=1.0):
    # 获取特征矩阵的维度
    nfeat = features.shape[1]
    # 定义隐藏层维度
    nhid = 256
    # 定义dropout概率
    dropout = 0.1
    # 初始化GCN模型
    model = GCN(nfeat=nfeat, nhid=nhid, nclass=num_classes, dropout=dropout)
    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # 设置模型为评估模式
    model.eval()

    # 在不计算梯度的情况下进行预测
    with torch.no_grad():
        # 获取模型输出
        output = model(features, adj)
        # 对输出进行温度缩放
        probabilities = torch.softmax(output / temperature, dim=1)  # Apply temperature scaling
        # 获取预测结果
        predictions = (probabilities[:, 1] >= threshold).long()

    # 返回预测结果和概率
    return predictions.numpy(), probabilities.numpy()
'''
DataFrame 是 Pandas 库中的二维表格数据结构，用于存储和操作结构化数据。
在这段代码中，DataFrame 用于将预测结果和概率组织成表格，并保存为 CSV 文件。
'''
def save_predictions(predictions, probabilities, output_path):
    # 创建一个DataFrame，包含预测结果和概率
    results = pd.DataFrame({
        'Index': range(len(predictions)),  # 创建一个索引列
        'Predicted Class': predictions,  # 创建一个预测结果列
        'Probability': probabilities.max(axis=1)  # 创建一个概率列，取每行的最大值
    })
    # 将DataFrame保存为CSV文件
    results.to_csv(output_path, index=False)


if __name__ == "__main__":
    for i in range(1, 644):
        features_path = f'C:\\project\\python\\pythonProject\\re\\keti\\dataset\\HCGCN-prediction set\\prediction set-{i}-.content'
        adj_path = f'C:\\project\\python\\pythonProject\\re\\keti\\dataset\\HCGCN-prediction set\\prediction set-{i}.cites'
        model_path = f'C:\\project\\python\\pythonProject\\re\\keti\\codes\\model.pth'
        output_path = f'C:\\project\\python\\pythonProject\\re\\keti\\dataset\\output\\predictions{i}.csv'

        features = load_features(features_path)
        num_nodes = features.shape[0]
        adj = load_adjacency(adj_path, num_nodes)

        num_classes = 2
        threshold = 0.8
        temperature = 30.0  # Set the temperature parameter

        predictions, probabilities = predict(model_path, features, adj, num_classes, threshold, temperature)

        save_predictions(predictions, probabilities, output_path)

        print(f"Predictions and probabilities for cora-zong-sum{i} have been saved to '{output_path}'.")
