import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from collections import defaultdict


def load_herb_component_data(csv_path):
    """加载中药成分数据，返回字典：{中药名: 成分列表}"""
    df = pd.read_csv(csv_path, header=None, encoding='gbk')
    herb_components = {}
    for row in df.values:
        herb = row[0].strip()
        components = [c.strip() for c in row[1:] if str(c).strip() != 'nan']
        herb_components[herb] = components
    return herb_components


def build_hypergraph_adj(herb_pairs_txt_path, herb_components):
    """
    构建超图邻接矩阵（基于成分共享）
    返回：超图邻接矩阵（稀疏矩阵，形状 [num_herb_pairs, num_components]）
    """
    # 读取药对数据
    with open(herb_pairs_txt_path, 'r', encoding='utf-8') as f:
        herb_pairs = [line.strip().split() for line in f if len(line.strip().split()) == 2]

    # 映射成分到唯一ID
    all_components = list(set(c for comps in herb_components.values() for c in comps))
    comp_to_id = {c: i for i, c in enumerate(all_components)}

    # 构建超边矩阵（药对-成分）
    num_pairs = len(herb_pairs)
    num_comps = len(all_components)
    rows, cols = [], []
    for pair_idx, (h1, h2) in enumerate(herb_pairs):
        comps = set(herb_components.get(h1, []) + herb_components.get(h2, []))
        for c in comps:
            cols.append(comp_to_id[c])
            rows.append(pair_idx)

    data = np.ones(len(rows))
    H = sp.coo_matrix((data, (rows, cols)), shape=(num_pairs, num_comps))
    # 转换为超图邻接矩阵: A = H * H^T - diag(H * H^T)
    adj = H.dot(H.T) - sp.diags(H.power(2).sum(axis=1).A1)
    adj[adj > 0] = 1  # 二值化
    adj.setdiag(0)  # 移除自环
    return adj


# 示例用法
if __name__ == "__main__":
    herb_components = load_herb_component_data('merged_filtered_herbs.csv')
    adj_hyper = build_hypergraph_adj('C:\\project\\python\\pythonProject\\re\\keti\\dataset\\herb pairs for training\\all herb pairs for training.txt', herb_components)
    adj_hyper = adj_hyper.tocoo()