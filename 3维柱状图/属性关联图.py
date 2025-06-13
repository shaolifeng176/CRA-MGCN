# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# from matplotlib.colors import ListedColormap
# import numpy as np
# import warnings
# warnings.filterwarnings("ignore")
#
# # 文件路径
# csv_file = r"C:\project\python\pythonProject\re\keti\3维柱状图\清热\清热英文.csv"
#
# # 自定义参数
# node_size_base = 1000  # 基础节点大小
# node_size_multiplier = 50  # 节点大小乘数因子
# font_size = 13  # 字体大小
# title_font_size = 16  # 标题字体大小
# edge_width = 1.5  # 边的宽度
# dpi = 800  # 图片分辨率
#
# # 定义节点颜色
# node_colors = {
#     'Latin Name': '#3498db',  # 蓝色
#     'Nature': '#e74c3c',  # 红色
#     'Taste': '#2ecc71',  # 绿色
#     'Meridian': '#f39c12'  # 橙色
# }
#
# # 读取CSV文件
# df = pd.read_csv(csv_file)
#
# # 创建一个空的无向图
# G = nx.Graph()
#
# # 添加节点和边
# latin_names = []
# nature_nodes = set()
# taste_nodes = set()
# meridian_nodes = set()
#
# for _, row in df.iterrows():
#     latin_name = row['Latin Name']
#     latin_names.append(latin_name)
#
#     # 添加Latin Name节点
#     G.add_node(latin_name, type='Latin Name')
#
#     # 添加Nature节点并连接
#     if pd.notna(row['Nature']):
#         natures = [n.strip() for n in row['Nature'].split(',')]
#         for nature in natures:
#             nature_nodes.add(nature)
#             G.add_node(nature, type='Nature')
#             G.add_edge(latin_name, nature)
#
#     # 添加Taste节点并连接
#     if pd.notna(row['Taste']):
#         tastes = [t.strip() for t in row['Taste'].split(',')]
#         for taste in tastes:
#             taste_nodes.add(taste)
#             G.add_node(taste, type='Taste')
#             G.add_edge(latin_name, taste)
#
#     # 添加Meridian节点并连接
#     if pd.notna(row['Meridian']):
#         meridians = [m.strip() for m in row['Meridian'].split(',')]
#         for meridian in meridians:
#             meridian_nodes.add(meridian)
#             G.add_node(meridian, type='Meridian')
#             G.add_edge(latin_name, meridian)
#
# # 计算节点大小（基于度）
# degrees = dict(G.degree())
# node_sizes = [degrees[node] * node_size_multiplier + node_size_base for node in G.nodes()]
#
# # 设置节点颜色
# node_color_list = [node_colors[G.nodes[node]['type']] for node in G.nodes()]
#
# # 定义四层圆形布局
# innermost_nodes = list(nature_nodes) + list(taste_nodes)
# second_nodes = list(meridian_nodes)
# outer_nodes = latin_names
#
# # 计算每层节点的位置
# innermost_radius = 1
# second_radius = 2
# third_radius = 3
# outermost_radius = 4
#
# innermost_pos = nx.circular_layout(innermost_nodes, scale=innermost_radius)
# second_pos = nx.circular_layout(second_nodes, scale=second_radius)
#
# # 将Latin Name节点分为两部分
# half_len = len(outer_nodes) // 2
# third_nodes = outer_nodes[:half_len]
# outermost_nodes = outer_nodes[half_len:]
#
# third_pos = nx.circular_layout(third_nodes, scale=third_radius)
# outermost_pos = nx.circular_layout(outermost_nodes, scale=outermost_radius)
#
# # 合并所有节点的位置
# pos = {**innermost_pos, **second_pos, **third_pos, **outermost_pos}
#
# # 创建图形
# plt.figure(figsize=(15, 15))
#
# # 绘制节点
# nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color_list,
#                        alpha=0.8, edgecolors='black', linewidths=1)
#
# # 绘制边
# nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.5, edge_color='gray')
#
# # 绘制标签
# labels = {node: node for node in G.nodes()}
# nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size,
#                         font_family='Times New Roman', font_weight='bold')
#
# # 设置标题
# plt.title('中药属性关联网络图', fontsize=title_font_size, fontweight='bold')
#
# # 移除坐标轴
# plt.axis('off')
#
# # 保存图形
# plt.savefig(r'C:\project\python\pythonProject\re\keti\3维柱状图\清热\herb_attribute_network.png', dpi=dpi, bbox_inches='tight')
#
# # 显示图形
# plt.tight_layout()
# plt.show()
#
# print("网络图已生成并保存为 'herb_attribute_network.png'")


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap
from matplotlib.patches import ConnectionPatch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 文件路径
csv_file = r"C:\project\python\pythonProject\re\keti\3维柱状图\清热\清热英文.csv"

# 自定义参数
node_size_base = 1000  # 基础节点大小
node_size_multiplier = 50  # 节点大小乘数因子
font_size = 13  # 字体大小
title_font_size = 16  # 标题字体大小
edge_width = 1.5  # 边的宽度
dpi = 800  # 图片分辨率

# 定义节点颜色
node_colors = {
    'Latin Name': '#3498db',  # 蓝色
    'Nature': '#e74c3c',  # 红色
    'Taste': '#2ecc71',  # 绿色
    'Meridian': '#f39c12'  # 橙色
}

# 读取CSV文件
df = pd.read_csv(csv_file)

# 创建一个空的无向图
G = nx.Graph()

# 添加节点和边
latin_names = []
nature_nodes = set()
taste_nodes = set()
meridian_nodes = set()

for _, row in df.iterrows():
    latin_name = row['Latin Name']
    latin_names.append(latin_name)

    # 添加Latin Name节点
    G.add_node(latin_name, type='Latin Name')

    # 添加Nature节点并连接
    if pd.notna(row['Nature']):
        natures = [n.strip() for n in row['Nature'].split(',')]
        for nature in natures:
            nature_nodes.add(nature)
            G.add_node(nature, type='Nature')
            G.add_edge(latin_name, nature)

    # 添加Taste节点并连接
    if pd.notna(row['Taste']):
        tastes = [t.strip() for t in row['Taste'].split(',')]
        for taste in tastes:
            taste_nodes.add(taste)
            G.add_node(taste, type='Taste')
            G.add_edge(latin_name, taste)

    # 添加Meridian节点并连接
    if pd.notna(row['Meridian']):
        meridians = [m.strip() for m in row['Meridian'].split(',')]
        for meridian in meridians:
            meridian_nodes.add(meridian)
            G.add_node(meridian, type='Meridian')
            G.add_edge(latin_name, meridian)

# 计算节点大小（基于度）
degrees = dict(G.degree())
node_sizes = [degrees[node] * node_size_multiplier + node_size_base for node in G.nodes()]

# 设置节点颜色
node_color_list = [node_colors[G.nodes[node]['type']] for node in G.nodes()]

# 定义四层圆形布局
innermost_nodes = list(nature_nodes) + list(taste_nodes)
second_nodes = list(meridian_nodes)
outer_nodes = latin_names

# 计算每层节点的位置
innermost_radius = 1
second_radius = 2
third_radius = 3
outermost_radius = 4

innermost_pos = nx.circular_layout(innermost_nodes, scale=innermost_radius)
second_pos = nx.circular_layout(second_nodes, scale=second_radius)

# 将Latin Name节点分为两部分
half_len = len(outer_nodes) // 2
third_nodes = outer_nodes[:half_len]
outermost_nodes = outer_nodes[half_len:]

third_pos = nx.circular_layout(third_nodes, scale=third_radius)
outermost_pos = nx.circular_layout(outermost_nodes, scale=outermost_radius)

# 合并所有节点的位置
pos = {**innermost_pos, **second_pos, **third_pos, **outermost_pos}

# 创建图形
plt.figure(figsize=(15, 15))

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color_list,
                       alpha=0.8, edgecolors='black', linewidths=1)

# 移除原有的直线边绘制代码
# nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.5, edge_color='gray')

# 绘制平滑曲线边
for u, v in G.edges():
    xyA = pos[u]
    xyB = pos[v]
    con = ConnectionPatch(xyA, xyB, coordsA="data", coordsB="data",
                          axesA=plt.gca(), axesB=plt.gca(),
                          arrowstyle="-", linestyle='-', linewidth=edge_width,
                          alpha=0.5, color='gray',
                          connectionstyle=f"arc3,rad={np.random.uniform(-0.3, 0.3)}")
    plt.gca().add_artist(con)

# 绘制标签
labels = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size,
                        font_family='Times New Roman', font_weight='bold')

# 设置标题
plt.title('中药属性关联网络图', fontsize=title_font_size, fontweight='bold')

# 移除坐标轴
plt.axis('off')

# 保存图形
plt.savefig(r'C:\project\python\pythonProject\re\keti\3维柱状图\清热\herb_attribute_network.png', dpi=dpi, bbox_inches='tight')

# 显示图形
plt.tight_layout()
plt.show()

print("网络图已生成并保存为 'herb_attribute_network.png'")