# import csv
# import os
#
# def read_herb_pairs_from_csv(file_path):#读取herb_pairs.txt文件
#     herb_pairs = []#创建一个空列表，用于存储读取的herb_pairs
#     encodings = ['utf-8', 'ISO-8859-1', 'cp1252']#尝试不同的编码方式
#     for encoding in encodings:#遍历可能的编码方式
#         try:#尝试使用指定的编码方式打开文件
#             with open(file_path, 'r', encoding=encoding) as file:#尝试使用指定的编码方式打开文件
#                 reader = csv.reader(file, delimiter='\t')#使用指定的分隔符读取CSV文件
#                 for row in reader:#遍历每一行
#                     # Suppose that the format of the drug pair is ' herb 1 herb 2'
#                     if len(row) == 1:#判断行是否只有一个元素
#                         parts = row[0].split(' ')#将行拆分为两个部分
#                         if len(parts) == 2:#判断两个部分是否长度为2
#                             herb_pairs.append((parts[0].strip(), parts[1].strip()))#将两个部分添加到herb_pairs列表中
#             if herb_pairs:#如果herb_pairs不为空
#                 print(f"Successfully read the drug pair: {herb_pairs}")#打印成功读取的herb_pairs
#             break#跳出循环
#         except UnicodeDecodeError:#如果出现UnicodeDecodeError异常
#             print(f"Encoding {encoding} cannot be used to read files.")#打印无法使用当前编码方式读取文件的信息
#             continue#继续下一次循环
#     return herb_pairs#返回herb_pairs列表
# #
# # # Save the adjacency matrix file
# def save_adjacency_matrix(data, file_path):#保存邻接矩阵文件
#     pairs = [f"{a} {b}" for a, b in data]#将herb_pairs列表转换为字符串列表
#     pairs_set = [frozenset(pair.split()) for pair in pairs]#将herb_pairs列表转换为 frozenset 对象的列表
#     edges = set()#创建一个空集合，用于存储边
#     for i, pair_i in enumerate(pairs_set):#遍历herb_pairs列表
#         for j, pair_j in enumerate(pairs_set):#遍历herb_pairs列表
#             if i != j and not pair_i.isdisjoint(pair_j):#判断两个pair是否相交
#                 edges.add((pairs[i], pairs[j]))#将两个pair添加到edges集合中
#
#     with open(file_path, 'w', newline='', encoding='utf-8') as file:#打开文件，以写入模式打开
#         writer = csv.writer(file, delimiter='\t')#创建CSV写入器
#         for edge in edges:#遍历edges集合
#             writer.writerow(edge)#将边写入文件
#
#
# # Save the adjacency matrix file with numeric IDs
# # def save_adjacency_matrix(data, file_path):
# #     # Step 1: Create a mapping from herb names to numeric IDs
# #     herbs = set()
# #     for a, b in data:
# #         herbs.add(a)
# #         herbs.add(b)
# #     herb_to_id = {herb: i for i, herb in enumerate(herbs)}  # Assign unique IDs
# #
# #     # Step 2: Convert herb pairs to numeric ID pairs
# #     numeric_pairs = [(herb_to_id[a], herb_to_id[b]) for a, b in data]
# #
# #     # Step 3: Save numeric ID pairs to file
# #     with open(file_path, 'w', newline='', encoding='utf-8') as file:
# #         writer = csv.writer(file, delimiter='\t')
# #         for a, b in numeric_pairs:
# #             writer.writerow([a, b])
#
# input_csv_file = 'D:\\Project\\pythonProject\\re\\keti\\dataset\\herb pairs for training\\all herb pairs for training.txt'#输入的CSV文件路径
#
# herb_pairs = read_herb_pairs_from_csv(input_csv_file)#读取herb_pairs.txt文件
#
# output_dir = 'D:\\Project\\pythonProject\\re\\keti\\dataset\\training data\\'#输出目录
# if not os.path.exists(output_dir):#如果输出目录不存在
#     os.makedirs(output_dir)#创建输出目录
#
#
# save_adjacency_matrix(herb_pairs, os.path.join(output_dir, 'HCGCN-all herb pairs.cites'))#保存邻接矩阵文件
#
# print(f"Adjacency matrix saved to {os.path.join(output_dir, 'HCGCN-all herb pairs.cites')}")#打印保存邻接矩阵文件的信息
import csv
import os
import random


def read_herb_pairs_from_csv(file_path):
    herb_pairs = []  # 存储读取的药对
    encodings = ['utf-8', 'ISO-8859-1', 'cp1252']  # 尝试不同的编码方式
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                reader = csv.reader(file, delimiter='\t')
                for row in reader:
                    if len(row) == 1:
                        parts = row[0].split(' ')
                        if len(parts) == 2:
                            herb_pairs.append((parts[0].strip(), parts[1].strip()))
            if herb_pairs:
                print(f"Successfully read {len(herb_pairs)} drug pairs.")
            break  # 如果成功读取文件，跳出循环
        except UnicodeDecodeError:
            print(f"Encoding {encoding} cannot be used to read files. Trying next encoding...")
            continue  # 如果编码失败，尝试下一个编码
    return herb_pairs


def save_adjacency_matrix_with_indices(data, file_path):
    # 为每对药分配一个索引（从0开始）
    indexed_pairs = [(i, pair) for i, pair in enumerate(data, start=0)]

    # 创建一个列表来保存所有共享药材的药对索引对
    edges = []
    n = len(indexed_pairs)
    for i in range(n):
        pair_i = indexed_pairs[i][1]
        for j in range(n):
            if i != j:  # 避免同一药对配对
                pair_j = indexed_pairs[j][1]
                # 检查是否有共享的药材
                if pair_i[0] in pair_j or pair_i[1] in pair_j:
                    edges.append((indexed_pairs[i][0], indexed_pairs[j][0]))

    # 去重并随机排列
    edges = list(set(edges))  # 去重
    random.shuffle(edges)  # 随机排列

    # 保存到文件
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        for edge in edges:
            writer.writerow(edge)

    print(f"Generated {len(edges)} edges and saved to {file_path}")


# 输入文件路径
input_csv_file =f'C:\project\python\pythonProject\re\keti\dataset\herb pairs for training\all herb pairs for training.txt'

# 读取药对
herb_pairs = read_herb_pairs_from_csv(input_csv_file)

# 输出目录
output_dir = 'C:\\project\\python\\pythonProject\\re\\keti\\dataset\\training data\\'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存结果到文件
output_file = os.path.join(output_dir, 'heat-clearing herb pairs for training.cites')
save_adjacency_matrix_with_indices(herb_pairs, output_file)

print(f"Processing completed. Output saved to {output_file}")