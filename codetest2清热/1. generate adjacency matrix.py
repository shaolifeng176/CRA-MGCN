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
input_csv_file = r'C:\project\python\pythonProject\re\keti\dataset\herb pairs for training\heat-clearing herb pairs for training.txt'

# 读取药对
herb_pairs = read_herb_pairs_from_csv(input_csv_file)

# 输出目录
output_dir = 'C:\\project\\python\\pythonProject\\re\\keti\\codetest2清热\\training data\\'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存结果到文件
output_file = os.path.join(output_dir, 'HCGCN-heat-clearing herb pairs.cites')
save_adjacency_matrix_with_indices(herb_pairs, output_file)

print(f"Processing completed. Output saved to {output_file}")