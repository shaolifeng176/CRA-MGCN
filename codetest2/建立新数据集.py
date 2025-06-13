import os

# 输入文件路径
input_file = r"C:\project\python\pythonProject\re\keti\dataset\HCGCN-prediction set\prediction set-8-.content"

# 输出文件路径
output_file = r"C:\project\python\pythonProject\re\keti\dataset\training data\new_dataset.content"

# 确保输出目录存在
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 读取输入文件并修改数据
modified_lines = []
with open(input_file, 'r') as f:
    for line in f:
        # 分割行数据
        columns = line.strip().split('\t')
        if columns:
            # 修改第一列，加100
            columns[0] = str(int(columns[0]) + 700)
            # 重新组合行数据
            modified_lines.append('\t'.join(columns))

# 将修改后的数据追加到输出文件
with open(output_file, 'a') as f:
    for line in modified_lines:
        f.write(line + '\n')

print(f"处理完成！已将修改后的数据追加到 {output_file}")