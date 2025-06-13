import pandas as pd
import random
import os

# 输入文件路径
input_file = r"C:\project\python\pythonProject\re\keti\dataset\HCGCN-ID for prediction sets\HCGCN-ID for prediction set-8.csv"

# 输出文件路径
output_file = r"C:\project\python\pythonProject\re\keti\dataset\herb pairs for training\new dataset for training.txt"

# 确保输出目录存在
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 读取CSV文件
df = pd.read_csv(input_file)

# 修改第一列索引值（加100）
df.iloc[:, 0] = df.iloc[:, 0] + 700

# 将DataFrame转换为列表
rows = df.values.tolist()

# 随机打乱行顺序
random.shuffle(rows)

# 将打乱后的数据写入输出文件，使用逗号分隔
with open(output_file, 'a', encoding='utf-8') as f:
    for row in rows:
        # 将每行数据转换为字符串并用逗号连接
        line = ','.join(map(str, row))
        f.write(line + '\n')

print(f"处理完成！已将修改后的数据追加到 {output_file}")