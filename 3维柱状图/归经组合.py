# import pandas as pd
# from collections import defaultdict
# import itertools
#
# # 文件路径
# input_file = r'C:\project\python\pythonProject\re\keti\3维柱状图\有效药对\清热四气五味.csv'
# output_file = r'C:\project\python\pythonProject\re\keti\3维柱状图\清热\清热归经组合.csv'
#
# # 十二归经标准列表
# standard_channels = ["肺", "心包", "心", "大肠", "三焦", "小肠",
#                      "胃", "胆", "膀胱", "脾", "肝", "肾"]
#
#
# def process_channel(cell):
#     """处理归经单元格内容，分割多个值并验证是否为标准归经"""
#     if pd.isna(cell):
#         return []
#     # 分割并去除空格，只保留标准归经
#     channels = [x.strip() for x in str(cell).split(',') if x.strip() in standard_channels]
#     return channels
#
#
# def get_channel_combinations(row, col6, col10):
#     """获取一行的所有归经组合"""
#     channels1 = process_channel(row[col6])
#     channels2 = process_channel(row[col10])
#
#     combinations = set()  # 使用set自动去重
#     for c1 in channels1:
#         for c2 in channels2:
#             # 创建不考虑顺序的组合键（排序后的元组）
#             combo = tuple(sorted((c1, c2)))
#             combinations.add(combo)
#     return combinations
#
#
# try:
#     # 读取CSV文件
#     df = pd.read_csv(input_file)
#
#     # 获取列名（第6列索引为5，第10列索引为9）
#     col6 = df.columns[5]
#     col10 = df.columns[9]
#
#     print(f"正在分析列: '{col6}' 和 '{col10}'...")
#
#     # 统计组合数量
#     combination_counts = defaultdict(int)
#
#     for _, row in df.iterrows():
#         combos = get_channel_combinations(row, col6, col10)
#         for combo in combos:
#             combination_counts[combo] += 1
#
#     # 准备结果数据
#     results = []
#     for (channel1, channel2), count in combination_counts.items():
#         results.append({
#             "归经1": channel1,
#             "归经2": channel2,
#             "出现次数": count
#         })
#
#     # 创建DataFrame并排序
#     result_df = pd.DataFrame(results)
#     result_df = result_df.sort_values(by=["出现次数", "归经1", "归经2"], ascending=[False, True, True])
#
#     # 保存结果
#     result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
#
#     print(f"\n统计完成！共分析 {len(df)} 行数据，生成 {len(result_df)} 种归经组合。")
#     print(f"结果已保存到: {output_file}")
#
#     # 打印统计摘要
#     print("\n归经组合统计摘要：")
#     print(f"总组合对数: {sum(combination_counts.values())}")
#     print(f"唯一组合数: {len(combination_counts)}")
#     print("\n最常见的前10个组合：")
#     print(result_df.head(10).to_string(index=False))
#
# except FileNotFoundError:
#     print(f"错误：文件 {input_file} 未找到！")
# except Exception as e:
#     print(f"发生错误：{str(e)}")
#     if 'df' in locals():
#         print("\n文件列名列表：")
#         print(df.columns.tolist())

import pandas as pd
from collections import defaultdict
import itertools

# 文件路径
input_file = r'C:\project\python\pythonProject\re\keti\3维柱状图\有效药对\调气四气五味.csv'
output_file = r'C:\project\python\pythonProject\re\keti\3维柱状图\调气\调气归经组合.csv'

# 十二归经中英文对照表
channel_mapping = {
    "肺": "Lun.M",
    "心包": "P.M",
    "心": "H.M",
    "大肠": "Lar.I.M",
    "三焦": "T.B.M",
    "小肠": "Sma.I.M",
    "胃": "Sto.M",
    "胆": "G.M",
    "膀胱": "B.M",
    "脾": "Spl.M",
    "肝": "Liv.M",
    "肾": "K.M"
}

# 标准归经列表（中文）
standard_channels = list(channel_mapping.keys())


def process_channel(cell):
    """处理归经单元格内容，分割多个值并验证是否为标准归经"""
    if pd.isna(cell):
        return []
    # 分割并去除空格，只保留标准归经
    channels = [x.strip() for x in str(cell).split(',') if x.strip() in standard_channels]
    return channels


def get_channel_combinations(row, col6, col10):
    """获取一行的所有归经组合（返回英文缩写）"""
    channels1 = process_channel(row[col6])
    channels2 = process_channel(row[col10])

    combinations = set()  # 使用set自动去重
    for c1 in channels1:
        for c2 in channels2:
            # 转换为英文缩写
            c1_en = channel_mapping[c1]
            c2_en = channel_mapping[c2]
            # 创建不考虑顺序的组合键（排序后的元组）
            combo = tuple(sorted((c1_en, c2_en)))
            combinations.add(combo)
    return combinations


try:
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 获取列名（第6列索引为5，第10列索引为9）
    col6 = df.columns[5]
    col10 = df.columns[9]

    print(f"正在分析列: '{col6}' 和 '{col10}'...")

    # 统计组合数量
    combination_counts = defaultdict(int)

    for _, row in df.iterrows():
        combos = get_channel_combinations(row, col6, col10)
        for combo in combos:
            combination_counts[combo] += 1

    # 准备结果数据
    results = []
    for (channel1_en, channel2_en), count in combination_counts.items():
        results.append({
            "Channel1": channel1_en,
            "Channel2": channel2_en,
            "Count": count
        })

    # 创建DataFrame并排序
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        by=["Count", "Channel1", "Channel2"],
        ascending=[False, True, True]
    )

    # 保存结果
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n统计完成！共分析 {len(df)} 行数据，生成 {len(result_df)} 种归经组合。")
    print(f"结果已保存到: {output_file}")

    # 打印统计摘要
    print("\n归经组合统计摘要：")
    print(f"总组合对数: {sum(combination_counts.values())}")
    print(f"唯一组合数: {len(combination_counts)}")
    print("\n最常见的前10个组合：")
    print(result_df.head(10).to_string(index=False))

    # 输出中英文对照表供参考
    print("\n十二归经中英文对照表：")
    for cn, en in channel_mapping.items():
        print(f"{cn}: {en}")

except FileNotFoundError:
    print(f"错误：文件 {input_file} 未找到！")
except Exception as e:
    print(f"发生错误：{str(e)}")
    if 'df' in locals():
        print("\n文件列名列表：")
        print(df.columns.tolist())