import pandas as pd
from collections import defaultdict
import itertools

# 文件路径
input_file = r'C:\project\python\pythonProject\re\keti\3维柱状图\有效药对\调气四气五味.csv'
output_file = r'C:\project\python\pythonProject\re\keti\3维柱状图\调气\调气五味组合.csv'


def process_combination(cell):
    """处理单元格内容，分割多个值并去除空格"""
    if pd.isna(cell):
        return []
    return [x.strip() for x in str(cell).split(',') if x.strip()]


def get_unique_combinations(row, col1, col2):
    """获取一行的所有唯一组合（不考虑顺序）"""
    values1 = process_combination(row[col1])
    values2 = process_combination(row[col2])

    unique_combinations = set()
    for v1 in values1:
        for v2 in values2:
            # 创建不考虑顺序的组合键
            combo = frozenset({v1, v2})
            unique_combinations.add(combo)
    return unique_combinations


try:
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 获取列名（第5列索引为4，第9列索引为8）
    col5 = df.columns[4]
    col9 = df.columns[8]

    print(f"正在分析列: '{col5}' 和 '{col9}'...")

    # 统计组合数量
    combination_counts = defaultdict(int)
    total_pairs = 0

    for _, row in df.iterrows():
        combos = get_unique_combinations(row, col5, col9)
        for combo in combos:
            combination_counts[combo] += 1
            total_pairs += 1

    # 准备结果数据
    results = []
    for combo, count in combination_counts.items():
        combo_list = sorted(combo)
        if len(combo_list) == 1:  # 相同属性的组合
            results.append({
                "组合": f"{combo_list[0]} & {combo_list[0]}",
                "出现次数": count,
                "占比(%)": round(count / total_pairs * 100, 2)
            })
        else:  # 不同属性的组合
            results.append({
                "组合": f"{combo_list[0]} & {combo_list[1]}",
                "出现次数": count,
                "占比(%)": round(count / total_pairs * 100, 2)
            })

    # 创建DataFrame并排序
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by="出现次数", ascending=False)

    # 保存完整结果（不只是前10个）
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n统计完成！共分析 {len(df)} 行数据，生成 {len(result_df)} 种唯一组合。")
    print(f"结果已保存到: {output_file}")

    # 打印完整统计结果
    print("\n完整组合统计：")
    print(result_df.to_string(index=False))

except FileNotFoundError:
    print(f"错误：文件 {input_file} 未找到！")
except Exception as e:
    print(f"发生错误：{str(e)}")
    if 'df' in locals():
        print("\n文件列名列表：")
        print(df.columns.tolist())