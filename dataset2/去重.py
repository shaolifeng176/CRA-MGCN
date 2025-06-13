import pandas as pd
from collections import defaultdict


def merge_duplicate_herbs(input_csv, output_csv):
    # 读取已筛选的CSV文件
    df = pd.read_csv(input_csv, encoding='utf-8')

    # 确保第一列是中药名称
    herb_col = df.columns[0]
    df = df.rename(columns={herb_col: '中药名'})

    # 创建一个字典来合并成分
    herb_components = defaultdict(list)

    # 获取所有成分列名（排除中药名列）
    component_columns = df.columns[1:]

    # 遍历每一行，收集所有成分
    for _, row in df.iterrows():
        herb_name = row['中药名']
        components = row[component_columns].dropna().tolist()  # 去除空值
        herb_components[herb_name].extend(components)

    # 去重每个中药的成分
    for herb in herb_components:
        herb_components[herb] = list(set(herb_components[herb]))  # 使用set去重

    # 创建新的DataFrame
    merged_data = []
    for herb, components in herb_components.items():
        merged_row = [herb] + components
        merged_data.append(merged_row)

    # 确定最大成分数量
    max_components = max(len(components) for components in herb_components.values())

    # 创建列名
    columns = ['中药名'] + [f'成分{i + 1}' for i in range(max_components)]

    # 创建DataFrame并保存
    merged_df = pd.DataFrame(merged_data, columns=columns)
    merged_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"合并后的数据已保存到 {output_csv}，共{len(merged_df)}行")


if __name__ == "__main__":
    input_csv = "filtered_herbs.csv"  # 已生成的筛选文件
    output_csv = "merged_filtered_herbs.csv"  # 合并后的输出文件

    merge_duplicate_herbs(input_csv, output_csv)