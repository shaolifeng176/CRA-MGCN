import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# 文件路径设置
content_file = r'C:\project\python\pythonProject\re\keti\dataset\training data\HCGCN-qi-regulating herb pairs.content'
herb_pairs_csv = r'C:\project\python\pythonProject\re\keti\dataset\training data\HCGCN-ID for qi-regulating herb pairs.csv'
herb_info_excel = r'C:\project\python\test\dataset\filtered_herbs.xlsx'

# 输出文件路径
filtered_pairs_output = r'C:\project\python\pythonProject\re\keti\3维柱状图\有效药对\调气.csv'
merged_info_output = r'C:\project\python\pythonProject\re\keti\3维柱状图\有效药对\调气四气五味.csv'


def filter_effective_herb_pairs():
    """第一步：筛选有效药对"""
    try:
        # 读取content文件
        content_data = pd.read_csv(content_file, sep='\t', header=None)

        # 筛选出标签为"yes"的行的第一列（索引）
        yes_indices = content_data[content_data[47] == 'yes'][0].tolist()

        # 读取药对CSV文件
        csv_data = pd.read_csv(herb_pairs_csv, encoding='gbk')

        # 筛选出有效药对
        filtered_data = csv_data[csv_data.iloc[:, 0].isin(yes_indices)]

        # 保存筛选结果
        filtered_data.to_csv(filtered_pairs_output, index=False)
        print(f"\n第一步完成：共筛选出 {len(filtered_data)} 条有效药对，已保存到 {filtered_pairs_output}")

        return filtered_data

    except Exception as e:
        print(f"\n第一步出错：{str(e)}")
        return None


def merge_herb_pair_info(filtered_pairs_df):
    """第二步：合并药对中两味中药的信息"""
    try:
        # 读取Excel中的中药信息（只读取前4列）
        herb_info = pd.read_excel(herb_info_excel, usecols=range(4))

        # 准备合并后的DataFrame
        merged_data = []
        missing_herbs = set()

        for _, row in filtered_pairs_df.iterrows():
            herb1, herb2 = str(row[1]).split(' ')  # 假设第二列是药对

            # 查找第一味中药的信息
            herb1_info = herb_info[herb_info.iloc[:, 0] == herb1]
            if herb1_info.empty:
                missing_herbs.add(herb1)
                continue

            # 查找第二味中药的信息
            herb2_info = herb_info[herb_info.iloc[:, 0] == herb2]
            if herb2_info.empty:
                missing_herbs.add(herb2)
                continue

            # 合并两味中药的信息（各取前4列）
            merged_row = {
                '药对索引': row[0],
                '药对': row[1],
                **{f'中药1_{col}': herb1_info.iloc[0, i] for i, col in enumerate(herb_info.columns)},
                **{f'中药2_{col}': herb2_info.iloc[0, i] for i, col in enumerate(herb_info.columns)}
            }
            merged_data.append(merged_row)

        # 转换为DataFrame
        merged_df = pd.DataFrame(merged_data)

        # 保存合并后的信息
        merged_df.to_csv(merged_info_output, index=False, encoding='utf-8')

        # 打印报告
        print(f"\n第二步完成：共合并 {len(merged_df)} 对中药信息")
        print(f"合并后的药对信息已保存到 {merged_info_output}")

        if missing_herbs:
            print("\n以下中药在Excel文件中未找到:")
            for herb in sorted(missing_herbs):
                print(f"- {herb}")
        else:
            print("\n所有中药都在Excel文件中找到了对应记录。")

    except Exception as e:
        print(f"\n第二步出错：{str(e)}")


# 主程序
if __name__ == "__main__":
    print("开始处理数据...")

    # 第一步：筛选有效药对
    effective_pairs = filter_effective_herb_pairs()

    if effective_pairs is not None and not effective_pairs.empty:
        # 第二步：合并药对信息
        merge_herb_pair_info(effective_pairs)

    print("\n所有处理完成！")
