import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# 读取txt文件并提取所有中药（去重）
def get_unique_herbs_from_txt(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as f:
        herbs = set()
        for line in f:
            herb1, herb2 = line.strip().split()
            herbs.add(herb1)
            herbs.add(herb2)
    return sorted(herbs)


# 主处理函数
def filter_herbs_in_csv(txt_file, csv_file, output_csv):
    # 获取txt中所有中药
    txt_herbs = get_unique_herbs_from_txt(txt_file)
    print(f"从txt文件中找到{len(txt_herbs)}种独特中药")

    # 读取csv文件（分块读取以处理大文件）
    chunks = []
    for chunk in pd.read_csv(csv_file, encoding='utf-8', chunksize=1000):
        chunks.append(chunk)
    df = pd.concat(chunks, axis=0)

    # 确保第一列是中药名称
    herb_col = df.columns[0]

    # 找出csv中存在的txt中药
    found_herbs = set(df[herb_col])
    missing_herbs = [h for h in txt_herbs if h not in found_herbs]

    if missing_herbs:
        print(f"警告：以下中药在csv文件中不存在: {', '.join(missing_herbs)}")
    else:
        print("所有中药在csv文件中都存在")

    # 筛选出csv中包含的txt中药
    filtered_df = df[df[herb_col].isin(txt_herbs)]

    # 保存到新csv文件
    filtered_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"筛选出的数据已保存到 {output_csv}，共{len(filtered_df)}行")

    return missing_herbs


# 使用示例
if __name__ == "__main__":
    txt_file = "all herb pairs for training.txt"  # 你的txt文件路径
    csv_file = "herb_related_ingredients.csv"  # 你的原始csv文件路径
    output_csv = "filtered_herbs.csv"  # 输出文件路径

    missing = filter_herbs_in_csv(txt_file, csv_file, output_csv)

    if missing:
        print("\n以下中药在csv文件中不存在:")
        for herb in missing:
            print(f"- {herb}")