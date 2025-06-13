import pandas as pd
import os

# 文件路径
herb_pair_csv = r"C:\project\python\pythonProject\re\keti\3维柱状图\有效药对\调气四气五味.csv"
latin_name_excel = r"C:\project\python\pythonProject\re\keti\dataset\herb-herb latin name.xlsx"
output_csv = r"C:\project\python\pythonProject\re\keti\3维柱状图\调气\调气英文.csv"

# 读取数据
herb_pairs = pd.read_csv(herb_pair_csv)
latin_names = pd.read_excel(latin_name_excel)

# 创建拉丁文名称映射字典
latin_map = dict(zip(latin_names['herb'], latin_names['latin name']))

# 定义中文到英文/缩写的映射
nature_map = {
    "寒": "cold", "热": "hot", "温": "warm", "凉": "cool", "平": "even"
}

taste_map = {
    "酸": "sour", "苦": "bitter", "甘": "sweet", "辛": "pungent", "咸": "salty","涩": "sour"
}

meridian_map = {
    "肺": "Lun.M", "心包": "P.M", "心": "H.M", "大肠": "Lar.I.M",
    "三焦": "T.B.M", "小肠": "Sma.I.M", "胃": "Sto.M", "胆": "G.M",
    "膀胱": "B.M", "脾": "Spl.M", "肝": "Liv.M", "肾": "K.M"
}

# 提取所有唯一的中药名称
all_herbs = set()
for _, row in herb_pairs.iterrows():
    herb1 = row.iloc[2]  # 第一个中药名称
    herb2 = row.iloc[6]  # 第二个中药名称
    all_herbs.add(herb1)
    all_herbs.add(herb2)

# 处理每种中药的数据
processed_data = []
for herb in all_herbs:
    # 找到包含该中药的所有行
    herb_rows = herb_pairs[(herb_pairs.iloc[:, 2] == herb) | (herb_pairs.iloc[:, 6] == herb)]

    # 初始化属性集合
    natures = set()
    tastes = set()
    meridians = set()

    # 收集属性
    for _, row in herb_rows.iterrows():
        # 检查是第一个还是第二个中药
        if row.iloc[2] == herb:
            nature_col = 3
            taste_col = 4
            meridian_col = 5
        else:
            nature_col = 7
            taste_col = 8
            meridian_col = 9

        # 处理可能包含多个值的属性
        nature = str(row.iloc[nature_col])
        taste = str(row.iloc[taste_col])
        meridian = str(row.iloc[meridian_col])

        # 添加到集合中
        if nature != 'nan':
            natures.update(nature.split(','))
        if taste != 'nan':
            tastes.update(taste.split(','))
        if meridian != 'nan':
            meridians.update(meridian.split(','))

    # 转换为英文/缩写并合并
    latin_name = latin_map.get(herb, herb)  # 如果没有找到拉丁文名称，使用中文名称
    nature_en = ','.join([nature_map.get(n, n) for n in natures])
    taste_en = ','.join([taste_map.get(t, t) for t in tastes])
    meridian_en = ','.join([meridian_map.get(m, m) for m in meridians])

    # 添加到结果列表
    processed_data.append([latin_name, nature_en, taste_en, meridian_en])

# 创建DataFrame并保存到CSV
result_df = pd.DataFrame(processed_data, columns=['Latin Name', 'Nature', 'Taste', 'Meridian'])
result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"处理完成，结果已保存到 {output_csv}")
