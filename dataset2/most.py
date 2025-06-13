# # # 成分
# import pandas as pd
# from collections import defaultdict
# import re
#
# # 读取Excel文件
# df = pd.read_excel('气味归经去重_更新.xlsx')
#
# # 初始化统计字典
# component_counter = defaultdict(int)
# herb_component_dict = defaultdict(list)
#
# # 处理成分数据
# for index, row in df.iterrows():
#     herb_name = str(row.iloc[0]).strip()  # 第一列是中药名
#     components = str(row.iloc[4]) if len(row) > 4 else ''  # 第五列是成分
#
#     # 使用正则表达式分割成分（支持;或|分隔）
#     if components and components.lower() != 'nan':
#         component_list = re.split(r'[;|]', components)
#         component_list = [c.strip() for c in component_list if c.strip()]
#
#         # 统计成分和中药对应关系
#         for component in component_list:
#             component_counter[component] += 1
#             if herb_name not in herb_component_dict[component]:
#                 herb_component_dict[component].append(herb_name)
#
# # 获取频率最高的10种成分
# top_10 = sorted(component_counter.items(), key=lambda x: x[1], reverse=True)[:10]
#
# # 准备结果DataFrame
# result_data = []
# for rank, (component, count) in enumerate(top_10, 1):
#     herbs = herb_component_dict[component]
#     result_data.append({
#         '排名': rank,
#         '成分名称': component,
#         '出现次数': count,
#         '包含中药数量': len(herbs),
#         '中药列表': '; '.join(herbs)
#     })
#
# result_df = pd.DataFrame(result_data)
#
# # 保存到CSV文件
# result_df.to_csv('高频成分统计结果.csv', index=False, encoding='utf-8-sig')
# print("统计结果已保存到'高频成分统计结果.csv'")
#
import csv
from collections import Counter

# 读取文件并解析数据
def read_ingredients_target(file_path):
    ingredients = []
    targets = []
    with open(file_path, 'r', encoding='utf-8') as file:
        # 跳过表头
        next(file)
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                ingredient = parts[0].strip()
                target_list = parts[1].strip().split('|')
                ingredients.append((ingredient, target_list))
    return ingredients

# 统计靶点的频率
def count_targets(ingredients):
    target_counter = Counter()
    target_to_ingredients = {}

    for ingredient, targets in ingredients:
        for target in targets:
            target_counter[target] += 1
            if target not in target_to_ingredients:
                target_to_ingredients[target] = []
            target_to_ingredients[target].append(ingredient)

    return target_counter, target_to_ingredients

# 获取最常见的10个靶点
def get_top_10_targets(target_counter):
    return target_counter.most_common(10)

# 写入CSV文件
def write_to_csv(top_targets, target_to_ingredients, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['靶点', '出现次数', '包含的成分'])

        for target, count in top_targets:
            ingredients_list = ', '.join(target_to_ingredients[target])
            writer.writerow([target, count, ingredients_list])

# 主程序
def main(input_file, output_file):
    # 读取数据
    ingredients = read_ingredients_target(input_file)

    # 统计靶点
    target_counter, target_to_ingredients = count_targets(ingredients)

    # 获取最常见的10个靶点
    top_targets = get_top_10_targets(target_counter)

    # 写入结果到CSV文件
    write_to_csv(top_targets, target_to_ingredients, output_file)

# 输入输出文件路径
input_file = 'ingredients_target.txt'
output_file = 'top_10_targets.csv'

# 执行程序
main(input_file, output_file)
