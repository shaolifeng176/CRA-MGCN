import pandas as pd
from collections import defaultdict

# 文件路径
input_file = r'C:\project\python\pythonProject\re\keti\3维柱状图\有效药对\调气四气五味.csv'
output_file = r'C:\project\python\pythonProject\re\keti\3维柱状图\调气\调气四气组合.csv'

# 读取CSV文件
try:
    df = pd.read_csv(input_file)

    # 确保列索引正确（第四列索引为3，第八列索引为7）
    col4 = df.columns[3]  # 第四列列名
    col8 = df.columns[7]  # 第八列列名

    # 统计组合数量（不考虑顺序）
    combination_counts = defaultdict(int)

    for _, row in df.iterrows():
        qi1 = str(row[col4]).strip()  # 第四列四气属性
        qi2 = str(row[col8]).strip()  # 第八列四气属性

        # 创建不考虑顺序的组合键
        combination = frozenset({qi1, qi2})

        # 统计组合出现次数
        combination_counts[combination] += 1

    # 准备结果数据
    results = []
    for combo, count in combination_counts.items():
        combo_list = sorted(combo)
        if len(combo_list) == 1:  # 相同属性的组合
            results.append({"组合": f"{combo_list[0]} & {combo_list[0]}", "数量": count})
        else:  # 不同属性的组合
            results.append({"组合": f"{combo_list[0]} & {combo_list[1]}", "数量": count})

    # 创建DataFrame并排序
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by="数量", ascending=False)

    # 保存结果
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"统计完成！结果已保存到 {output_file}")
    print("\n四气组合统计结果：")
    print(result_df.to_string(index=False))

except FileNotFoundError:
    print(f"错误：文件 {input_file} 未找到！")
except Exception as e:
    print(f"发生错误：{str(e)}")