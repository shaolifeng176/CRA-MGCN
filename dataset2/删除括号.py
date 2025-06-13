import re


def clean_herb_file(input_file, output_file):
    """
    1. 删除前两列中所有全角/半角括号及括号内内容
    2. 将所有全角/半角问号替换为空格
    """
    with open(input_file, 'r', encoding='utf-8') as fin, \
            open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            # 分割列（假设制表符分隔）
            columns = line.strip().split('\t')

            if len(columns) >= 2:
                # 处理前两列
                for i in range(2):
                    # 删除所有全角/半角括号及内容
                    columns[i] = re.sub(r'[\(（].*?[）)]', '', columns[i])
                    # 替换所有问号为空格
                    columns[i] = columns[i].replace('？', ' ').replace('?', ' ')
                    # 去除多余空格
                    columns[i] = ' '.join(columns[i].split()).strip()

                # 重新组合行
                new_line = '\t'.join(columns)
                fout.write(new_line + '\n')
            else:
                fout.write(line)

    print(f"清洗完成！结果已保存到 {output_file}")


# 使用示例
clean_herb_file("herb_browse_cleaned.txt", "herb_browse_final_cleaned.txt")