import pandas as pd


def excel_to_csv(input_excel, output_csv):
    """
    将Excel文件转换为CSV文件

    参数:
        input_excel (str): 输入的Excel文件路径
        output_csv (str): 输出的CSV文件路径
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(input_excel)

        # 保存为CSV文件，使用utf-8-sig编码确保Excel能正确识别中文
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')

        print(f"成功将 {input_excel} 转换为 {output_csv}")
        print(f"转换后的CSV文件编码: UTF-8 with BOM")

    except Exception as e:
        print(f"转换过程中出现错误: {e}")
        # 尝试使用GBK编码作为备选方案
        try:
            df.to_csv(output_csv, index=False, encoding='gbk')
            print("已使用GBK编码作为备选方案保存CSV文件")
        except Exception as e2:
            print(f"备选方案也失败: {e2}")


# 使用示例
if __name__ == "__main__":
    input_file = "气味归经去重_更新.xlsx"  # 输入的Excel文件
    output_file = "气味归经去重_更新1.csv"  # 输出的CSV文件

    excel_to_csv(input_file, output_file)