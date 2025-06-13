# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import auc
# import seaborn as sns
# from matplotlib.font_manager import FontProperties
# import warnings
# warnings.filterwarnings("ignore")
# # 设置字体
# plt.rcParams["font.family"] = ["SimHei"]  # 中文字体
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#
# # 创建英文专用字体
# english_font = FontProperties(fname=r'C:\Windows\Fonts\times.ttf')  # Windows系统下Times New Roman字体路径
#
#
# # 如果在Linux/Mac系统上，可以使用:
# # english_font = FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf')
#
# def plot_roc_curves_with_auc(data_file, output_file=None):
#     """
#     绘制多个模型的ROC曲线并计算AUC值
#
#     参数:
#     data_file: CSV文件路径，包含Model、FPR、TPR三列
#     output_file: 图像保存路径，若为None则显示图像
#     """
#     # 读取数据
#     df = pd.read_csv(data_file)
#
#     # 确保数据包含所需列
#     required_columns = ['Model', 'FPR', 'TPR']
#     if not all(col in df.columns for col in required_columns):
#         raise ValueError(f"数据文件必须包含{required_columns}列")
#
#     # 按模型分组
#     models = df['Model'].unique()
#
#     # 创建画布
#     plt.figure(figsize=(10, 8))
#
#     # 绘制对角线（随机分类器）
#     plt.plot([0, 1], [0, 1], 'k--', color='gray')
#
#     # 存储各模型的AUC值
#     model_auc = {}
#
#     # 绘制每个模型的ROC曲线
#     for model in models:
#         model_data = df[df['Model'] == model]
#         # 按FPR排序（确保曲线平滑）
#         model_data = model_data.sort_values('FPR')
#
#         # 计算AUC
#         model_auc[model] = auc(model_data['FPR'], model_data['TPR'])
#
#         # 绘制ROC曲线
#         plt.plot(model_data['FPR'], model_data['TPR'], label=f'{model} (AUC={model_auc[model]:.4f})',
#                  linewidth=2)
#
#     # 设置图表属性
#     plt.xlim([-0.01, 1.01])
#     plt.ylim([-0.01, 1.01])
#
#     # 使用英文专用字体设置坐标轴标签
#     plt.xlabel('False Positive Rate', fontproperties=english_font,size=20)
#     plt.ylabel('True Positive Rate', fontproperties=english_font,size=20)
#
#     # 使用英文专用字体设置标题
#     plt.title('Mean Receiver Operating Characteristic', fontproperties=english_font,size=20)
#
#     # 使用英文专用字体设置图例
#     legend = plt.legend(loc='lower right')
#     for text in legend.get_texts():
#         text.set_fontproperties(english_font)
#
#     plt.grid(True, linestyle='--', alpha=0.7)
#
#     # 添加AUC值表格
#     auc_df = pd.DataFrame.from_dict(model_auc, orient='index', columns=['AUC值'])
#     auc_df = auc_df.sort_values('AUC值', ascending=False)
#
#     print("各模型AUC值:")
#     print(auc_df)
#
#     # 保存或显示图像
#     if output_file:
#         plt.savefig(output_file, dpi=1000, bbox_inches='tight')
#         print(f"ROC曲线已保存至: {output_file}")
#     else:
#         plt.show()
#
#     return model_auc, auc_df
#
#
# if __name__ == "__main__":
#     # 替换为你的数据文件路径
#     data_file = r"C:\project\python\pythonProject\re\keti\codetest3\roc_data_origin_avg.csv"
#     # 替换为你想保存的图像路径
#     output_image = "roc_curves_comparison.png"
#
#     try:
#         model_auc, auc_df = plot_roc_curves_with_auc(data_file, output_image)
#     except Exception as e:
#         print(f"执行过程中出错: {e}")
#
#
#
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import seaborn as sns
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings("ignore")

# 设置字体
plt.rcParams["font.family"] = ["SimHei"]  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建英文专用字体
english_font = FontProperties(fname=r'C:\Windows\Fonts\times.ttf')  # Windows系统下Times New Roman字体路径

def plot_roc_curves_with_auc(data_file, output_file=None, legend_fontsize=14, tick_fontsize=12):
    """
    绘制多个模型的ROC曲线并计算AUC值

    参数:
    data_file: CSV文件路径，包含Model、FPR、TPR三列
    output_file: 图像保存路径，若为None则显示图像
    legend_fontsize: 图例文字大小
    tick_fontsize: 坐标轴刻度字体大小
    """
    # 读取数据
    df = pd.read_csv(data_file)

    # 确保数据包含所需列
    required_columns = ['Model', 'FPR', 'TPR']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"数据文件必须包含{required_columns}列")

    # 按模型分组
    models = df['Model'].unique()

    # 创建画布
    plt.figure(figsize=(10, 8))

    # 绘制对角线（随机分类器）
    plt.plot([0, 1], [0, 1], 'k--', color='gray')

    # 存储各模型的AUC值
    model_auc = {}

    # 绘制每个模型的ROC曲线
    for model in models:
        model_data = df[df['Model'] == model]
        # 按FPR排序（确保曲线平滑）
        model_data = model_data.sort_values('FPR')

        # 计算AUC
        model_auc[model] = auc(model_data['FPR'], model_data['TPR'])

        # 绘制ROC曲线
        plt.plot(model_data['FPR'], model_data['TPR'], label=f'{model} (AUC={model_auc[model]:.4f})',
                 linewidth=2)

    # 设置图表属性
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])

    # 使用英文专用字体设置坐标轴标签
    plt.xlabel('False Positive Rate', fontproperties=english_font, size=20)
    plt.ylabel('True Positive Rate', fontproperties=english_font, size=20)

    # 使用英文专用字体设置标题
    plt.title('Mean Receiver Operating Characteristic', fontproperties=english_font, size=20)

    # 设置坐标轴刻度数字字体为Times New Roman，并设置字体大小
    for tick in plt.gca().get_xticklabels():
        tick.set_fontproperties(english_font)
        tick.set_fontsize(tick_fontsize)
    for tick in plt.gca().get_yticklabels():
        tick.set_fontproperties(english_font)
        tick.set_fontsize(tick_fontsize)

    # 使用英文专用字体设置图例，并设置图例文字大小
    legend = plt.legend(loc='lower right', fontsize=legend_fontsize)
    for text in legend.get_texts():
        text.set_fontproperties(english_font)

    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加AUC值表格
    auc_df = pd.DataFrame.from_dict(model_auc, orient='index', columns=['AUC值'])
    auc_df = auc_df.sort_values('AUC值', ascending=False)

    print("各模型AUC值:")
    print(auc_df)

    # 保存或显示图像
    if output_file:
        plt.savefig(output_file, dpi=1000, bbox_inches='tight')
        print(f"ROC曲线已保存至: {output_file}")
    else:
        plt.show()

    return model_auc, auc_df


if __name__ == "__main__":
    # 替换为你的数据文件路径
    data_file = r"C:\project\python\pythonProject\re\keti\codetest2 - 副本\roc_curve_data.csv"
    # 替换为你想保存的图像路径
    output_image = "roc_curves_comparison.png"

    # 自定义参数
    legend_fontsize = 20  # 图例文字大小
    tick_fontsize = 16    # 坐标轴刻度字体大小

    try:
        model_auc, auc_df = plot_roc_curves_with_auc(data_file, output_image, legend_fontsize, tick_fontsize)
    except Exception as e:
        print(f"执行过程中出错: {e}")