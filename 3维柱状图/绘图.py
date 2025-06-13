import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 准备数据
x_categories = ['Cold', 'Cool', 'Even', 'Warm', 'Hot']
y_categories = ['Cold', 'Cool', 'Even', 'Warm', 'Hot']
x_pos, y_pos = np.meshgrid(np.arange(len(x_categories)), np.arange(len(y_categories)))
x_pos = x_pos.flatten()
y_pos = y_pos.flatten()

# z_heights = np.array([66, 2, 10, 10, 0,
#                       2, 0, 0, 8, 0,
#                       10, 0, 0, 0, 0,
#                       10, 8, 0, 0, 0,
#                       0, 0, 0, 0, 0])#清热

z_heights = np.array([0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0,
                      0, 0, 2, 4, 8,
                      0, 0, 4, 24, 38,
                      0, 0, 8, 38, 4])#温热

width = depth = 0.4

# 2. 创建颜色列表
num_bars = len(z_heights)
default_color = 'skyblue'  # 默认颜色
colors = [default_color] * num_bars

# 3. 指定特定柱子的颜色（例如，将第10个柱子设为#0085c1）
colors[0] = '#00a5ce'
colors[1] = '#0085c1'
colors[5] = '#0085c1'
colors[2] = '#2ddaa7'
colors[6] = '#2ddaa7'
colors[10] = '#2ddaa7'
colors[3] = '#f9f871'
colors[7] = '#f9f871'
colors[11] = '#f9f871'
colors[15] = '#f9f871'
colors[4] = '#e59c24'
colors[8] = '#e59c24'
colors[12] = '#e59c24'
colors[16] = '#e59c24'
colors[20] = '#e59c24'
colors[9] = '#8e63b4'
colors[13] = '#8e63b4'
colors[17] = '#8e63b4'
colors[21] = '#8e63b4'
colors[14] = '#eee8a9'
colors[18] = '#eee8a9'
colors[22] = '#eee8a9'
colors[19] = '#eefbff'
colors[23] = '#eefbff'
colors[24] = '#d84c60'


# 4. 创建3D绘图对象
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 5. 绘制3D柱状图
ax.bar3d(x_pos, y_pos, np.zeros_like(z_heights),
         width, depth, z_heights,
         color=colors, edgecolor='black')

# 6. 设置标签和视角
ax.set_xticks(np.arange(len(x_categories)))
ax.set_xticklabels(x_categories)
ax.set_xlabel('X Category')

ax.set_yticks(np.arange(len(y_categories)))
ax.set_yticklabels(y_categories)
ax.set_ylabel('Y Category')

ax.set_zlabel('Herb Pairs')
# ax.set_title('3D Bar Chart with Custom Color')

ax.view_init(azim=45, elev=30)

# 7. 显示图形
plt.tight_layout()
plt.show()