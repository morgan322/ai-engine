import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成数据
x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)
Z = X ** 2 - Y ** 2  # 定义函数 f(x,y)=x² - y²

# 创建 3D 图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制 3D 曲面，rstride 和 cstride 控制网格线密度，cmap 设置颜色映射
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')  

# 设置坐标轴标签
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# 设置图形标题
ax.set_title('3D plot of $x^2 - y^2$')  

# 添加颜色条，展示颜色与数值的对应关系
fig.colorbar(surf, shrink=0.5, aspect=10)  

# 调整视角，让图形展示更清晰（可根据需要调整 elev 和 azim 参数）
ax.view_init(elev=30, azim=30)  

# 显示图形
plt.show()