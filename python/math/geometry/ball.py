import numpy as np
import matplotlib.pyplot as plt

# 1. 绘制 2-范数球（2D 圆）
def plot_2norm_ball(ax, x_c, r, color='b', label='2-Norm Ball'):
    # 生成角度
    theta = np.linspace(0, 2*np.pi, 100)
    # 球的参数方程：x = x_c[0] + r*cos(theta), y = x_c[1] + r*sin(theta)
    x = x_c[0] + r * np.cos(theta)
    y = x_c[1] + r * np.sin(theta)
    ax.plot(x, y, color=color, label=label)
    ax.fill(x, y, color=color, alpha=0.2)
    ax.scatter(x_c[0], x_c[1], color='k', label='Center')  # 画中心

# 2. 绘制椭球（2D 椭圆）
def plot_ellipsoid(ax, x_c, P, color='r', label='Ellipsoid'):
    # 分解 P 为 A*A^T（Cholesky 分解，因 P 正定）
    A = np.linalg.cholesky(P)
    A_inv = np.linalg.inv(A)  # 用于生成椭球的逆变换
    
    # 生成单位圆上的点
    theta = np.linspace(0, 2*np.pi, 100)
    u = np.array([np.cos(theta), np.sin(theta)])
    
    # 椭球的参数方程：x = x_c + A * u
    x_ellipse = x_c[0] + A[0,0]*u[0] + A[0,1]*u[1]
    y_ellipse = x_c[1] + A[1,0]*u[0] + A[1,1]*u[1]
    
    ax.plot(x_ellipse, y_ellipse, color=color, label=label)
    ax.fill(x_ellipse, y_ellipse, color=color, alpha=0.2)
    ax.scatter(x_c[0], x_c[1], color='k')  # 画中心

# 绘图设置
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')  # 保证坐标轴比例一致

# 定义球和椭球的参数
x_c = np.array([0, 0])  # 中心在原点
r_ball = 2  # 球的半径
P_ellipsoid = np.array([[4, 1],  # 正定矩阵 P，这里设置一个椭圆（非圆）
                        [1, 1]])

# 绘制
plot_2norm_ball(ax, x_c, r_ball, color='b', label='2-Norm Ball (Circle)')
plot_ellipsoid(ax, x_c, P_ellipsoid, color='r', label='Ellipsoid')

# 标注
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
plt.grid(True)
plt.show()