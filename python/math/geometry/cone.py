import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 1. 正定锥（二维矩阵：x>0, z>0, xz>y² ，严格正定 ）
def plot_definite_cone(ax):
    x = np.linspace(0.1, 2, 100)  # x>0，避免除零
    z = np.linspace(0.1, 2, 100)  # z>0
    X, Z = np.meshgrid(x, z)
    Y = np.sqrt(X * Z - 1e-6)     # 严格大于，微小偏移避免数值问题
    Y_lower = -np.sqrt(X * Z - 1e-6)
    
    # 绘制上、下曲面（严格正定区域）
    ax.plot_surface(X, Y, Z, alpha=0.6, cmap=cm.Reds, linewidth=0)
    ax.plot_surface(X, Y_lower, Z, alpha=0.6, cmap=cm.Reds, linewidth=0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Positive Definite Cone (2x2: xz>y², x>0, z>0)')

# 2. 半正定锥（二维矩阵：x≥0, z≥0, xz≥y² ）
def plot_semidefinite_cone(ax):
    x = np.linspace(0, 2, 100)
    z = np.linspace(0, 2, 100)
    X, Z = np.meshgrid(x, z)
    Y = np.sqrt(X * Z)  # 上半部分
    Y_lower = -np.sqrt(X * Z)  # 下半部分（对称）
    
    # 绘制上、下曲面（半正定区域）
    ax.plot_surface(X, Y, Z, alpha=0.6, cmap=cm.Blues, linewidth=0)
    ax.plot_surface(X, Y_lower, Z, alpha=0.6, cmap=cm.Blues, linewidth=0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Positive Semidefinite Cone (2x2: xz≥y², x≥0, z≥0)')

# 3. 二阶锥（二次锥，标准形式：||(x,y)||₂ ≤ z, z≥0 ）
def plot_second_order_cone(ax):
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, 2, 100)
    Theta, Z = np.meshgrid(theta, z)
    X = Z * np.cos(Theta)
    Y = Z * np.sin(Theta)
    
    ax.plot_surface(X, Y, Z, alpha=0.6, cmap=cm.Greens, linewidth=0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Second-Order Cone (||(x,y)||₂ ≤ z, z≥0)')

# 4. 双曲锥（题目定义形式：基于二次型，转化为 x² + y² ≤ z², z≥0 简化示例 ）
def plot_hyperbolic_cone(ax):
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, 2, 100)  # 题目中 c^T x ≥0，取 z≥0 简化
    Theta, Z = np.meshgrid(theta, z)
    X = Z * np.cos(Theta)
    Y = Z * np.sin(Theta)
    
    ax.plot_surface(X, Y, Z, alpha=0.6, cmap=cm.Oranges, linewidth=0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Hyperbolic Cone (x² + y² ≤ z², z≥0)')

# 5. 二次锥（同二阶锥，换个示例：x² + y² + z² ≤ t², t≥0 ，3D 球锥 ）
def plot_quadratic_cone(ax):
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, np.pi/2, 50)  # 上半球
    Theta, Phi = np.meshgrid(theta, phi)
    t = np.linspace(0, 2, 100)
    T, _ = np.meshgrid(t, phi)
    
    X = T * np.sin(Phi) * np.cos(Theta)
    Y = T * np.sin(Phi) * np.sin(Theta)
    Z = T * np.cos(Phi)
    
    ax.plot_surface(X, Y, Z, alpha=0.6, cmap=cm.Purples, linewidth=0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Quadratic Cone (x²+y²+z² ≤ t², t≥0)')

# 绘图布局：5 个子图
fig = plt.figure(figsize=(20, 4))
ax1 = fig.add_subplot(151, projection='3d')
ax2 = fig.add_subplot(152, projection='3d')
ax3 = fig.add_subplot(153, projection='3d')
ax4 = fig.add_subplot(154, projection='3d')
ax5 = fig.add_subplot(155, projection='3d')

plot_definite_cone(ax1)
plot_semidefinite_cone(ax2)
plot_second_order_cone(ax3)
plot_hyperbolic_cone(ax4)
plot_quadratic_cone(ax5)

plt.tight_layout()
plt.show()


# Define the function to plot norm cones
def plot_norm_cone(p, ax):
    """
    Plot the l_p norm cone
    p: norm order (1, 2, 3)
    ax: matplotlib 3D axis
    """
    # Generate grid data
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate the t values for the l_p norm
    if p == 1:
        T = np.abs(X) + np.abs(Y)
    elif p == 2:
        T = np.sqrt(X**2 + Y**2)
    elif p == 3:
        T = (np.abs(X)**3 + np.abs(Y)**3)**(1/3)
    else:
        raise ValueError("Only p=1,2,3 are supported")
    
    # Plot the upper surface of the norm cone
    surf = ax.plot_surface(X, Y, T, alpha=0.7, cmap=cm.coolwarm, 
                           linewidth=0, antialiased=True)
    
    # Set view and labels
    ax.view_init(elev=30, azim=45)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('t')
    ax.set_title(f'$\ell_{p}$ Norm Cone: $||(x,y)||_{p} \leq t$')
    
    # Add color bar
    cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('t value')

# Create figure and subplots
fig = plt.figure(figsize=(18, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

# Plot the three norm cones
plot_norm_cone(1, ax1)
plot_norm_cone(2, ax2)
plot_norm_cone(3, ax3)

plt.tight_layout()
plt.show()