import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as mcolors

# 设置字体以确保显示正常
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['animation.html'] = 'jshtml'

# 优化器基类
class Optimizer:
    def __init__(self, params, lr=0.01, name="Optimizer", loss_fn=None):
        self.params = np.array(params, dtype=np.float32)
        self.lr = lr
        self.name = name
        self.loss_fn = loss_fn
        self.history = [self.params.copy()]  # 跟踪参数更新路径
        self.loss_history = [self._compute_loss()]  # 跟踪损失值

    def _compute_loss(self):
        """计算当前参数的损失值"""
        if self.loss_fn is None:
            raise ValueError("未提供损失函数")
        return self.loss_fn(self.params)

    def step(self):
        """执行一步优化更新"""
        raise NotImplementedError

    def get_history(self):
        """返回参数更新历史"""
        return np.array(self.history)

    def get_loss_history(self):
        """返回损失值历史"""
        return np.array(self.loss_history)

# 1. 梯度下降
class GradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, loss_fn=None, grad_fn=None):
        super().__init__(params, lr, "Gradient Descent", loss_fn=loss_fn)
        self.grad_fn = grad_fn

    def step(self):
        grad = self.grad_fn(self.params)
        self.params -= self.lr * grad
        self.history.append(self.params.copy())
        self.loss_history.append(self._compute_loss())

# 2. 随机梯度下降
class StochasticGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, loss_fn=None, grad_fn=None):
        super().__init__(params, lr, "SGD", loss_fn=loss_fn)
        self.grad_fn = grad_fn

    def step(self):
        grad = self.grad_fn(self.params)
        self.params -= self.lr * grad
        self.history.append(self.params.copy())
        self.loss_history.append(self._compute_loss())

# 3. 小批量梯度下降
class MiniBatchGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, loss_fn=None, grad_fn=None):
        super().__init__(params, lr, "Mini-Batch GD", loss_fn=loss_fn)
        self.grad_fn = grad_fn

    def step(self):
        grad = self.grad_fn(self.params)
        self.params -= self.lr * grad
        self.history.append(self.params.copy())
        self.loss_history.append(self._compute_loss())

# 4. 带动量的SGD
class MomentumSGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, loss_fn=None, grad_fn=None):
        super().__init__(params, lr, "Momentum", loss_fn=loss_fn)
        self.momentum = momentum
        self.v = np.zeros_like(self.params)
        self.grad_fn = grad_fn

    def step(self):
        grad = self.grad_fn(self.params)
        self.v = self.momentum * self.v + self.lr * grad
        self.params -= self.v
        self.history.append(self.params.copy())
        self.loss_history.append(self._compute_loss())

# 5. NAG (Nesterov加速梯度)
class NAG(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, loss_fn=None, grad_fn=None):
        super().__init__(params, lr, "NAG", loss_fn=loss_fn)
        self.momentum = momentum
        self.v = np.zeros_like(self.params)
        self.grad_fn = grad_fn

    def step(self):
        # 前瞻点
        lookahead = self.params - self.momentum * self.v
        grad = self.grad_fn(lookahead)
        self.v = self.momentum * self.v + self.lr * grad
        self.params -= self.v
        self.history.append(self.params.copy())
        self.loss_history.append(self._compute_loss())

# 6. AdaGrad (自适应梯度下降)
class AdaGrad(Optimizer):
    def __init__(self, params, lr=0.01, eps=1e-8, loss_fn=None, grad_fn=None):
        super().__init__(params, lr, "AdaGrad", loss_fn=loss_fn)
        self.eps = eps
        self.s = np.zeros_like(self.params)
        self.grad_fn = grad_fn

    def step(self):
        grad = self.grad_fn(self.params)
        self.s += grad ** 2
        adaptive_lr = self.lr / (np.sqrt(self.s) + self.eps)
        self.params -= adaptive_lr * grad
        self.history.append(self.params.copy())
        self.loss_history.append(self._compute_loss())

# 7. RMSProp (均方根传播)
class RMSProp(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.9, eps=1e-8, loss_fn=None, grad_fn=None):
        super().__init__(params, lr, "RMSProp", loss_fn=loss_fn)
        self.alpha = alpha
        self.eps = eps
        self.s = np.zeros_like(self.params)
        self.grad_fn = grad_fn

    def step(self):
        grad = self.grad_fn(self.params)
        self.s = self.alpha * self.s + (1 - self.alpha) * (grad ** 2)
        adaptive_lr = self.lr / (np.sqrt(self.s) + self.eps)
        self.params -= adaptive_lr * grad
        self.history.append(self.params.copy())
        self.loss_history.append(self._compute_loss())

# 8. AdaDelta
class AdaDelta(Optimizer):
    def __init__(self, params, lr=1.0, rho=0.95, eps=1e-6, loss_fn=None, grad_fn=None):
        super().__init__(params, lr, "AdaDelta", loss_fn=loss_fn)
        self.rho = rho
        self.eps = eps
        self.s = np.zeros_like(self.params)
        self.delta = np.zeros_like(self.params)
        self.grad_fn = grad_fn

    def step(self):
        grad = self.grad_fn(self.params)
        self.s = self.rho * self.s + (1 - self.rho) * (grad ** 2)
        update = - (np.sqrt(self.delta + self.eps) / np.sqrt(self.s + self.eps)) * grad
        self.params += update
        self.delta = self.rho * self.delta + (1 - self.rho) * (update ** 2)
        self.history.append(self.params.copy())
        self.loss_history.append(self._compute_loss())

# 9. Adam (自适应矩估计)
class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, loss_fn=None, grad_fn=None):
        super().__init__(params, lr, "Adam", loss_fn=loss_fn)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros_like(self.params)
        self.v = np.zeros_like(self.params)
        self.t = 0
        self.grad_fn = grad_fn

    def step(self):
        self.t += 1
        grad = self.grad_fn(self.params)
        
        # 更新一阶矩估计
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        
        # 更新二阶矩估计
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        # 偏差校正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # 参数更新
        self.params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        self.history.append(self.params.copy())
        self.loss_history.append(self._compute_loss())

# 10. AdamW (带权重衰减的Adam)
class AdamW(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, loss_fn=None, grad_fn=None):
        super().__init__(params, lr, "AdamW", loss_fn=loss_fn)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = np.zeros_like(self.params)
        self.v = np.zeros_like(self.params)
        self.t = 0
        self.grad_fn = grad_fn

    def step(self):
        self.t += 1
        grad = self.grad_fn(self.params)
        
        # 权重衰减
        self.params -= self.lr * self.weight_decay * self.params
        
        # 更新一阶矩估计
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        
        # 更新二阶矩估计
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        # 偏差校正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # 参数更新
        self.params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        self.history.append(self.params.copy())
        self.loss_history.append(self._compute_loss())

# 定义非凸损失函数（Rosenbrock函数）
def rosenbrock_function(theta):
    """Rosenbrock函数: f(x,y) = (1-x)² + 100(y-x²)², 最小值在(1,1)"""
    x, y = theta
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_gradient(theta):
    """Rosenbrock函数的梯度"""
    x, y = theta
    dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    dy = 200 * (y - x ** 2)
    return np.array([dx, dy])

# 定义鞍点损失函数
def saddle_function(theta):
    """鞍点函数: f(x,y) = x² - y², 鞍点在(0,0)"""
    x, y = theta
    return x ** 2 - y ** 2

def saddle_gradient(theta):
    """鞍点函数的梯度"""
    x, y = theta
    return np.array([2 * x, -2 * y])

# 3D优化过程可视化函数
def visualize_optimization_3d(loss_fn, grad_fn, initial_params, title, min_point=None):
    # 初始化优化器，使用精心调整的学习率
    optimizers = {
        "Gradient Descent": GradientDescent(initial_params.copy(), lr=0.001, loss_fn=loss_fn, grad_fn=grad_fn),
        "SGD": StochasticGradientDescent(initial_params.copy(), lr=0.001, loss_fn=loss_fn, grad_fn=grad_fn),
        "Mini-Batch GD": MiniBatchGradientDescent(initial_params.copy(), lr=0.001, loss_fn=loss_fn, grad_fn=grad_fn),
        "Momentum": MomentumSGD(initial_params.copy(), lr=0.001, momentum=0.9, loss_fn=loss_fn, grad_fn=grad_fn),
        "NAG": NAG(initial_params.copy(), lr=0.001, momentum=0.9, loss_fn=loss_fn, grad_fn=grad_fn),
        "AdaGrad": AdaGrad(initial_params.copy(), lr=0.1, loss_fn=loss_fn, grad_fn=grad_fn),
        "RMSProp": RMSProp(initial_params.copy(), lr=0.01, alpha=0.9, loss_fn=loss_fn, grad_fn=grad_fn),
        "AdaDelta": AdaDelta(initial_params.copy(), lr=1.0, rho=0.95, loss_fn=loss_fn, grad_fn=grad_fn),
        "Adam": Adam(initial_params.copy(), lr=0.01, beta1=0.9, beta2=0.999, loss_fn=loss_fn, grad_fn=grad_fn),
        "AdamW": AdamW(initial_params.copy(), lr=0.01, beta1=0.9, beta2=0.999, weight_decay=0.001, loss_fn=loss_fn, grad_fn=grad_fn)
    }

    # 生成高分辨率的3D曲面网格
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = loss_fn([X[i, j], Y[i, j]])
    
    # 判断函数是否有负值
    has_negative = np.any(Z < 0)
    
    # 创建3D图形，改进尺寸和dpi
    fig = plt.figure(figsize=(14, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # 使用适当的颜色映射和归一化
    if has_negative:
        # 对于有负值的函数（如鞍点函数）
        cmap = plt.get_cmap('coolwarm')
        norm = mcolors.Normalize(vmin=Z.min(), vmax=Z.max())
    else:
        # 对于仅含正值的函数（如Rosenbrock）
        cmap = plt.get_cmap('plasma')
        norm = mcolors.LogNorm(vmin=Z.min(), vmax=Z.max())
    
    # 绘制3D曲面，改进光照和颜色
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.7, linewidth=0, antialiased=True,
                          rstride=1, cstride=1, norm=norm)
    
    # 添加颜色条，带标签和样式
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.05)
    cbar.set_label('Loss Value', fontsize=12, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)
    
    # 设置标签和标题，改进格式
    ax.set_xlabel('Parameter x', fontsize=12, labelpad=10)
    ax.set_ylabel('Parameter y', fontsize=12, labelpad=10)
    ax.set_zlabel('Loss', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=16, pad=20)
    
    # 如果提供了最小值点，则标记出来
    if min_point is not None:
        ax.scatter(min_point[0], min_point[1], loss_fn(min_point), color='red', s=150, 
                  marker='*', label='Minimum', edgecolors='black', linewidth=1.5)
    
    # 设置网格样式
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 设置初始视角
    ax.view_init(elev=30, azim=45)
    
    # 使用高对比度、易分辨的颜色方案
    # 这些颜色在视觉上有明显差异，适合区分不同优化器
    colors = [
        '#FF0000',   # 红色 - 容易识别
        '#00FF00',   # 绿色 - 与红色形成鲜明对比
        '#0000FF',   # 蓝色 - 基本色，易识别
        '#FFFF00',   # 黄色 - 明亮，易识别
        '#FF00FF',   # 洋红色 - 独特，易区分
        '#00FFFF',   # 青色 - 独特，与蓝色不同
        '#FF8000',   # 橙色 - 温暖色调，易识别
        '#8000FF',   # 紫色 - 独特，易区分
        '#008000',   # 深绿 - 与浅绿区分
        '#000080'    # 深蓝 - 与浅蓝区分
    ]
    
    # 初始化轨迹线和点，改进样式
    trajectories = {}
    current_points = {}
    labels = list(optimizers.keys())
    
    for i, (name, opt) in enumerate(optimizers.items()):
        line, = ax.plot([], [], [], color=colors[i], label=name, linewidth=2.5, alpha=0.9)
        point, = ax.plot([], [], [], 'o', color=colors[i], markersize=8, alpha=1.0, 
                        markeredgecolor='black', markeredgewidth=1)
        trajectories[name] = line
        current_points[name] = point
    
    # 添加图例，改进位置和样式
    legend = ax.legend(loc='upper right', frameon=True, framealpha=0.9, 
                      edgecolor='black', fontsize=10, title='Optimizers')
    legend.get_title().set_fontsize(12)
    legend.get_title().set_fontweight('bold')
    
    # 初始化函数
    def init():
        for name in optimizers:
            trajectories[name].set_data([], [])
            trajectories[name].set_3d_properties([])
            current_points[name].set_data([], [])
            current_points[name].set_3d_properties([])
        return list(trajectories.values()) + list(current_points.values())
    
    # 动画更新函数
    def update(frame):
        for name, opt in optimizers.items():
            if frame > 0:  # 跳过第一帧以显示初始状态
                opt.step()
            
            history = opt.get_history()
            if len(history) > 0:
                x_vals = history[:, 0]
                y_vals = history[:, 1]
                z_vals = np.array([loss_fn(p) for p in history])
                
                trajectories[name].set_data(x_vals, y_vals)
                trajectories[name].set_3d_properties(z_vals)
                
                current_points[name].set_data(x_vals[-1:], y_vals[-1:])
                current_points[name].set_3d_properties(z_vals[-1:])
        
        # 平滑相机旋转
        angle = 360 * (frame / 300)
        ax.view_init(elev=30, azim=angle)
        
        # 更新标题显示当前迭代
        ax.set_title(f"{title} (Iteration {frame})", fontsize=16, pad=20)
        
        return list(trajectories.values()) + list(current_points.values())
    
    # 创建动画，改进持续时间和平滑度
    ani = FuncAnimation(fig, update, frames=1000, init_func=init, interval=80, 
                       blit=True, repeat=False)
    
    plt.tight_layout()
    return ani

# 在Rosenbrock函数上可视化优化过程
ani1 = visualize_optimization_3d(
    rosenbrock_function, 
    rosenbrock_gradient, 
    initial_params=np.array([-1.5, 1.5]), 
    title="Optimization on Rosenbrock Function (Non-Convex)",
    min_point=np.array([1, 1])
)

# 在鞍点函数上可视化优化过程
ani2 = visualize_optimization_3d(
    saddle_function, 
    saddle_gradient, 
    initial_params=np.array([1.0, 1.0]), 
    title="Optimization on Saddle Point Function",
    min_point=None  # 鞍点没有全局最小值
)

# 显示动画
plt.show()