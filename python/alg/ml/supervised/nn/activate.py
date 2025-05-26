import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 设置图片清晰度和字体大小
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12  # 增大基础字体大小
plt.rcParams['axes.labelsize'] = 14  # 增大坐标轴标签字体
plt.rcParams['axes.titlesize'] = 16  # 增大标题字体
plt.rcParams['xtick.labelsize'] = 12  # 增大x轴刻度字体
plt.rcParams['ytick.labelsize'] = 12  # 增大y轴刻度字体

# 生成x值（范围扩大到-20到20）
x = np.linspace(-20, 20, 4000)  # 增加点数使曲线更平滑
dx = 0.001  # 用于数值计算导数的步长

# 定义激活函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.1):
    return np.maximum(alpha * x, x)

def leaky_relu_derivative(x, alpha=0.1):
    return np.where(x > 0, 1, alpha)

def prelu(x, alpha=0.25):
    return np.maximum(alpha * x, x)

def prelu_derivative(x, alpha=0.25):
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

def rrelu(x, alpha=0.229):
    return np.where(x > 0, x, alpha * x)

def rrelu_derivative(x, alpha=0.229):
    return np.where(x > 0, 1, alpha)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x):
    # 近似计算GELU导数
    c = np.sqrt(2 / np.pi)
    return 0.5 * (1 + np.tanh(c * (x + 0.044715 * x**3))) + 0.5 * x * (1 - np.tanh(c * (x + 0.044715 * x**3))**2) * c * (1 + 0.134145 * x**2)

def swish(x, beta=1.0):
    return x * sigmoid(beta * x)

def swish_derivative(x, beta=1.0):
    return beta * swish(x, beta) + sigmoid(beta * x) * (1 - beta * swish(x, beta))

def selu(x, alpha=1.67326, scale=1.0507):
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu_derivative(x, alpha=1.67326, scale=1.0507):
    return scale * np.where(x > 0, 1, alpha * np.exp(x))

# 激活函数列表及参数
activation_functions = [
    (sigmoid, sigmoid_derivative, "Sigmoid"),
    (tanh, tanh_derivative, "Tanh"),
    (relu, relu_derivative, "ReLU"),
    (leaky_relu, leaky_relu_derivative, "Leaky ReLU (α = 0.1)"),
    (prelu, prelu_derivative, "PReLU (α = 0.25)"),
    (elu, elu_derivative, "ELU (α = 1.0)"),
    (rrelu, rrelu_derivative, "RReLU (α = 0.229)"),
    (gelu, gelu_derivative, "GELU"),
    (swish, swish_derivative, "Swish (β = 1.0)"),
    (selu, selu_derivative, "SELU (α = 1.67326, scale = 1.0507)")
]

# 获取多种颜色
colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.BASE_COLORS.values())

# 为每个函数创建单独的图像
for i, (func, derivative, name) in enumerate(activation_functions):
    # 创建更大的画布
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 计算函数值
    x_range = x  # 所有函数使用相同的x范围
    
    y = func(x_range)
    y_derivative = derivative(x_range)
    
    # 绘制原函数
    ax1 = axes[0]
    ax1.plot(x_range, y, color=colors[0], linewidth=3)  # 增加线宽
    ax1.set_title(f"{name}", fontsize=18, pad=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.axhline(y=0, color='k', linewidth=1)  # 增加坐标轴粗细
    ax1.axvline(x=0, color='k', linewidth=1)
    
    # 设置x轴范围
    ax1.set_xlim(-20, 20)
    
    # 调整y轴范围以更好展示函数特性
    if name.startswith("Sigmoid") or name.startswith("Tanh"):
        ax1.set_ylim(-1.2, 1.2)
    elif name.startswith("ReLU") or name.startswith("Leaky") or name.startswith("PReLU") or name.startswith("RReLU"):
        ax1.set_ylim(-2, 20)  # 扩大上限
    elif name.startswith("ELU"):
        ax1.set_ylim(-2, 20)  # 扩大上限
    elif name.startswith("GELU"):
        ax1.set_ylim(-10, 20)  # 调整范围
    elif name.startswith("Swish"):
        ax1.set_ylim(-20, 20)  # 全范围展示
    elif name.startswith("SELU"):
        ax1.set_ylim(-5, 20)  # 调整范围
    
    # 绘制导函数
    ax2 = axes[1]
    ax2.plot(x_range, y_derivative, color=colors[1], linewidth=3)  # 增加线宽
    ax2.set_title(f"{name} Derivative", fontsize=18, pad=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.axhline(y=0, color='k', linewidth=1)  # 增加坐标轴粗细
    ax2.axvline(x=0, color='k', linewidth=1)
    
    # 设置x轴范围
    ax2.set_xlim(-20, 20)
    
    # 调整导函数y轴范围
    if name.startswith("Sigmoid"):
        ax2.set_ylim(-0.1, 0.3)
    elif name.startswith("Tanh"):
        ax2.set_ylim(-0.1, 1.1)
    elif name.startswith("ReLU"):
        ax2.set_ylim(-0.1, 1.1)
    elif name.startswith("Leaky") or name.startswith("PReLU") or name.startswith("RReLU"):
        ax2.set_ylim(-0.1, 1.1)
    elif name.startswith("ELU"):
        ax2.set_ylim(-0.1, 5.0)  # 扩大上限
    elif name.startswith("GELU"):
        ax2.set_ylim(-0.1, 1.5)  # 调整范围
    elif name.startswith("Swish"):
        ax2.set_ylim(-5, 20)  # 调整范围
    elif name.startswith("SELU"):
        ax2.set_ylim(-0.1, 5.0)  # 调整范围
    
    # 调整布局
    plt.tight_layout(pad=4.0)
    plt.savefig(f'../../../../../data/result/activate/activation_{name.replace(" (α = 0.1)", "").replace(" (α = 0.25)", "").replace(" (α = 0.229)", "").replace(" (β = 1.0)", "").replace(" (α = 1.67326, scale = 1.0507)", "").lower()}.png', bbox_inches='tight', dpi=300)
    plt.close() 