import numpy as np
import matplotlib.pyplot as plt

# 生成输入数据（均值0，方差1）
input_data = np.random.randn(1000, 50)  # 1000个样本，50个特征

# 对比不同初始化方法
init_methods = {
    'Random': lambda n_in, n_out: np.random.randn(n_in, n_out) * 0.1,
    'Xavier': lambda n_in, n_out: np.random.randn(n_in, n_out) * np.sqrt(1/n_in)
}

# 模拟5层网络的前向传播
n_layers = 5
results = {method: [] for method in init_methods}

for method, init_func in init_methods.items():
    x = input_data.copy()
    for i in range(n_layers):
        n_in = x.shape[1]
        n_out = 50  # 每层50个神经元
        W = init_func(n_in, n_out)
        x = np.tanh(np.dot(x, W))  # 使用tanh激活函数
        results[method].append(x)

# 绘制各层激活值的分布
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, method in enumerate(init_methods):
    ax = axes[i]
    ax.set_title(method)
    for layer, data in enumerate(results[method]):
        ax.hist(data.flatten(), bins=30, alpha=0.7, 
                label=f'Layer {layer+1}', density=True)
    ax.legend()
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(0, 3)

plt.tight_layout()
plt.show()