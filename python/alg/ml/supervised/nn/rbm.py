import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from skimage.metrics import structural_similarity as ssim
import os


# 定义限制玻尔兹曼机类
class RBM:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.weights = np.random.normal(0, 0.1, (n_visible, n_hidden))
        self.visible_bias = np.zeros((n_visible, 1))
        self.hidden_bias = np.zeros((n_hidden, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_hidden(self, visible_states):
        activation = np.dot(self.weights.T, visible_states) + self.hidden_bias
        p_hidden = self.sigmoid(activation)
        return (p_hidden > np.random.rand(self.n_hidden, visible_states.shape[1])).astype(int)

    def sample_visible(self, hidden_states):
        activation = np.dot(self.weights, hidden_states) + self.visible_bias
        p_visible = self.sigmoid(activation)
        return (p_visible > np.random.rand(self.n_visible, hidden_states.shape[1])).astype(int)

    def train(self, data, epochs=100, batch_size=128, initial_learning_rate=0.01, decay_rate=0.99):
        n_samples = data.shape[0]
        for epoch in range(epochs):
            learning_rate = initial_learning_rate * (decay_rate ** epoch)
            for i in range(0, n_samples, batch_size):
                batch_data = data[i:i + batch_size].T
                hidden_states = self.sample_hidden(batch_data)
                visible_states_recon = self.sample_visible(hidden_states)
                hidden_states_recon = self.sample_hidden(visible_states_recon)

                positive_association = np.dot(batch_data, hidden_states.T)
                negative_association = np.dot(visible_states_recon, hidden_states_recon.T)
                self.weights += learning_rate * (positive_association - negative_association) / batch_size
                self.visible_bias += learning_rate * np.sum(batch_data - visible_states_recon, axis=1,
                                                            keepdims=True) / batch_size
                self.hidden_bias += learning_rate * np.sum(hidden_states - hidden_states_recon, axis=1,
                                                           keepdims=True) / batch_size
            print(f'Epoch {epoch + 1} completed, learning rate: {learning_rate}')

    def reconstruct(self, data):
        data = data.T
        hidden_states = self.sample_hidden(data)
        return self.sample_visible(hidden_states).T


# 检查本地是否存在 MNIST 数据文件
data_file = '/home/morgan/ubt/data/ml/mnist_784.npz'
if os.path.exists(data_file):
    data = np.load(data_file)
    X = data['data']
else:
    # 加载 MNIST 数据集
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    X = mnist.data
    X = np.array(X, dtype=float)
    # 保存数据到本地
    np.savez(data_file, data=X)

# 数据归一化到 [0, 1]
X = X / 255.0

# 初始化 RBM 模型，784 个可见层神经元，256 个隐藏层神经元
rbm = RBM(n_visible=784, n_hidden=256)
# 训练 RBM
rbm.train(X, epochs=100, batch_size=128, initial_learning_rate=0.01, decay_rate=0.99)

# 测试 RBM，取 10 个样本进行重构
test_data = X[:10]
reconstructed_data = rbm.reconstruct(test_data)

# 计算量化指标
mse_values = []
ssim_values = []
for i in range(10):
    mse = np.mean((test_data[i] - reconstructed_data[i]) ** 2)
    mse_values.append(mse)
    ssim_val = ssim(test_data[i].reshape(28, 28), reconstructed_data[i].reshape(28, 28), data_range=1)
    ssim_values.append(ssim_val)

print(f"均方误差 (MSE) 均值: {np.mean(mse_values)}")
print(f"结构相似性指数 (SSIM) 均值: {np.mean(ssim_values)}")

# 可视化原始图像和重构图像
plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(test_data[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.title('Original')

    plt.subplot(2, 10, i + 11)
    plt.imshow(reconstructed_data[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.title('Reconstructed')

plt.show()
