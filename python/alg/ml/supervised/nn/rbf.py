import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 定义 RBF 核函数 （Gaussian Kernel）
def rbf_kernel(x, c, s): 
    return np.exp(-1 / (2 * s**2) * (np.linalg.norm(x - c))**2)

# 定义 RBF 神经网络类
class RBFNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # 使用 K-Means 初始化中心点 or random sample
        self.kmeans = KMeans(n_clusters=hidden_size, random_state=42)
        self.centers = None
        # 初始化标准差
        self.sigmas = np.random.rand(hidden_size)
        # 随机初始化权重
        self.weights = np.random.rand(hidden_size, output_size)

    def initialize_centers(self, X): 
        self.kmeans.fit(X)
        self.centers = self.kmeans.cluster_centers_

    def forward(self, X):
        G = np.zeros((X.shape[0], self.hidden_size))
        for i in range(X.shape[0]):
            for j in range(self.hidden_size):
                G[i, j] = rbf_kernel(X[i], self.centers[j], self.sigmas[j])
        # 计算输出
        output = np.dot(G, self.weights)
        return output

    def train(self, X, y, learning_rate, epochs, decay_rate=0.0001):
        losses = []
        self.initialize_centers(X)
        for epoch in range(epochs):
            current_learning_rate = learning_rate / (1 + decay_rate * epoch)
            for i in range(X.shape[0]):
                # 前向传播
                x = X[i].reshape(1, -1)
                y_true = y[i].reshape(1, -1)
                output = self.forward(x)

                # 计算误差
                error = output - y_true

                # 计算梯度
                G = np.zeros((1, self.hidden_size))
                for j in range(self.hidden_size):
                    G[0, j] = rbf_kernel(x, self.centers[j], self.sigmas[j])
                delta_weights = np.dot(G.T, error)

                # 更新权重
                self.weights -= current_learning_rate * delta_weights

            # 计算当前 epoch 的损失
            outputs = self.forward(X)
            loss = np.mean((outputs - y) ** 2)
            losses.append(loss)
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

        return losses


# 加载 Iris 数据集
iris = load_iris()
X = iris.data
y = iris.target

# 对标签进行 one-hot 编码
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 RBF 神经网络实例
input_size = X_train.shape[1]
hidden_size = 15  # 增加隐藏层神经元数量
output_size = y_train.shape[1]
rbf_net = RBFNet(input_size, hidden_size, output_size)

# 训练模型
learning_rate = 0.1
epochs = 200
losses = rbf_net.train(X_train, y_train, learning_rate, epochs)

# 在测试集上进行预测
test_outputs = rbf_net.forward(X_test)
test_predictions = np.argmax(test_outputs, axis=1)
test_labels = np.argmax(y_test, axis=1)

# 计算准确率
accuracy = np.mean(test_predictions == test_labels)
print(f'Test Accuracy: {accuracy}')

# 可视化损失曲线
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
