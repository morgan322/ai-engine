import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 加载 Iris 数据集，只取两类（方便线性可分示例，这里选 setosa 和 versicolor）
iris = load_iris()
X = iris.data[:100, :2]  # 取前 100 条数据（对应两类），并只取前两个特征用于可视化
y = iris.target[:100]
y[y == 0] = -1  # 将标签转为 -1 和 1，符合 SVM 习惯

scaler = StandardScaler()
X = scaler.fit_transform(X)

class SimpleSVM:
    def __init__(self, max_iter=100, tol=1e-3):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.b = 0.0
        self.X = None
        self.y = None
        self.m = None  # 样本数量

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.m = len(y)
        self.alpha = np.zeros(self.m)

        iter_num = 0
        while iter_num < self.max_iter:
            alpha_changed = 0
            for i in range(self.m):
                # 计算预测值和误差
                f_xi = np.sum(self.alpha * self.y * np.dot(self.X, self.X[i].T)) + self.b
                E_i = f_xi - self.y[i]

                # 检查是否满足 KKT 条件（简化版判断）
                if (self.y[i] * E_i < -self.tol and self.alpha[i] < 1) or (self.y[i] * E_i > self.tol and self.alpha[i] > 0):
                    # 随机选另一个变量 j
                    j = np.random.randint(0, self.m)
                    while j == i:
                        j = np.random.randint(0, self.m)

                    f_xj = np.sum(self.alpha * self.y * np.dot(self.X, self.X[j].T)) + self.b
                    E_j = f_xj - self.y[j]

                    alpha_i_old = self.alpha[i].copy()
                    alpha_j_old = self.alpha[j].copy()

                    # 计算上下界 L 和 H
                    if self.y[i] != self.y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(1, 1 + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - 1)
                        H = min(1, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue

                    # 计算 eta
                    eta = 2 * np.dot(self.X[i], self.X[j].T) - np.dot(self.X[i], self.X[i].T) - np.dot(self.X[j], self.X[j].T)
                    if eta >= 0:
                        continue

                    # 更新 alpha_j
                    self.alpha[j] -= self.y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    # 检查 alpha_j 是否有足够变化
                    if abs(self.alpha[j] - alpha_j_old) < self.tol:
                        self.alpha[j] = alpha_j_old
                        continue

                    # 更新 alpha_i
                    self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])

                    # 更新 b
                    b1 = self.b - E_i - self.y[i] * (self.alpha[i] - alpha_i_old) * np.dot(self.X[i], self.X[i].T) - \
                         self.y[j] * (self.alpha[j] - alpha_j_old) * np.dot(self.X[i], self.X[j].T)
                    b2 = self.b - E_j - self.y[i] * (self.alpha[i] - alpha_i_old) * np.dot(self.X[i], self.X[j].T) - \
                         self.y[j] * (self.alpha[j] - alpha_j_old) * np.dot(self.X[j], self.X[j].T)

                    if 0 < self.alpha[i] < 1:
                        self.b = b1
                    elif 0 < self.alpha[j] < 1:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    alpha_changed += 1

            if alpha_changed == 0:
                iter_num += 1
            else:
                iter_num = 0

    def predict(self, X):
        return np.sign(np.sum(self.alpha * self.y * np.dot(X, self.X.T), axis=1) + self.b)
    
# 训练模型
svm = SimpleSVM(max_iter=1000, tol=1e-3)
svm.fit(X, y)

# 预测（这里可视化，所以用训练数据演示，实际可换测试集）
y_pred = svm.predict(X)
accuracy = np.mean(y_pred == y)
print(f"模型准确率: {accuracy:.4f}")
# 可视化决策边界
def plot_decision_boundary(model, X, y):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("SVM Decision Boundary on Iris Dataset (2 features)")
    plt.show()

plot_decision_boundary(svm, X, y)