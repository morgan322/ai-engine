import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


def gaussian_pdf(x, mean, cov):
    """
    计算高斯分布的概率密度函数
    :param x: 数据点
    :param mean: 均值
    :param cov: 协方差矩阵
    :return: 概率密度
    """
    n = len(x)
    det = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    diff = x - mean
    exponent = -0.5 * np.dot(np.dot(diff.T, inv_cov), diff)
    return (1 / (np.sqrt((2 * np.pi) ** n * det))) * np.exp(exponent)


def em_algorithm(data, num_components, max_iterations=500, tolerance=1e-6, n_init=10, reg_covar=1e-6):
    """
    EM 算法估计高斯混合模型的参数
    :param data: 输入数据
    :param num_components: 高斯分量的数量
    :param max_iterations: 最大迭代次数
    :param tolerance: 收敛阈值
    :param n_init: 初始化次数
    :param reg_covar: 协方差矩阵正则化项
    :return: 权重、均值、协方差矩阵、后验概率
    """
    best_log_likelihood = -np.inf
    best_weights = None
    best_means = None
    best_covs = None
    best_responsibilities = None

    for _ in range(n_init):
        num_samples, num_features = data.shape

        # 使用 K-means 初始化
        kmeans = KMeans(n_clusters=num_components, n_init=10).fit(data)
        labels = kmeans.labels_
        weights = np.array([np.sum(labels == i) / num_samples for i in range(num_components)])
        means = kmeans.cluster_centers_
        covs = []
        for i in range(num_components):
            subset = data[labels == i]
            if len(subset) > 0:
                cov = np.cov(subset.T) + reg_covar * np.eye(num_features)
                covs.append(cov)
            else:
                covs.append(np.eye(num_features))

        log_likelihood = -np.inf
        for iteration in range(max_iterations):
            # E 步
            responsibilities = np.zeros((num_samples, num_components))
            for i in range(num_samples):
                for k in range(num_components):
                    responsibilities[i, k] = weights[k] * gaussian_pdf(data[i], means[k], covs[k])
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)

            # 计算对数似然
            new_log_likelihood = np.sum(np.log(np.sum(responsibilities * weights, axis=1)))

            # 检查收敛
            if np.abs(new_log_likelihood - log_likelihood) < tolerance:
                break
            log_likelihood = new_log_likelihood

            # M 步
            new_weights = responsibilities.sum(axis=0) / num_samples
            new_means = np.zeros((num_components, num_features))
            new_covs = [np.zeros((num_features, num_features)) for _ in range(num_components)]

            for k in range(num_components):
                resp_sum = responsibilities[:, k].sum()
                new_means[k] = np.dot(responsibilities[:, k], data) / resp_sum
                diff = data - new_means[k]
                new_covs[k] = np.dot(responsibilities[:, k] * diff.T, diff) / resp_sum + reg_covar * np.eye(num_features)

            weights = new_weights
            means = new_means
            covs = new_covs

        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_weights = weights
            best_means = means
            best_covs = covs
            best_responsibilities = responsibilities

    return best_weights, best_means, best_covs, best_responsibilities


# 示例数据
np.random.seed(42)
data1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
data2 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 100)
data = np.vstack((data1, data2))
# 为数据添加真实标签
true_labels = np.hstack((np.zeros(100), np.ones(100))).astype(int)

# 使用 EM 算法估计参数
weights, means, covs, responsibilities = em_algorithm(data, num_components=2)

# 根据后验概率预测标签
predicted_labels = np.argmax(responsibilities, axis=1)

# 计算准确率
accuracy = np.mean(true_labels == predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 计算混淆矩阵
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 绘制数据点和拟合的高斯分布
sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=true_labels, palette='viridis', ax=axes[0])
x = np.linspace(-5, 10, 200)
y = np.linspace(-5, 10, 200)
X, Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

for k in range(len(weights)):
    rv = multivariate_normal(means[k], covs[k])
    Z = rv.pdf(pos)
    axes[0].contour(X, Y, Z, levels=5, colors='red', alpha=0.5)

axes[0].set_title('Data with True Labels and Fitted Gaussians')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

# 绘制混淆矩阵
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', ax=axes[1])
axes[1].set_title('Confusion Matrix')
axes[1].set_xlabel('Predicted Labels')
axes[1].set_ylabel('True Labels')

plt.tight_layout()
plt.show()