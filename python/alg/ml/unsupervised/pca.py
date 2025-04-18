"Principal Component Analysis"
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# 手动实现 PCA
def manual_pca(X, n_components):
    # 数据标准化
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    # X_standardized = (X - X_mean) / X_std
    X_standardized = X - X_mean

    # 计算协方差矩阵
    cov_matrix = np.cov(X_standardized.T)

    # 计算协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 对特征值进行排序，并获取对应的索引
    sorted_indices = np.argsort(eigenvalues)[::-1]
    # sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # 选择前 n_components 个特征向量
    top_eigenvectors = sorted_eigenvectors[:, :n_components]

    # 投影变换
    X_pca = np.dot(X_standardized, top_eigenvectors)

    return X_pca


# 示例数据
np.random.seed(42)
X = np.random.randn(100, 3)

# 手动实现 PCA
n_components = 2
X_pca_manual = manual_pca(X, n_components)

# 使用 sklearn 实现 PCA
pca_sklearn = PCA(n_components=n_components)
X_pca_sklearn = pca_sklearn.fit_transform(X)

# 可视化手动实现的 PCA 结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca_manual[:, 0], X_pca_manual[:, 1], alpha=0.7)
plt.title('Manual PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# 可视化 sklearn 实现的 PCA 结果
plt.subplot(1, 2, 2)
plt.scatter(X_pca_sklearn[:, 0], X_pca_sklearn[:, 1], alpha=0.7)
plt.title('Sklearn PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()
    