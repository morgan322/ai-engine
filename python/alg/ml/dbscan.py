"Density-Based Spatial Clustering of Applications with Noise"
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

def dbscan(data, eps, min_points):
    """
    DBSCAN 聚类算法实现
    :param data: 数据集，二维 numpy 数组，每行代表一个数据点
    :param eps: Epsilon 邻域半径
    :param min_points: 最少点数目
    :return: 聚类标签数组，-1 表示异常点
    """
    n = data.shape[0]
    labels = np.full(n, -1)  # 初始化所有点的标签为 -1（异常点）
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue  # 如果该点已经被处理过，跳过

        # 计算当前点的 Epsilon 邻域内的点
        neighbors = get_neighbors(data, i, eps)

        if len(neighbors) < min_points:
            labels[i] = -1  # 不是核心点，标记为异常点
            continue

        # 是核心点，开始一个新的聚类
        cluster_id += 1
        labels[i] = cluster_id
        seeds = list(neighbors)

        while seeds:
            j = seeds.pop(0)
            if labels[j] == -1:
                labels[j] = cluster_id
                new_neighbors = get_neighbors(data, j, eps)
                if len(new_neighbors) >= min_points:
                    seeds.extend(new_neighbors)

    return labels


def get_neighbors(data, index, eps):
    """
    获取指定点的 Epsilon 邻域内的点的索引
    :param data: 数据集，二维 numpy 数组，每行代表一个数据点
    :param index: 指定点的索引
    :param eps: Epsilon 邻域半径
    :return: 邻域内点的索引列表
    """
    distances = euclidean_distances(data, data[index].reshape(1, -1)).flatten()
    return np.where(distances <= eps)[0]


if __name__ == "__main__":
    # 示例数据集
    data = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    # Epsilon 邻域半径
    eps = 3
    # 最少点数目
    min_points = 2

    labels = dbscan(data, eps, min_points)
    print("聚类标签:", labels)

    # 可视化聚类结果
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # 异常点用黑色表示
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xy = data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

    plt.title('DBSCAN Clustering')
    plt.show()
    