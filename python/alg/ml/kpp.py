import time
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def distEclud(vecA,vecB):
    """
    计算两个向量的欧式距离
    """
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))

def get_closest_dist(point, centroids):
    """
    计算样本点与当前已有聚类中心之间的最短距离
    """
    min_dist = np.inf  # 初始设为无穷大
    for i, centroid in enumerate(centroids):
        dist = distEclud(np.array(centroid), np.array(point))
        if dist < min_dist:
            min_dist = dist
    return min_dist

def RWS(P, r):
    """利用轮盘法选择下一个聚类中心"""
    q = 0  # 累计概率
    for i in range(len(P)):
        q += P[i]  # P[i]表示第i个个体被选中的概率
        if i == (len(P) - 1):  # 对于由于概率计算导致累计概率和小于1的，设置为1
            q = 1
        if r <= q:  # 产生的随机数在m~m+P[i]间则认为选中了i
            return i

def getCent(dataSet, k):
    """
    按K-Means++算法生成k个点作为质心
    """
    n = dataSet.shape[1]  # 获取数据的维度
    m = dataSet.shape[0]  # 获取数据的数量
    centroids = np.mat(np.zeros((k, n)))
    # 1. 随机选出一个样本点作为第一个聚类中心
    index = np.random.randint(0, n, size=1)
    centroids[0, :] = dataSet[index, :]
    d = np.mat(np.zeros((m, 1)))  # 初始化D(x)
    for j in range(1, k):
        # 2. 计算D(x)
        for i in range(m):
            d[i, 0] = get_closest_dist(dataSet[i], centroids)  # 与最近一个聚类中心的距离
        # 3. 计算概率
        P = np.square(d) / np.square(d).sum()
        r = np.random.random()  # r为0至1的随机数
        choiced_index = RWS(P, r)  # 利用轮盘法选择下一个聚类中心
        centroids[j, :] = dataSet[choiced_index]
    return centroids

def kMeans_plus2(dataSet,k,distMeas=distEclud):
    """
    k-Means++聚类算法,返回最终的k各质心和点的分配结果
    """
    m = dataSet.shape[0]  #获取样本数量
    # 构建一个簇分配结果矩阵，共两列，第一列为样本所属的簇类值，第二列为样本到簇质心的误差
    clusterAssment = np.mat(np.zeros((m,2)))
    # 1. 初始化k个质心
    centroids = getCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            # 2. 找出最近的质心
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 3. 更新每一行样本所属的簇
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:]=minIndex,minDist**2
        print(centroids) # 打印质心
        # 4. 更新质心
        for cent in range(k):
            ptsClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]] # 获取给定簇的所有点
            centroids[cent,:] = np.mean(ptsClust,axis=0) # 沿矩阵列的方向求均值
    return centroids,clusterAssment

def plotResult(myCentroids,clustAssing,X):
    """将结果用图展示出来"""
    centroids = myCentroids.A  # 将matrix转换为ndarray类型
    # 获取聚类后的样本所属的簇值，将matrix转换为ndarray
    y_kmeans = clustAssing[:, 0].A[:, 0]
    # 未聚类前的数据分布
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], s=50)
    plt.title("未聚类前的数据分布")
    plt.subplots_adjust(wspace=0.5)

    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, alpha=0.5)
    plt.title("用K-Means++算法原理聚类的效果")
    plt.show()

def load_data_make_blobs():
    """
    生成模拟数据
    """
    from sklearn.datasets import make_blobs  # 导入产生模拟数据的方法
    k = 5  # 给定聚类数量
    X, Y = make_blobs(n_samples=1000, n_features=2, centers=k, random_state=1)
    return X,k

if __name__ == '__main__':
    X, k=load_data_make_blobs()  # 获取模拟数据和聚类数量
    s = time.time()
    myCentroids, clustAssing = kMeans_plus2(X, k, distMeas=distEclud) # myCentroids为簇质心
    print("用K-Means++算法原理聚类耗时：", time.time() - s)
    plotResult(myCentroids, clustAssing, X)