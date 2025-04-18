import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 定义 SVC 类，手动实现支持向量分类器的核心逻辑
class SVCustom:
    def __init__(self, C=1.0, kernel='linear', max_iter=1000):
        # 正则化参数，用于控制间隔最大化和分类误差最小化之间的平衡
        self.C = C
        # 核函数类型，这里仅支持线性核
        self.kernel = kernel
        # 最大迭代次数，防止算法无限循环
        self.max_iter = max_iter
        # 权重向量
        self.w = None
        # 偏置项
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # 初始化权重向量和偏置项
        self.w = np.zeros(n_features)
        self.b = 0

        # 开始迭代训练
        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X):
                # 计算决策函数值
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # 如果样本被正确分类且在间隔外，更新权重向量
                    self.w -= self.C * (2 * 1 / self.max_iter) * self.w
                else:
                    # 如果样本被错误分类或在间隔内，更新权重向量和偏置项
                    self.w -= self.C * (2 * 1 / self.max_iter * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.C * (-y[idx])

    def predict(self, X):
        # 计算决策函数值
        linear_output = np.dot(X, self.w) + self.b
        # 根据决策函数值进行分类，大于 0 为正类，小于 0 为负类
        return np.sign(linear_output)


# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 为了简化，只考虑二分类问题，选择类别 0 和 1
selected_indices = np.logical_or(y == 0, y == 1)
X = X[selected_indices]
y = y[selected_indices]
# 将标签转换为 -1 和 1
y = np.where(y == 0, -1, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模拟部分标签缺失
rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(len(y_train)) < 0.3
# 提取有标签的数据
labeled_X = X_train[~random_unlabeled_points]
labeled_y = y_train[~random_unlabeled_points]
# 提取无标签的数据
unlabeled_X = X_train[random_unlabeled_points]

# 初始使用有标签数据训练一个自定义的 SVM
svm = SVCustom()
svm.fit(labeled_X, labeled_y)

# 对无标签数据进行预测，生成伪标签
pseudo_labels = svm.predict(unlabeled_X)

# 合并有标签数据和伪标签数据
combined_X = np.vstack((labeled_X, unlabeled_X))
combined_y = np.hstack((labeled_y, pseudo_labels))

# 重新训练 SVM
svm.fit(combined_X, combined_y)

# 在测试集上进行预测
y_pred = svm.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"半监督 SVM 的准确率: {accuracy:.2f}")
    