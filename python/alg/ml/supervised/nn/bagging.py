from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建决策树分类器作为基学习器
base_estimator = DecisionTreeClassifier()

# 创建 Bagging 分类器
bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=10, random_state=42)

# 在训练集上训练 Bagging 分类器
bagging.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = bagging.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Bagging 分类器的准确率: {accuracy:.2f}")
    