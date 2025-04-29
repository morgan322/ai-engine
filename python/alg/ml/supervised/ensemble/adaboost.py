from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 生成示例数据集
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)
                                                                                                                                                                                                                                                                                                                                                                                                                                   
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建基分类器
base_classifier = DecisionTreeClassifier(max_depth=1)

# 创建 AdaBoost 分类器
ada_boost = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=50, random_state=42)

# 训练模型
ada_boost.fit(X_train, y_train)

# 进行预测
y_pred = ada_boost.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"AdaBoost 准确率: {accuracy}")
