import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# 定义激活函数（sigmoid函数）
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


# 定义带 L2 正则化的逻辑回归损失函数
def logistic_loss(X, y, theta, lambda_=0.1):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    reg_term = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    loss = -1 / m * np.sum(y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10)) + reg_term
    return loss


# 梯度下降法实现带 L2 正则化的逻辑回归
def gradient_descent(X, y, learning_rate=0.01, num_iterations=1000, lambda_=0.1, tol=1e-4):
    m, n = X.shape
    theta = np.zeros(n)
    prev_loss = np.inf
    for i in range(num_iterations):
        h = sigmoid(np.dot(X, theta))
        reg_theta = theta.copy()
        reg_theta[0] = 0
        gradient = (np.dot(X.T, (h - y)) + lambda_ * reg_theta) / m
        theta -= learning_rate * gradient
        current_loss = logistic_loss(X, y, theta, lambda_)
        if np.abs(current_loss - prev_loss) < tol:
            break
        prev_loss = current_loss
    return theta


# 牛顿法实现带 L2 正则化的逻辑回归
def newton_method(X, y, num_iterations=100, lambda_=0.1, tol=1e-4):
    m, n = X.shape
    theta = np.zeros(n)
    prev_loss = np.inf
    for i in range(num_iterations):
        h = sigmoid(np.dot(X, theta))
        reg_theta = theta.copy()
        reg_theta[0] = 0
        gradient = (np.dot(X.T, (h - y)) + lambda_ * reg_theta) / m
        hessian = np.dot(X.T * h * (1 - h), X) / m
        reg_hessian = np.eye(n) * lambda_ / m
        reg_hessian[0, 0] = 0
        hessian += reg_hessian
        try:
            delta_theta = np.linalg.solve(hessian, gradient)
            theta -= delta_theta
            current_loss = logistic_loss(X, y, theta, lambda_)
            if np.abs(current_loss - prev_loss) < tol:
                break
            prev_loss = current_loss
        except np.linalg.LinAlgError:
            print("Hessian矩阵不可逆，终止迭代")
            break
    return theta


# 拟牛顿法（BFGS）实现带 L2 正则化的逻辑回归
def bfgs(X, y, num_iterations=100, lambda_=0.1, tol=1e-4):
    m, n = X.shape
    theta = np.zeros(n)
    I = np.eye(n)
    H = I
    prev_loss = np.inf
    for i in range(num_iterations):
        h = sigmoid(np.dot(X, theta))
        reg_theta = theta.copy()
        reg_theta[0] = 0
        g = (np.dot(X.T, (h - y)) + lambda_ * reg_theta) / m
        if np.linalg.norm(g) < tol:
            break
        p = -np.dot(H, g)

        # 回溯线搜索
        alpha = 1
        rho = 0.5
        c = 0.1
        while True:
            new_theta = theta + alpha * p
            new_h = sigmoid(np.dot(X, new_theta))
            new_g = (np.dot(X.T, (new_h - y)) + lambda_ * new_theta) / m
            new_loss = logistic_loss(X, y, new_theta, lambda_)
            if new_loss <= prev_loss + c * alpha * np.dot(g.T, p):
                break
            alpha *= rho

        s = alpha * p
        y_ = new_g - g
        rho = 1 / (np.dot(y_.T, s))
        A1 = I - rho * np.outer(s, y_)
        A2 = I - rho * np.outer(y_, s)
        H = np.dot(A1, np.dot(H, A2)) + rho * np.outer(s, s)
        theta = new_theta
        current_loss = logistic_loss(X, y, theta, lambda_)
        if np.abs(current_loss - prev_loss) < tol:
            break
        prev_loss = current_loss
    return theta


# 定义 Softmax 函数
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# 定义带 L2 正则化的 Softmax 回归损失函数
def softmax_loss(X, y, theta, lambda_=0.1):
    m = len(y)
    num_classes = len(np.unique(y))
    y_one_hot = np.eye(num_classes)[y]
    z = np.dot(X, theta)
    probs = softmax(z)
    reg_term = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    loss = -1 / m * np.sum(y_one_hot * np.log(probs + 1e-10)) + reg_term
    return loss


# 梯度下降法实现带 L2 正则化的 Softmax 回归
def softmax_gradient_descent(X, y, learning_rate=0.01, num_iterations=1000, lambda_=0.1, tol=1e-4):
    m, n = X.shape
    num_classes = len(np.unique(y))
    theta = np.zeros((n, num_classes))
    prev_loss = np.inf
    for i in range(num_iterations):
        z = np.dot(X, theta)
        probs = softmax(z)
        y_one_hot = np.eye(num_classes)[y]
        reg_theta = theta.copy()
        reg_theta[0] = 0
        gradient = (np.dot(X.T, (probs - y_one_hot)) + lambda_ * reg_theta) / m
        theta -= learning_rate * gradient
        current_loss = softmax_loss(X, y, theta, lambda_)
        if np.abs(current_loss - prev_loss) < tol:
            break
        prev_loss = current_loss
    return theta


# 预测函数
def predict(X, theta):
    if len(theta.shape) == 1:  # 二分类情况
        z = np.dot(X, theta)
        h = sigmoid(z)
        return np.where(h > 0.5, 1, 0)
    else:  # 多分类情况
        z = np.dot(X, theta)
        probs = softmax(z)
        return np.argmax(probs, axis=1)


# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 添加截距项
X = np.c_[np.ones(len(X)), X]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 正则化参数
lambda_ = 0.1

# 分别使用三种方法训练二分类逻辑回归模型（这里将类别 0 作为正类，其余作为负类）
mask = y_train == 0
y_train_binary = np.where(mask, 1, 0)

theta_gd = gradient_descent(X_train, y_train_binary, lambda_=lambda_)
theta_newton = newton_method(X_train, y_train_binary, lambda_=lambda_)
theta_bfgs = bfgs(X_train, y_train_binary, lambda_=lambda_)

# 计算二分类逻辑回归测试集损失和指标
mask = y_test == 0
y_test_binary = np.where(mask, 1, 0)

loss_gd = logistic_loss(X_test, y_test_binary, theta_gd, lambda_)
y_pred_gd = predict(X_test, theta_gd)
accuracy_gd = accuracy_score(y_test_binary, y_pred_gd)
recall_gd = recall_score(y_test_binary, y_pred_gd)
f1_gd = f1_score(y_test_binary, y_pred_gd)
cm_gd = confusion_matrix(y_test_binary, y_pred_gd)

loss_newton = logistic_loss(X_test, y_test_binary, theta_newton, lambda_)
y_pred_newton = predict(X_test, theta_newton)
accuracy_newton = accuracy_score(y_test_binary, y_pred_newton)
recall_newton = recall_score(y_test_binary, y_pred_newton)
f1_newton = f1_score(y_test_binary, y_pred_newton)
cm_newton = confusion_matrix(y_test_binary, y_pred_newton)

loss_bfgs = logistic_loss(X_test, y_test_binary, theta_bfgs, lambda_)
y_pred_bfgs = predict(X_test, theta_bfgs)
accuracy_bfgs = accuracy_score(y_test_binary, y_pred_bfgs)
recall_bfgs = recall_score(y_test_binary, y_pred_bfgs)
f1_bfgs = f1_score(y_test_binary, y_pred_bfgs)
cm_bfgs = confusion_matrix(y_test_binary, y_pred_bfgs)

print(f"梯度下降法 - 二分类：参数: {theta_gd}，损失值: {loss_gd}，准确率: {accuracy_gd}，召回率: {recall_gd}，F1值: {f1_gd}")
print(f"牛顿法 - 二分类：参数: {theta_newton}，损失值: {loss_newton}，准确率: {accuracy_newton}，召回率: {recall_newton}，F1值: {f1_newton}")
print(f"BFGS拟牛顿法 - 二分类：参数: {theta_bfgs}，损失值: {loss_bfgs}，准确率: {accuracy_bfgs}，召回率: {recall_bfgs}，F1值: {f1_bfgs}")

# 使用梯度下降法训练多分类Softmax回归模型
theta_softmax = softmax_gradient_descent(X_train, y_train, lambda_=lambda_)

# 计算多分类Softmax回归测试集损失和指标
loss_softmax = softmax_loss(X_test, y_test, theta_softmax, lambda_)
y_pred_softmax = predict(X_test, theta_softmax)
accuracy_softmax = accuracy_score(y_test, y_pred_softmax)
recall_softmax = recall_score(y_test, y_pred_softmax, average='weighted')
f1_softmax = f1_score(y_test, y_pred_softmax, average='weighted')
cm_softmax = confusion_matrix(y_test, y_pred_softmax)

# 明确Softmax回归的分类结果是几类
num_classes_softmax = len(np.unique(y))
print(f"Softmax回归的分类结果是 {num_classes_softmax} 类")

print(f"Softmax回归 - 多分类：参数形状: {theta_softmax.shape}，损失值: {loss_softmax}，准确率: {accuracy_softmax}，召回率: {recall_softmax}，F1值: {f1_softmax}")

# 绘制混淆矩阵
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.heatmap(cm_gd, annot=True, fmt='d', cmap='Blues')
plt.title(f'Gradient Descent - Binary Classification\nAccuracy: {accuracy_gd:.2f}, F1: {f1_gd:.2f}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(2, 2, 2)
sns.heatmap(cm_newton, annot=True, fmt='d', cmap='Blues')
plt.title(f'Newton\'s Method - Binary Classification\nAccuracy: {accuracy_newton:.2f}, F1: {f1_newton:.2f}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(2, 2, 3)
sns.heatmap(cm_bfgs, annot=True, fmt='d', cmap='Blues')
plt.title(f'BFGS Quasi-Newton - Binary Classification\nAccuracy: {accuracy_bfgs:.2f}, F1: {f1_bfgs:.2f}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(2, 2, 4)
sns.heatmap(cm_softmax, annot=True, fmt='d', cmap='Blues')
plt.title(f'Softmax Regression - Multi-class Classification\nAccuracy: {accuracy_softmax:.2f}, F1: {f1_softmax:.2f}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()
