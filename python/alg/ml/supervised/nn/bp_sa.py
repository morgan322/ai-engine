import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import random


# 定义 BP 神经网络模型
class BPNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 加载 Iris 数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据转换为张量
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)


# 计算准确率
def accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred.data, 1)
    total = y_true.size(0)
    correct = (predicted == y_true).sum().item()
    return correct / total


# 模拟退火算法
def simulated_annealing(model, X_train, y_train, X_test, y_test, T=1000, T_min=1, alpha=0.99):
    best_model = model
    best_acc = accuracy(model(X_test), y_test)
    current_model = model
    current_acc = best_acc

    while T > T_min:
        # 随机生成隐藏层大小
        new_hidden_size = random.randint(5, 20)
        new_model = BPNet(input_size=4, hidden_size=new_hidden_size, output_size=3)

        optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(100):
            optimizer.zero_grad()
            outputs = new_model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        new_acc = accuracy(new_model(X_test), y_test)

        delta = new_acc - current_acc
        if delta > 0 or random.random() < np.exp(delta / T):
            current_model = new_model
            current_acc = new_acc

        if current_acc > best_acc:
            best_model = current_model
            best_acc = current_acc

        T = T * alpha

    return best_model, best_acc


# 初始化 BP 神经网络
input_size = 4
hidden_size = 10
output_size = 3
model = BPNet(input_size, hidden_size, output_size)

# 运行模拟退火算法
best_model, best_acc = simulated_annealing(model, X_train, y_train, X_test, y_test)

print(f"Best accuracy: {best_acc}")
