import math
import copy
from graphviz import Digraph
from sklearn.datasets import load_iris
import numpy as np


# 计算信息熵
def entropy(data):
    label_counts = {}
    total_count = len(data)
    for row in data:
        label = row[-1]
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    ent = 0
    for count in label_counts.values():
        p = count / total_count
        ent -= p * math.log2(p)
    return ent


# 计算条件熵
def conditional_entropy(data, feature_index):
    feature_values = set(row[feature_index] for row in data)
    cond_ent = 0
    total_count = len(data)
    for value in feature_values:
        subset = [row for row in data if row[feature_index] == value]
        prob = len(subset) / total_count
        cond_ent += prob * entropy(subset)
    return cond_ent


# 计算信息增益
def information_gain(data, feature_index):
    return entropy(data) - conditional_entropy(data, feature_index)


# 计算分裂信息
def split_information(data, feature_index):
    feature_values = set(row[feature_index] for row in data)
    split_info = 0
    total_count = len(data)
    for value in feature_values:
        subset = [row for row in data if row[feature_index] == value]
        prob = len(subset) / total_count
        split_info -= prob * math.log2(prob)
    return split_info


# 计算信息增益率
def information_gain_ratio(data, feature_index):
    gain = information_gain(data, feature_index)
    split_info = split_information(data, feature_index)
    if split_info == 0:
        return 0
    return gain / split_info


# 选择最佳特征
def choose_best_feature(data):
    num_features = len(data[0]) - 1
    best_gain_ratio = -1
    best_feature_index = -1
    for i in range(num_features):
        gain_ratio = information_gain_ratio(data, i)
        if gain_ratio > best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_feature_index = i
    return best_feature_index


# 多数表决确定叶节点类别
def majority_vote(labels):
    label_counts = {}
    for label in labels:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    return max(label_counts, key=label_counts.get)


# 递归构建决策树
def build_tree(data, feature_names):
    labels = [row[-1] for row in data]
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    if len(data[0]) == 1:
        return majority_vote(labels)
    best_feature_index = choose_best_feature(data)
    best_feature_name = feature_names[best_feature_index]
    tree = {best_feature_name: {}}
    feature_values = set(row[best_feature_index] for row in data)
    new_feature_names = feature_names[:best_feature_index] + feature_names[best_feature_index + 1:]
    for value in feature_values:
        subset = [row for row in data if row[best_feature_index] == value]
        new_data = [row[:best_feature_index] + row[best_feature_index + 1:] for row in subset]
        subtree = build_tree(new_data, new_feature_names)
        tree[best_feature_name][value] = subtree
    return tree


# 预测
def predict(tree, feature_names, sample):
    while isinstance(tree, dict):
        feature_name = list(tree.keys())[0]
        feature_index = feature_names.index(feature_name)
        feature_value = sample[feature_index]
        if feature_value in tree[feature_name]:
            tree = tree[feature_name][feature_value]
        else:
            break
    return tree


# 可视化决策树
def visualize_tree(tree, feature_names, graph, parent_node=None, edge_label=None):
    if not isinstance(tree, dict):
        leaf_node = f"{tree}"
        graph.node(leaf_node)
        if parent_node:
            graph.edge(parent_node, leaf_node, label=edge_label)
        return

    feature_name = list(tree.keys())[0]
    node_label = f"{feature_name}"
    graph.node(node_label)
    if parent_node:
        graph.edge(parent_node, node_label, label=edge_label)

    sub_tree = tree[feature_name]
    feature_index = feature_names.index(feature_name)
    for value, subtree in sub_tree.items():
        new_feature_names = feature_names[:feature_index] + feature_names[feature_index + 1:]
        visualize_tree(subtree, new_feature_names, graph, parent_node=node_label, edge_label=str(value))


# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# 为了方便处理，将特征值离散化
num_bins = 3
X_discretized = np.zeros_like(X)
for i in range(X.shape[1]):
    bins = np.linspace(X[:, i].min(), X[:, i].max(), num_bins + 1)
    X_discretized[:, i] = np.digitize(X[:, i], bins)

# 合并特征和标签
data = np.hstack((X_discretized, y.reshape(-1, 1)))
data = data.tolist()

# 创建 Graphviz 图
graph = Digraph(comment='C4.5 Decision Tree for Iris Dataset')
graph.attr(rankdir='TB', size='8,8')

# 构建决策树
my_tree = build_tree(data, feature_names)

# 可视化决策树
visualize_tree(my_tree, feature_names, graph)

# 保存并渲染图
graph.render('c45_iris_decision_tree.gv', view=True)

# 测试样本（这里取第一个样本）
test_sample = X_discretized[0].tolist()
result = predict(my_tree, feature_names, test_sample)
print(f"预测结果: {result}，实际类别: {y[0]}")
    