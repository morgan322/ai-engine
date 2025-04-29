import math
from graphviz import Digraph
from sklearn.datasets import load_iris

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

def conditional_entropy(data, feature_index):
    feature_values = set(row[feature_index] for row in data)
    cond_ent = 0
    total_count = len(data)
    for value in feature_values:
        subset = [row for row in data if row[feature_index] == value]
        prob = len(subset) / total_count
        cond_ent += prob * entropy(subset)
    return cond_ent

def information_gain(data, feature_index):
    return entropy(data) - conditional_entropy(data, feature_index)

def choose_best_feature(data):
    num_features = len(data[0]) - 1
    best_gain = -1
    best_feature_index = -1
    for i in range(num_features):
        gain = information_gain(data, i)
        if gain > best_gain:
            best_gain = gain
            best_feature_index = i
    return best_feature_index, best_gain

def majority_vote(labels):
    label_counts = {}
    for label in labels:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    return max(label_counts, key=label_counts.get)

def build_tree(data, feature_names, graph, parent_node=None, edge_label=None):
    labels = [row[-1] for row in data]
    if labels.count(labels[0]) == len(labels):
        leaf_node = f"{labels[0]}"
        graph.node(leaf_node)
        if parent_node:
            graph.edge(parent_node, leaf_node, label=edge_label)
        return labels[0]
    if len(data[0]) == 1:
        majority_label = majority_vote(labels)
        leaf_node = f"{majority_label}"
        graph.node(leaf_node)
        if parent_node:
            graph.edge(parent_node, leaf_node, label=edge_label)
        return majority_label
    best_feature_index, best_gain = choose_best_feature(data)
    best_feature_name = feature_names[best_feature_index]
    node_label = f"{best_feature_name}\nIG: {best_gain:.4f}"
    graph.node(node_label)
    if parent_node:
        graph.edge(parent_node, node_label, label=edge_label)
    best_feature = [row[best_feature_index] for row in data]
    unique_values = set(best_feature)
    tree = {}
    tree[best_feature_index] = {}
    new_feature_names = feature_names[:best_feature_index] + feature_names[best_feature_index + 1:]
    for value in unique_values:
        subset = [row for row in data if row[best_feature_index] == value]
        new_data = [row[:best_feature_index] + row[best_feature_index + 1:] for row in subset]
        subtree = build_tree(new_data, new_feature_names, graph, parent_node=node_label, edge_label=str(value))
        tree[best_feature_index][value] = subtree
    return tree

def predict(tree, feature_names, sample):
    while isinstance(tree, dict):
        feature_index = list(tree.keys())[0]
        feature_name = feature_names[feature_index]
        feature_value = sample[feature_index]
        if feature_value in tree[feature_index]:
            tree = tree[feature_index][feature_value]
        else:
            break
    return tree

# 加载鸢尾花数据集
iris = load_iris()
data = iris.data.tolist()
for i in range(len(data)):
    data[i].append(iris.target[i])

feature_names = iris.feature_names

# 创建 Graphviz 图
graph = Digraph(comment='ID3 Decision Tree')
graph.attr(rankdir='TB', size='8,8')

# 构建决策树并可视化
my_tree = build_tree(data, feature_names, graph)

# 保存并渲染图
graph.render('id3_iris_decision_tree.gv', view=True)

# 测试样本
test_sample = iris.data[0]
result = predict(my_tree, feature_names, test_sample)
print(f"预测结果: {iris.target_names[result]}")    