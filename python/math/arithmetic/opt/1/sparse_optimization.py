import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 查找并设置支持中文的字体，Ubuntu下常用的中文字体是SimHei或WenQuanYi Zen Hei
try:
    font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
except:
    print("未找到文泉驿正黑字体，你可以尝试安装该字体或更换其他字体。")
    import sys
    sys.exit(1)

# 设置矩阵和向量的维度
m = 128
n = 256

# 生成一个 m x n 的随机矩阵 A
A = np.random.randn(m, n)
# 生成一个稀疏向量 u，稀疏度为 0.1
u = np.random.randn(n)
u[np.abs(u) < np.percentile(np.abs(u), 90)] = 0
# 计算 b = A * u
b = np.dot(A, u)

# 求解 l1 范数优化问题 (1.2.3)
x_l1 = cp.Variable(n)
obj_l1 = cp.Minimize(cp.norm(x_l1, 1))
constraints_l1 = [A @ x_l1 == b]
prob_l1 = cp.Problem(obj_l1, constraints_l1)
prob_l1.solve()

# 求解 l2 范数优化问题 (1.2.4)
x_l2 = cp.Variable(n)
obj_l2 = cp.Minimize(cp.norm(x_l2, 2))
constraints_l2 = [A @ x_l2 == b]
prob_l2 = cp.Problem(obj_l2, constraints_l2)
prob_l2.solve()

# 可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.stem(u)
plt.title('精确解u', fontproperties=font)
plt.xlabel('索引', fontproperties=font)
plt.ylabel('值', fontproperties=font)

plt.subplot(1, 3, 2)
plt.stem(x_l1.value)
plt.title('l1范数优化', fontproperties=font)
plt.xlabel('索引', fontproperties=font)
plt.ylabel('值', fontproperties=font)

plt.subplot(1, 3, 3)
plt.stem(x_l2.value)
plt.title('l2范数优化', fontproperties=font)
plt.xlabel('索引', fontproperties=font)
plt.ylabel('值', fontproperties=font)

plt.tight_layout()
plt.show()
    