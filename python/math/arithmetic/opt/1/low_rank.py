import numpy as np
from sklearn.datasets import make_low_rank_matrix
from numpy.linalg import norm


def low_rank_matrix_recovery(M, omega, r, alpha=0.1, beta=0.1, max_iter=100, tol=1e-6):
    """
    低秩矩阵恢复函数
    :param M: 观测到的部分矩阵
    :param omega: 观测位置的掩码矩阵（元素为1表示对应位置有观测值，0表示无观测值）
    :param r: 低秩矩阵的秩
    :param alpha: 正则化参数L
    :param beta: 正则化参数R
    :param max_iter: 最大迭代次数
    :param tol: 收敛阈值
    :return: 恢复后的矩阵X（通过L和R计算得到），矩阵L，矩阵R
    """
    m, n = M.shape
    L = np.random.rand(m, r)
    R = np.random.rand(n, r)
    prev_loss = np.inf
    for iter in range(max_iter):
        # 固定R，更新L
        L = np.linalg.inv(R.T @ R + alpha * np.eye(r)) @ R.T @ (omega * M).T
        L = L.T

        # 固定L，更新R
        R = (np.linalg.inv(L.T @ L + beta * np.eye(r)) @ L.T @ (omega * M)).T

        # 计算当前的目标函数值
        X = L @ R.T
        loss = np.sum((omega * (X - M)) ** 2) + alpha * norm(L, 'fro') ** 2 + beta * norm(R, 'fro') ** 2

        if iter > 0:
            if np.abs(prev_loss - loss) < tol:
                break
        prev_loss = loss

    return L @ R.T, L, R


# 生成低秩矩阵及观测矩阵
m, n, r = 100, 80, 2  # 矩阵的行数、列数、秩
true_rank_matrix = make_low_rank_matrix(m, n, effective_rank=r, tail_strength=0.0)

# 随机生成观测位置
sampling_rate = 1
omega = np.random.choice([0, 1], size=(m, n), p=[1 - sampling_rate, sampling_rate])
# 得到观测矩阵M
M = omega * true_rank_matrix

# 进行低秩矩阵恢复
recovered_matrix, L, R = low_rank_matrix_recovery(M, omega, r)

# 打印恢复效果（可计算恢复矩阵与原矩阵的误差等指标）
error = np.linalg.norm(true_rank_matrix - recovered_matrix) / np.linalg.norm(true_rank_matrix)
print(f"恢复矩阵与原矩阵的相对误差: {error}")