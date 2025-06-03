import numpy as np

def calculate_matrix_norms(matrix):
    """
    计算矩阵的三种常用范数：Frobenius范数、算子范数（谱范数）和核范数
    
    参数:
    matrix (np.ndarray): 输入的矩阵
    
    返回:
    dict: 包含三种范数计算结果的字典
    """
    # 计算Frobenius范数（元素平方和的平方根）
    frobenius_norm = np.linalg.norm(matrix, 'fro')
    
    # 计算算子范数（谱范数，即最大奇异值）
    operator_norm = np.linalg.norm(matrix, 2)
    
    # 计算核范数（所有奇异值之和）
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    nuclear_norm = np.sum(singular_values)
    
    return {
        'Frobenius范数': frobenius_norm,
        '算子范数（谱范数）': operator_norm,
        '核范数': nuclear_norm
    }

# 示例使用
if __name__ == "__main__":
    x = np.array([3, -4, 12])

    # 计算L1范数
    l1_norm = np.linalg.norm(x, 1)

    # 计算L2范数
    l2_norm = np.linalg.norm(x, 2)

    # 计算L∞范数
    linf_norm = np.linalg.norm(x, np.inf) # 或者使用 np.max(np.abs(x))
    
    # 创建一个示例矩阵
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    norms = calculate_matrix_norms(A)
    
    print("示例矩阵:")
    print(A)
    print("\n计算结果:")
    for norm_type, value in norms.items():
        print(f"{norm_type}: {value:.4f}")    