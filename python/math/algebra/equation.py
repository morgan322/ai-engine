import numpy as np

# 定义 5 次方程
def fifth_degree_equation(x):
    # 示例方程：f(x) = 2x^5 + 3x^4 - 4x^3 + 5x^2 - 6x + 7
    return 2 * x**5 + 3 * x**4 - 4 * x**3 + 5 * x**2 - 6 * x + 7

# 定义 5 次方程的导数
def derivative_fifth_degree_equation(x):
    # 对示例方程求导：f'(x) = 10x^4 + 12x^3 - 12x^2 + 10x - 6
    return 10 * x**4 + 12 * x**3 - 12 * x**2 + 10 * x - 6

# 梯度下降法求解方程近似解
def gradient_descent(learning_rate=0.001, max_iterations=10000, tolerance=1e-6):
    # 初始化 x
    x = 0
    for i in range(max_iterations):
        # 计算当前的梯度
        gradient = 2 * fifth_degree_equation(x) * derivative_fifth_degree_equation(x)
        # 更新 x
        x = x - learning_rate * gradient
        # 判断是否满足收敛条件
        if np.abs(gradient) < tolerance:
            print(f"在第 {i} 次迭代时收敛")
            break
    return x

# 调用梯度下降法求解近似解
approx_solution = gradient_descent()
print(f"方程的近似解为: {approx_solution}")    
