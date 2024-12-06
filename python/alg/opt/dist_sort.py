import numpy as np
import matplotlib.pyplot as plt

# 原始数据点
distances_cm = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
pixels = np.array([600, 565, 450, 385, 345, 312, 288, 270, 257, 245])

# 使用多项式拟合
degree = 3  # 多项式的度数，根据需要可以调整
coefficients = np.polyfit(distances_cm, pixels, degree)
polynomial = np.poly1d(coefficients)

# 计算5厘米刻度的距离
new_distances_cm = np.arange(10, 101, 5)
new_pixels = polynomial(new_distances_cm)

# 打印结果
for dist, pix in zip(new_distances_cm, new_pixels):
    print(f"{dist} cm: {pix:.2f} pixels")

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(distances_cm, pixels, 'o', label='原始数据')
plt.plot(new_distances_cm, new_pixels, '-', label='拟合数据')
plt.xlabel('距离 (cm)')
plt.ylabel('像素')
plt.title('距离与像素的关系 (曲线拟合)')
plt.legend()
plt.grid(True)
plt.show()
