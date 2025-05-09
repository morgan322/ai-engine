import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine


class DistanceCalculator:
    def __init__(self, point1, point2):
        """
        初始化类，接收两个点作为输入
        :param point1: 第一个点，numpy 数组
        :param point2: 第二个点，numpy 数组
        """
        self.point1 = np.array(point1)
        self.point2 = np.array(point2)

    def euclidean_distance(self):
        """
        计算欧几里得距离
        :return: 欧几里得距离
        """
        return np.linalg.norm(self.point1 - self.point2)

    def manhattan_distance(self):
        """
        计算曼哈顿距离
        :return: 曼哈顿距离
        """
        return np.sum(np.abs(self.point1 - self.point2))

    def chebyshev_distance(self):
        """
        计算切比雪夫距离
        :return: 切比雪夫距离
        """
        return np.max(np.abs(self.point1 - self.point2))

    def cosine_distance(self):
        """
        计算余弦距离
        :return: 余弦距离
        """
        return cosine(self.point1, self.point2)

    def plot_distances(self):
        """
        绘制各种距离的可视化图形
        """
        euclidean = self.euclidean_distance()
        manhattan = self.manhattan_distance()
        chebyshev = self.chebyshev_distance()
        cosine_dist = self.cosine_distance()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 绘制欧几里得距离
        axes[0, 0].scatter(*self.point1, color='blue', label='Point 1')
        axes[0, 0].scatter(*self.point2, color='red', label='Point 2')
        axes[0, 0].plot([self.point1[0], self.point2[0]], [self.point1[1], self.point2[1]],
                        linestyle='--', color='green', label=f'Euclidean Distance: {euclidean:.2f}')
        axes[0, 0].set_title('Euclidean Distance')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        axes[0, 0].legend()

        # 绘制曼哈顿距离
        axes[0, 1].scatter(*self.point1, color='blue', label='Point 1')
        axes[0, 1].scatter(*self.point2, color='red', label='Point 2')
        axes[0, 1].plot([self.point1[0], self.point2[0], self.point2[0]],
                        [self.point1[1], self.point1[1], self.point2[1]],
                        linestyle='--', color='orange', label=f'Manhattan Distance: {manhattan:.2f}')
        axes[0, 1].set_title('Manhattan Distance')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Y')
        axes[0, 1].legend()

        # 绘制切比雪夫距离
        axes[1, 0].scatter(*self.point1, color='blue', label='Point 1')
        axes[1, 0].scatter(*self.point2, color='red', label='Point 2')
        x_max = max(self.point1[0], self.point2[0])
        x_min = min(self.point1[0], self.point2[0])
        y_max = max(self.point1[1], self.point2[1])
        y_min = min(self.point1[1], self.point2[1])
        if np.abs(self.point1[0] - self.point2[0]) > np.abs(self.point1[1] - self.point2[1]):
            axes[1, 0].plot([x_min, x_max, x_max], [y_min, y_min, y_max],
                            linestyle='--', color='purple', label=f'Chebyshev Distance: {chebyshev:.2f}')
        else:
            axes[1, 0].plot([x_min, x_min, x_max], [y_min, y_max, y_max],
                            linestyle='--', color='purple', label=f'Chebyshev Distance: {chebyshev:.2f}')
        axes[1, 0].set_title('Chebyshev Distance')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        axes[1, 0].legend()

        # 绘制余弦距离
        axes[1, 1].quiver(0, 0, *self.point1, angles='xy', scale_units='xy', scale=1, color='blue', label='Vector 1')
        axes[1, 1].quiver(0, 0, *self.point2, angles='xy', scale_units='xy', scale=1, color='red', label='Vector 2')
        axes[1, 1].set_xlim([0, max(self.point1[0], self.point2[0]) + 1])
        axes[1, 1].set_ylim([0, max(self.point1[1], self.point2[1]) + 1])
        axes[1, 1].set_title(f'Cosine Distance: {cosine_dist:.2f}')
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Y')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()


# 示例使用
if __name__ == "__main__":
    point1 = [1, 2]
    point2 = [4, 6]
    calculator = DistanceCalculator(point1, point2)
    calculator.plot_distances()
    

from scipy.stats import norm

# Define parameters for real and generated distributions
mu_r = 1
sigma_r = 0.3
mu_g = 3
sigma_g = 0.3

# Generate sample points
x = np.linspace(-2, 6, 1000)
pr = norm.pdf(x, mu_r, sigma_r)
pg = norm.pdf(x, mu_g, sigma_g)

# Plot distributions
plt.figure(figsize=(10, 6))
plt.plot(x, pr, label='Real Distribution $P_r$')
plt.plot(x, pg, label='Generated Distribution $P_g$')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Example of 1D Normal Distributions')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Discretize into bins
bins = 20
hist_r, bin_edges_r = np.histogram(x, bins=bins, weights=pr)
hist_g, bin_edges_g = np.histogram(x, bins=bins, weights=pg)
bin_centers = (bin_edges_r[:-1] + bin_edges_r[1:]) / 2

# Simple transport plan (approximate, not optimal)
wasserstein_distance_approx = 0
for i in range(bins):
    for j in range(bins):
        distance = np.abs(bin_centers[i] - bin_centers[j])
        # Simplified transport mass as product of bin probabilities
        mass_transported = hist_r[i] * hist_g[j]
        wasserstein_distance_approx += distance * mass_transported

print(f'Approximate Wasserstein Distance: {wasserstein_distance_approx:.4f}')

# Visualize transport plan (partial connections for clarity)
plt.figure(figsize=(12, 6))
plt.bar(bin_centers, hist_r, width=0.2, alpha=0.5, label='Real Distribution (discretized)')
plt.bar(bin_centers, hist_g, width=0.2, alpha=0.5, label='Generated Distribution (discretized)')

# Plot partial connections to avoid clutter
for i in range(0, bins, 2):  # Plot connections for every other bin
    for j in range(0, bins, 2):
        if hist_r[i] > 0 and hist_g[j] > 0:
            # Fix: Scale alpha to be within 0-1 range
            alpha = min(1.0, min(hist_r[i], hist_g[j]) * 10)  # Ensure alpha is between 0 and 1
            plt.plot([bin_centers[i], bin_centers[j]], [hist_r[i]/2, hist_g[j]/2], 
                     'k-', alpha=alpha, linewidth=0.5)

plt.xlabel('x')
plt.ylabel('Probability Density (discretized)')
plt.title('Visualization of Wasserstein Distance Transport Plan')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()    