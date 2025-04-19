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
    