import numpy as np
import math


def haar_feature(image):
    height, width = image.shape
    # 简单示例，计算水平方向的 Haar 特征
    left_sum = np.sum(image[:, :width // 2], dtype=np.int32)
    right_sum = np.sum(image[:, width // 2:], dtype=np.int32)
    return right_sum - left_sum


def gabor_kernel(sigma, theta, lambd, gamma, psi):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    (y, x) = np.meshgrid(np.arange(-sigma_y * 3, sigma_y * 3 + 1), np.arange(-sigma_x * 3, sigma_x * 3 + 1))

    # 计算 Gabor 核
    gb = np.exp(-(x ** 2 / (2. * sigma_x ** 2) + y ** 2 / (2. * sigma_y ** 2))) * \
         np.cos(2 * math.pi * x / lambd + psi)
    return gb


def gabor_feature(image):
    # 简单示例，使用一个 Gabor 核提取特征
    sigma = 10
    theta = 0
    lambd = 10
    gamma = 0.5
    psi = 0
    kernel = gabor_kernel(sigma, theta, lambd, gamma, psi)
    # 裁剪卷积核使其和图像大小一致
    kernel = kernel[:image.shape[0], :image.shape[1]]
    # 确保裁剪后的 kernel 与 image 形状完全一致
    if kernel.shape != image.shape:
        new_kernel = np.zeros_like(image)
        min_height = min(kernel.shape[0], image.shape[0])
        min_width = min(kernel.shape[1], image.shape[1])
        new_kernel[:min_height, :min_width] = kernel[:min_height, :min_width]
        kernel = new_kernel
    # 简单卷积操作
    feature = np.sum(image * kernel)
    return feature


def lbp_feature(image):
    height, width = image.shape
    lbp_image = np.zeros((height - 2, width - 2), dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            center = image[y, x]
            code = 0
            code |= (image[y - 1, x - 1] >= center) << 7
            code |= (image[y - 1, x] >= center) << 6
            code |= (image[y - 1, x + 1] >= center) << 2
            code |= (image[y, x - 1] >= center) << 1
            code |= (image[y, x + 1] >= center)
            code |= (image[y + 1, x - 1] >= center) << 5
            code |= (image[y + 1, x] >= center) << 4
            code |= (image[y + 1, x + 1] >= center) << 3
            lbp_image[y - 1, x - 1] = code
    # 简单示例，计算 LBP 特征向量（这里简单求和）
    feature = np.sum(lbp_image)
    return feature


def sift_keypoint_detection(image):
    # 简单示例，这里只是简单的极值点检测
    height, width = image.shape
    keypoints = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            current = image[y, x]
            if current > np.max(image[y - 1:y + 2, x - 1:x + 2]) and current > np.min(image[y - 1:y + 2, x - 1:x + 2]):
                keypoints.append((x, y))
    return keypoints


def hog_feature(image):
    height, width = image.shape
    # 计算梯度
    gx = np.zeros((height, width))
    gy = np.zeros((height, width))
    gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
    gy[1:-1, :] = image[2:, :] - image[:-2, :]
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx)

    # 简单示例，划分单元格并统计直方图
    cell_size = 8
    num_cells_y = height // cell_size
    num_cells_x = width // cell_size
    hog_vector = []
    for cy in range(num_cells_y):
        for cx in range(num_cells_x):
            cell_magnitude = magnitude[cy * cell_size:(cy + 1) * cell_size, cx * cell_size:(cx + 1) * cell_size]
            cell_orientation = orientation[cy * cell_size:(cy + 1) * cell_size, cx * cell_size:(cx + 1) * cell_size]
            hist = np.zeros(9)
            for y in range(cell_size):
                for x in range(cell_size):
                    bin = int(cell_orientation[y, x] * 9 / (2 * np.pi))
                    hist[bin] += cell_magnitude[y, x]
            hog_vector.extend(hist)
    return np.array(hog_vector)


def surf_hessian(image):
    height, width = image.shape
    # 计算二阶导数
    dx = np.zeros((height, width))
    dy = np.zeros((height, width))
    dxx = np.zeros((height, width))
    dyy = np.zeros((height, width))
    dxy = np.zeros((height, width))

    dx[:, 1:-1] = image[:, 2:] - image[:, :-2]
    dy[1:-1, :] = image[2:, :] - image[:-2, :]
    dxx[:, 1:-1] = dx[:, 2:] - dx[:, :-2]
    dyy[1:-1, :] = dy[2:, :] - dy[:-2, :]
    dxy[1:-1, 1:-1] = (image[2:, 2:] - image[2:, :-2] - image[:-2, 2:] + image[:-2, :-2]) / 4

    # 计算 Hessian 矩阵的行列式和迹
    det_hessian = dxx * dyy - dxy ** 2
    trace_hessian = dxx + dyy
    return det_hessian, trace_hessian


if __name__ == "__main__":
    # 生成一个简单的示例图像
    sample_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    # 提取各种特征
    haar = haar_feature(sample_image)
    gabor = gabor_feature(sample_image)
    lbp = lbp_feature(sample_image)
    sift_keypoints = sift_keypoint_detection(sample_image)
    hog = hog_feature(sample_image)
    det_hessian, trace_hessian = surf_hessian(sample_image)

    print(f"Haar 特征值: {haar}")
    print(f"Gabor 特征值: {gabor}")
    print(f"LBP 特征值: {lbp}")
    print(f"SIFT 关键点数量: {len(sift_keypoints)}")
    print(f"HOG 特征向量长度: {len(hog)}")
    print(f"SURF Hessian 行列式形状: {det_hessian.shape}")
    print(f"SURF Hessian 迹形状: {trace_hessian.shape}")
    