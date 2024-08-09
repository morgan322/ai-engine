import cv2
import numpy as np
from scipy.spatial.distance import cdist

# 计算最近距离的函数
def closest_distance(contour1, contour2):
    points1 = contour1[:, 0]
    points2 = contour2[:, 0]
    distances = cdist(points1, points2)
    min_distance = np.min(distances)
    return min_distance

# 并查集数据结构
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    
    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def merge_to_min_rotated_rect(contours, cluster_indices):
    merged_points = []
    
    for indices in cluster_indices:
        all_points = []
        for idx in indices:
            all_points.extend(contours[idx][:, 0])  # 将轮廓点添加到一个列表中
        
        # 将点列表转换为numpy数组
        all_points = np.array(all_points, dtype=np.float32)
        
        # 计算最小外接矩形
        rect = cv2.minAreaRect(all_points)
        
        # 获取最小外接矩形的四个顶点
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # 添加到合并后的点列表中
        merged_points.append(box)
    
    return merged_points

def expand_rect_points(points, expand_pixels):
    expanded_points = np.copy(points)
    center = np.mean(points, axis=0)
    for i in range(4):
        direction = points[i] - center
        direction *= (expand_pixels / np.linalg.norm(direction))
        expanded_points[i] += direction.astype(int)
    return expanded_points

def main(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Find contours in the image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv2.imwrite("contours.jpg", image)

    # # Sort contours by area and filter out contours that are not square-like
    # contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # List to store merged contours
    merged_contours = []
    
    # Process each contour
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the approximate contour has four corners, consider it for merging
        if len(approx) == 4:
            # Merge close contours
            # if merged_contours:
            #     # Calculate distance between centers of current and last merged contours
            #     M1 = cv2.moments(approx)
            #     center1 = (int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]))
            #     M2 = cv2.moments(merged_contours[-1])
            #     center2 = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))
            #     distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    
            #     # Adjust this threshold based on your image scale
            #     if distance < 50:
            #         merged_contours[-1] = np.concatenate((merged_contours[-1], approx))
            #     else:
            #         merged_contours.append(approx)
            # else:
            merged_contours.append(approx)

    # Create a mask with the same size as the original image
    mask = np.zeros_like(gray, dtype=np.uint8)
    square_merged_contours = []
    for contour in merged_contours:
        if abs(max(contour[:, 0, 0])  -  min(contour[:, 0, 0])  -  max(contour[:, 0, 1])  +  min(contour[:, 0, 1])) < 150:
            square_merged_contours.append(contour)

    # 创建并查集实例
    n_contours = len(square_merged_contours)
    uf = UnionFind(n_contours)

    # 构建轮廓之间的距离矩阵
    distances = np.zeros((n_contours, n_contours))
    for i in range(n_contours):
        for j in range(i + 1, n_contours):
            distances[i, j] = closest_distance(square_merged_contours[i], square_merged_contours[j])
            distances[j, i] = distances[i, j]

    # 合并轮廓成团
    for i in range(n_contours):
        for j in range(i + 1, n_contours):
            if distances[i, j] < 15:  # 根据距离阈值合并
                uf.union(i, j)

    # 输出结果
    clusters = {}
    for i in range(n_contours):
        root = uf.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)
    
    # 使用 OpenCV 绘制最小旋转外接矩形
    for root, indices in clusters.items():
        contours_to_merge = [square_merged_contours[idx] for idx in indices]
        # 计算最小外接矩形
        points = np.vstack(contours_to_merge)
        rect = cv2.minAreaRect(points)
        
        # 获取最小外接矩形的四个顶点
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        expanded_box = expand_rect_points(box, 7)

        # Create a blank mask image with the same size as your target image
        mask = np.zeros_like(gray, dtype=np.uint8)
        
        # Draw the expanded box on the mask (fill with white color, 255)
        cv2.fillPoly(mask, [expanded_box], 255)
        result_image = np.zeros_like(image, dtype=np.uint8)
        result_image[mask == 255] = image[mask == 255]
        cv2.imwrite("./mask/" + str(root) + ".jpg", result_image)
        

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    main(image_path)
