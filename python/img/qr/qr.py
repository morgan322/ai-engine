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

def compute_iou(box1, box2):
    # 计算矩形的凸包
    poly1 = cv2.convexHull(box1)
    poly2 = cv2.convexHull(box2)
    
    # 获取交集
    intersection = cv2.intersectConvexConvex(poly1, poly2)
    
    if intersection[0] == 0:
        # 没有交集，IoU为0
        return 0.0
    
    # 交集区域的面积
    intersection_area = cv2.contourArea(intersection[1])
    
    # 计算两个矩形的面积
    area1 = cv2.contourArea(poly1)
    area2 = cv2.contourArea(poly2)
    
    # 计算并集的面积（并集面积 = area1 + area2 - intersection_area）
    union_area = area1 + area2 - intersection_area
    
    # 计算IoU
    iou = intersection_area / union_area
    return iou

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


def filter_boxes(boxs):
    filtered_boxes = []
    for box in boxs:
        to_add = True  
        for box0 in filtered_boxes:
            if compute_iou(box0, box) > 0.1:  
                to_add = False
        if to_add:
            filtered_boxes.append(box)
    return filtered_boxes

def main(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detection
    edges = cv2.Canny(gray, 150, 200)

    # Find contours in the image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 合并所有轮廓的凸包
    contours = [cv2.convexHull(contour) for contour in contours]
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    # cv2.imwrite("contours.jpg", image)

    merged_contours = []
    
    # Process each contour
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            merged_contours.append(approx)
    cv2.drawContours(image, merged_contours, -1, (0, 255, 0), 3)
    cv2.imwrite("mcontours.jpg", image)
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

    
    distances_flat = distances[np.triu_indices(n_contours, k=1)]
    if len(distances_flat) > 24:
        top_10_distances = np.sort(distances_flat)[:24]
        average_min_10_distance = np.mean(top_10_distances)
    min_distance = np.min(distances_flat)
    threshold = average_min_10_distance * 2.5

    # 合并轮廓成团
    for i in range(n_contours):
        for j in range(i + 1, n_contours):
            if distances[i, j] < threshold:  # 根据距离阈值合并
                uf.union(i, j)

    # 输出结果
    clusters = {}
    for i in range(n_contours):
        root = uf.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)
        
    boxs = []
    for root, indices in clusters.items():
        contours_to_merge = [square_merged_contours[idx] for idx in indices]
        points = np.vstack(contours_to_merge)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)  
        areas = cv2.contourArea(box)
        if areas < 80 * 80 * 3 and len(indices) < 3:
            continue
        flag = 0
        for idx in indices:
            point0 = square_merged_contours[idx]
            rect0 = cv2.minAreaRect(point0)
            box0 = cv2.boxPoints(rect0)
            box0 = np.int0(box0)
            area0 = cv2.contourArea(box0)
            if area0 > 1 * 10000 :
                boxs.append(box0)
                flag += 1 
                print(area0, flag)
        if flag == 0 :
            boxs.append(box)
    
    id = 0
    boxs = filter_boxes(boxs)
    print("filter_boxes", len(boxs))
    for box in boxs:
        expanded_box = expand_rect_points(box, 7)
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillPoly(mask, [expanded_box], 255)
        result_image = np.zeros_like(image, dtype=np.uint8)
        result_image[mask == 255] = image[mask == 255]
        id += 1
        cv2.imwrite("./mask/" + str(id) + ".jpg", result_image)
        
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <image_path> <type>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    type = sys.argv[2]

    if type == "0":
        import os
        for dirpath, dirnames, filenames in os.walk(image_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                print(file_path)
                main(file_path)
    elif type == "1":
        main(image_path)