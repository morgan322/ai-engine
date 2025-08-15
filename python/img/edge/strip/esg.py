import cv2
import numpy as np
import os

def get_subpixel_corners_near_vertices(gray_img, box, search_radius=10):
    """在矩形框四个顶点附近搜索并提取亚像素级角点"""
    subpix_corners = []
    # 遍历矩形框的四个顶点
    for (x, y) in box:
        # 1. 确定顶点附近的搜索区域（以顶点为中心，search_radius为半径的正方形）
        x_start = max(0, x - search_radius)
        x_end = min(gray_img.shape[1], x + search_radius + 1)
        y_start = max(0, y - search_radius)
        y_end = min(gray_img.shape[0], y + search_radius + 1)
        
        # 2. 提取搜索区域的ROI
        roi = gray_img[y_start:y_end, x_start:x_end]
        if roi.size == 0:
            continue
        
        # 3. 在ROI内检测初始角点（使用Shi-Tomasi算法，对边缘敏感）
        corners = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=5,  # 每个顶点附近最多找5个候选角点
            qualityLevel=0.01,
            minDistance=3,
            blockSize=3
        )
        if corners is None:
            continue
        
        # 4. 转换ROI内的角点坐标到原图坐标
        corners = np.int32(corners).reshape(-1, 2)
        corners = [(cx + x_start, cy + y_start) for (cx, cy) in corners]
        
        # 5. 找到距离原始顶点最近的角点（确保在顶点附近）
        corners_with_dist = [((cx - x)**2 + (cy - y)** 2, cx, cy) for (cx, cy) in corners]
        corners_with_dist.sort()  # 按距离排序
        nearest_corner = (corners_with_dist[0][1], corners_with_dist[0][2])
        
        # 6. 亚像素级细化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        subpix = cv2.cornerSubPix(
            gray_img,
            np.float32([nearest_corner]),
            (5, 5),
            (-1, -1),
            criteria
        )
        subpix_corners.append(subpix[0])
    
    return np.array(subpix_corners, dtype=np.float32)

def process_gray_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"图片 {image_path} 未找到或不是有效灰度图！")
        return None, None

    # 生成用于绘制的彩色图
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 原始矩形框检测逻辑（完全保留）
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    valid_boxes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        rotated_rect = cv2.minAreaRect(contour)
        (_, (width, height), _) = rotated_rect
        if min(width, height) < 20:
            continue

        box = cv2.boxPoints(rotated_rect)
        box = np.int32(box)
        x_min, y_min = np.min(box, axis=0)
        x_max, y_max = np.max(box, axis=0)
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1] - 1, x_max)
        y_max = min(image.shape[0] - 1, y_max)

        aspect_ratio = max(width, height) / min(width, height)
        roi_gray = image[y_min:y_max, x_min:x_max]
        mean_brightness = np.mean(roi_gray)

        if len(cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)) == 4 or mean_brightness > 80:
            valid_boxes.append((aspect_ratio, box))

    if len(valid_boxes) == 2:
        for _, box in valid_boxes:
            boxes.append(box)
    else:
        valid_boxes.sort(key=lambda x: x[0], reverse=True)
        top_one_boxes = [box for _, box in valid_boxes[:1]]
        extra_boxes = [box for _, box in valid_boxes[1:]]

        for box in top_one_boxes:
            boxes.append(box)

        if extra_boxes:
            all_points = np.concatenate(extra_boxes, axis=0)
            merged_rect = cv2.minAreaRect(all_points)
            merged_box = cv2.boxPoints(merged_rect)
            merged_box = np.int32(merged_box)
            boxes.append(merged_box)
    
    return boxes, result, image

def traverse_gray_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            save_path = os.path.join(output_folder, filename)
            
            # 获取检测到的矩形框和图像
            boxes, result, gray_image = process_gray_image(image_path)
            if boxes is None or result is None:
                continue
            
            # 处理每个矩形框，在四个顶点附近找亚像素角点
            for box in boxes:
                # 绘制原始矩形框（可选，用于对比）
                # cv2.polylines(result, [box], True, (0, 255, 0), 2)
                
                # 在顶点附近搜索并提取亚像素角点
                subpix_corners = get_subpixel_corners_near_vertices(gray_image, box)
                
                # 绘制亚像素角点（红色实心圆）
                for (x, y) in subpix_corners:
                    cv2.circle(result, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)
                corners_int = [(int(round(x)), int(round(y))) for (x, y) in subpix_corners]
                for i in range (4):
                    start = corners_int [i]
                    end = corners_int [(i + 1) % 4] # 第 4 个点连接回第 1 个点
                    cv2.line (result, start, end, (0, 255, 0), 1) # 绿色线条
            # 保存结果
            cv2.imwrite(save_path, result)
            print(f"已处理并保存：{save_path}")

if __name__ == "__main__":
    input_folder = r"D:\dmt\data\strip"
    output_folder = r"D:\dmt\data\so"
    traverse_gray_folder(input_folder, output_folder)
    