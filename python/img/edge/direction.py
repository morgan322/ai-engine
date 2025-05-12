import cv2
import numpy as np
import sys

def merge_contours(edges, morph_size=3):
    """合并Canny边缘检测后的相邻轮廓"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def main(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 转换为灰度图并检测边缘
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 100, 150)
    
    # 合并相连的轮廓
    merged_contours = merge_contours(edges)
    
    min_rects = []
    for contour in merged_contours:
        min_rect = cv2.minAreaRect(contour)
        (cx, cy), (width, height), angle = min_rect  # 解包矩形参数
        
        area = width * height
        print(f"轮廓面积: {area:.2f}, 角度: {angle:.2f}°")
        
        if area < 500:
            continue
        
        aspect_ratio = width / height if height > 0 else float('inf')
        square_threshold = 0.8
        is_square = (1/square_threshold >= aspect_ratio >= square_threshold)
        
        if is_square:
            print("找到正方形")
        else:
            min_rects.append(min_rect)
        
        # 绘制最小旋转矩形
        box = cv2.boxPoints(min_rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
        
        # 绘制矩形方向线（从中心点沿长轴方向）
        major_axis_length = max(width, height) / 2
        angle_rad = angle * np.pi / 180  # 转换为弧度
        
        # 计算方向线终点
        if width > height:
            end_x = int(cx + major_axis_length * np.cos(angle_rad))
            end_y = int(cy + major_axis_length * np.sin(angle_rad))
        else:
            end_x = int(cx + major_axis_length * np.cos(angle_rad + np.pi/2))
            end_y = int(cy + major_axis_length * np.sin(angle_rad + np.pi/2))
        
        # 绘制方向线（绿色）
        cv2.line(image, (int(cx), int(cy)), (end_x, end_y), (0, 255, 0), 2)
        
        # 显示角度和面积信息
        label = f"{angle:.1f}° {'(square)' if is_square else ''}"
        cv2.putText(image, label, 
                    (int(cx) + 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (0, 255, 0) if is_square else (255, 255, 255), 1)

    # 保存结果
    cv2.imwrite("merged_contours_with_min_rects.jpg", image)

if __name__ == "__main__":
    main(sys.argv[1])