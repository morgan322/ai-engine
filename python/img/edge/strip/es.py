import cv2
import numpy as np

# 读取图像
image = cv2.imread("2.jpg")
if image is None:
    raise FileNotFoundError("图片未找到，请检查路径！")

# 复制原图用于绘制结果
result = image.copy()

# 提前转换为灰度图，避免后面引用时未定义
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 转换为HSV颜色空间，便于检测黑色
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 1. 检测黑色底座（优化：增强形态学 + 轮廓筛选）
# 黑色的HSV范围（V通道放宽到120，覆盖更多暗部）
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 120])  

# 获取黑色区域掩码
black_mask = cv2.inRange(hsv, lower_black, upper_black)

# 形态学操作优化：先闭运算（连接区域）再开运算（去噪），加大核尺寸
kernel = np.ones((7, 7), np.uint8)  # 更大的核，强化噪声过滤
black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # 迭代2次，更好连接底座
black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)

# 2. 找到黑色底座的轮廓（优化：按面积+形状筛选）
contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

base_contour = None
max_area = 0
x, y, w, h = 0, 0, 0, 0  # 初始化底座坐标
for cnt in contours:
    area = cv2.contourArea(cnt)
    # 计算轮廓的外接矩形长宽比（底座应为近似矩形）
    x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
    aspect_ratio = w_cnt / h_cnt if h_cnt != 0 else 0
    # 面积>1000 + 长宽比在0.5~2之间（接近矩形）
    if area > 1000 and 0.5 < aspect_ratio < 2 and area > max_area:  
        max_area = area
        base_contour = cnt
        x, y, w, h = x_cnt, y_cnt, w_cnt, h_cnt  # 更新底座坐标

if base_contour is None:
    raise ValueError("未检测到有效黑色底座")

# 3. 在黑色底座区域内寻找白色反光条（优化：HSV+灰度双阈值，统一尺寸）
# 创建底座ROI掩码
base_mask = np.zeros_like(black_mask)
cv2.drawContours(base_mask, [base_contour], -1, 255, -1)

# 转换为HSV，在ROI内提取白色
hsv_roi = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
white_mask_hsv_roi = cv2.inRange(hsv_roi, np.array([0, 0, 200]), np.array([180, 30, 255]))

# 将HSV掩码扩展到原图尺寸（与gray掩码匹配）
white_mask_hsv = np.zeros_like(gray)
white_mask_hsv[y:y+h, x:x+w] = white_mask_hsv_roi  # 将ROI掩码放到原图对应位置

# 同时用灰度阈值提取（双保险）
gray_roi = cv2.bitwise_and(gray, gray, mask=base_mask)
_, white_mask_gray = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)

# 合并两个掩码（现在尺寸匹配了）
white_mask = cv2.bitwise_or(white_mask_hsv, white_mask_gray)

# 4. 检测反光条轮廓并提取角点（优化：角点排序+筛选）
strip_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

all_corners = []
for cnt in strip_contours:
    area = cv2.contourArea(cnt)
    if area > 50:  # 适当提高面积阈值，过滤小噪声
        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.02 * perimeter  # 降低逼近精度，保留更多细节
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 筛选4角点 + 长宽比接近长条（反光条特征）
        if len(approx) == 4:
            x_strip, y_strip, w_strip, h_strip = cv2.boundingRect(approx)
            strip_aspect = w_strip / h_strip if h_strip != 0 else 0
            # 反光条应为长条，长宽比>2或<0.5
            if strip_aspect > 2 or strip_aspect < 0.5:  
                corners = approx.reshape(-1, 2)
                all_corners.append(corners)
                
                # 绘制结果
                cv2.drawContours(result, [approx], -1, (0, 0, 255), 2)
                for (cx, cy) in corners:
                    cv2.circle(result, (cx, cy), 5, (0, 255, 0), -1)

# 输出与显示
print(f"检测到 {len(all_corners)} 个反光条")
for i, corners in enumerate(all_corners):
    print(f"反光条 {i+1} 角点坐标:")
    for (x_coord, y_coord) in corners:
        print(f"({x_coord}, {y_coord})")

cv2.imshow("Black Base Mask", black_mask)
cv2.imshow("Reflective Strips Detection", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
