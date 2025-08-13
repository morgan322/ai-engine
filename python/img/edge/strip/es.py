import cv2
import numpy as np

def merge_contours(edges, morph_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

image = cv2.imread(r"D:\dmt\data/20250812-170338.jpg")
if image is None:
    raise FileNotFoundError("图片未找到，请检查路径！")

result = image.copy()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])
white_mask = cv2.inRange(hsv, lower_white, upper_white)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, 100, 200)
# contours = merge_contours(edges)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(result, contours, -1, (0, 255, 5), 5)
for contour in contours:
    area = cv2.contourArea(contour)
    if area < 100:
        continue

    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    rotated_rect = cv2.minAreaRect(contour)
    (cx, cy), (width, height), angle = rotated_rect
    box = cv2.boxPoints(rotated_rect)
    box = np.int32(box)
    x_min, y_min = np.min(box, axis=0)
    x_max, y_max = np.max(box, axis=0)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1]-1, x_max)
    y_max = min(image.shape[0]-1, y_max)
    roi_white = white_mask[y_min:y_max, x_min:x_max]
    white_area = cv2.countNonZero(roi_white)
    total_area = (x_max - x_min) * (y_max - y_min)
    aspect_ratio = max(width, height) / min(width, height)
    if total_area == 0 or aspect_ratio < 2:
        continue
    ratio = white_area / total_area
    
    if ratio > 0.3 or len(approx) == 4:
        # print(ratio,area,aspect_ratio)
        cv2.polylines(result, [box], isClosed=True, color=(0, 255, 0), thickness=2)

cv2.imwrite('rotated_contour_results.jpg', result)

