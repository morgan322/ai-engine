import cv2
import numpy as np
import os

def process_gray_image(image_path):
    """
    处理单张灰度图：边缘检测 + 反光条识别（增加灰度图白色判断）
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    if image is None:
        print(f"图片 {image_path} 未找到或不是有效灰度图！")
        return None
    

    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  
    

    blur = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(result, contours, -1, (255, 255, 5), 5)
    for contour in contours:
        area = cv2.contourArea(contour)

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
        x_max = min(image.shape[1] - 1, x_max)
        y_max = min(image.shape[0] - 1, y_max)
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio < 2 or area < 100:
            continue
        
        roi_gray = image[y_min:y_max, x_min:x_max]
        mean_brightness = np.mean(roi_gray)
        if len(approx) == 4 or mean_brightness > 80:  
           
            cv2.polylines(result, [box], isClosed=True, color=(0, 255, 0), thickness=2)
    
    return result

def traverse_gray_folder(input_folder, output_folder):
    """
    遍历文件夹处理灰度图，保存结果
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  
            image_path = os.path.join(input_folder, filename)
            processed_img = process_gray_image(image_path)
            
            if processed_img is not None:
                save_path = os.path.join(output_folder, filename)
                cv2.imwrite(save_path, processed_img)
                print(f"已处理并保存：{save_path}")

if __name__ == "__main__":
    # 输入文件夹（存放灰度图）
    input_folder = r"D:\dmt\data\strip"  
    # 输出文件夹（保存带检测结果的彩色图）
    output_folder = r"D:\dmt\data\so"  
    traverse_gray_folder(input_folder, output_folder)