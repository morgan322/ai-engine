
import tensorflow as tf
# import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


model_name = "/media/nvidia/D6612D9737620A9A/deploy/rk3568/model/tflite/aeke_heavyweight.tflite"


interpreter = tf.lite.Interpreter(model_path=model_name)
interpreter.allocate_tensors()

def movenet(input_image):
    """Runs detection on an input image.

    Args:
        input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.

    interpreter.invoke()

    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

def draw_connections(frame, keypoints, edges, confidence_threshold):
    # print('frame', frame)
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
 
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
 
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    # print("shaped in draw_keypoints:", shaped)
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)


def resize_with_pad(image, target_size):
    height, width = image.shape[:2]

    # 计算缩放比例
    ratio = max(target_size[0] / float(width), target_size[1] / float(height))
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # 缩放图像
    resized = cv2.resize(image, (new_width, new_height))

    # 计算填充量
    pad_width = max(target_size[0] - new_width, 0)
    pad_height = max(target_size[1] - new_height, 0)

    # 填充图像
    padded = cv2.copyMakeBorder(resized, 
                                top=0, bottom=pad_height, 
                                left=0, right=pad_width, 
                                borderType=cv2.BORDER_CONSTANT, 
                                value=0)
    return padded


# Load the input image.
image_path = './20231205-143209.jpg'
video_path = '../data/ose.mp4'
output_video = '../data/output_h.mp4'
image = tf.io.read_file(image_path)
image = tf.compat.v1.image.decode_jpeg(image)


# Resize and pad the image to keep the aspect ratio and fit the expected size.
input_size = 256
input_image = tf.expand_dims(image, axis=0)
input_image = tf.image.resize_with_pad(input_image, input_size, input_size)


# Run model inference.
keypoints_with_scores = movenet(input_image)
display_image = tf.cast(tf.image.resize_with_pad(image, 1280, 1280), dtype = tf.int32)
display_image = np.array(display_image)
origin_image = np.copy(display_image)

loop_through_people(display_image, keypoints_with_scores, EDGES, 0.1)


# plt.subplot(1, 2, 1)
# plt.imshow(origin_image)
# plt.subplot(1, 2, 2)
# plt.imshow(display_image)
# plt.show()



# 打开视频文件
video_capture = cv2.VideoCapture(video_path)

# 获取视频的宽度和高度
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)
# 创建视频编写器对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while True:
    # 读取一帧视频
    ret, frame = video_capture.read()
    # 如果视频读取完毕，则退出循环
    if not ret:
        break
    input_image = tf.expand_dims(frame, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

    # Run model inference.
    prev_frame_time = time.time()
    keypoints_with_scores = movenet(input_image)
    current_frame_time = time.time()
    processing_time = (current_frame_time - prev_frame_time) * 1000  # 转换为毫秒

    # 打印处理时间
    print("帧处理时间: {:.2f} 毫秒".format(processing_time))

    # display_image = tf.cast(tf.image.resize_with_pad(frame, 1280, 1280), dtype = tf.int32)
    # display_image = np.array(display_image)
    # origin_image = np.copy(display_image)
    resized_frame = resize_with_pad(frame, (1280, 1280))

    
    loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
    
    

    # 写入处理后的帧到输出视频
    output_video.write(frame)

# 释放资源
video_capture.release()
output_video.release()




