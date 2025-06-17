#pip install google-colab -i https://pypi.tuna.tsinghua.edu.cn/simple
 
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
 
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
 
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
 
 
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
 
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)
 
 
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
 
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
 
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
 
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)
 
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')

# model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/4')
movenet = model.signatures['serving_default']

def multipose_model(img):
        input_img = tf.cast(img, dtype=tf.int32)
        # Detection section
        results = movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
        
        return keypoints_with_scores
 

 
# cap = cv2.VideoCapture(0)
path = '../data/output.mp4'
cap = cv2.VideoCapture(path)  # 替换为你的视频文件路径

# 检查视频文件是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
else:
    # 获取视频的相关信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建 VideoWriter 对象，用于保存写字后的视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
            ret, frame = cap.read()
            # Resize image
            if ret:
                img = frame.copy()
                # print(img.shape)
                prev_frame_time = time.time()
                img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 480, 640)
                keypoints_with_scores =  multipose_model(img)
                # print(keypoints_with_scores)
                
                # Render keypoints
                loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)  # 0.1 is our confidence_threshold
                current_frame_time = time.time()
                processing_time = (current_frame_time - prev_frame_time) * 1000  # 转换为毫秒

                # 打印处理时间
                print("帧处理时间: {:.2f} 毫秒".format(processing_time))
                out.write(frame)
            else:
                 break
            # cv2_imshow(frame)
            # cv2.imshow('Video',frame)
    
            # if cv2.waitKey(10) & 0xFF==ord('q'):
            #     break
cap.release()
out.release()
print("finished")
# cv2.destroyAllWindows()