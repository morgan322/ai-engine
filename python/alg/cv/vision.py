import MNN.nn as nn
import MNN.expr as expr
import numpy as np
import cv2
import argparse
import os
from datetime import datetime
import time

class ObjectDetector:
    def __init__(self, model_path,lite_model_path,prototxt,weights,thr = 0.2):
        self.input_name = ['data']
        self.output_name = ['detection_out']
        self.lite_input_name = ['normalized_input_image_tensor']
        self.lite_output_name = ['TFLite_Detection_PostProcess', 'TFLite_Detection_PostProcess:2', 'TFLite_Detection_PostProcess:3','TFLite_Detection_PostProcess:1']
        self.thr = thr
        self.mean = (127.5, 127.5, 127.5)
        self.nomal = (1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5)
        self.size = (300, 300)
        self.class_name =  { 0: 'background', 1: 'grass', 2: 'non'}
        self.rt = nn.create_runtime_manager(({'precision': 'low','power': 'high', 'memory': 'high','backend': "CUDA", 'numThread': 68},))
        self.net = nn.load_module_from_file(model_path, self.input_name, self.output_name, runtime_manager=self.rt)
        self.lite_net = nn.load_module_from_file(lite_model_path, self.lite_input_name, self.lite_output_name,runtime_manager=self.rt)
        self.caffe_net = cv2.dnn.readNetFromCaffe(prototxt, weights)
    
    def infer_lite(self, image):
        clone_image = image.copy()
        frame_resized = cv2.resize(clone_image,self.size)
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, self.size, self.mean, False)
        self.caffe_net.setInput(blob)
        detections = self.caffe_net.forward()
        box_img = self.darw_boxes_caffe(image, detections)
        return box_img
    def darw_boxes_caffe(self, image, detections):
        clone_image = image.copy()
        cols = image.shape[1] 
        rows = image.shape[0]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2] 
            if confidence > args.thr: 
                class_id = int(detections[0, 0, i, 1]) 
                xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)
                center_x = xLeftBottom + 5
                center_y = (yLeftBottom + yRightTop) / 2
                cv2.rectangle(clone_image, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (255, 0, 0), 2)
                cv2.putText(clone_image, f'{self.class_name[class_id]}: {confidence:.2f}', (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.putText(clone_image, "CAFFE", (int(cols/2), int(100)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
        return clone_image
    
    def infer_image(self, image):
        clone_image = image.copy()
        clone_image = cv2.cvtColor(clone_image, cv2.COLOR_BGR2RGB)
        clone_image = clone_image[..., ::-1]
        clone_image = cv2.resize(clone_image, self.size)
        clone_image = clone_image - self.mean
        clone_image = clone_image * self.nomal
        clone_image = clone_image.astype(np.float32)
        
        input_var = expr.placeholder([1, 300, 300, 3], expr.NHWC)
        input_var.write(clone_image)
        input_var = expr.convert(input_var, expr.NC4HW4)
        output_var = self.net.forward([input_var])
        output_var = output_var[0]
        output_var = expr.convert(output_var, expr.NHWC)
        box_img = self.darw_boxes(image, output_var)
        
        return box_img
    def infer_lite(self, image):
        
        clone_image = image.copy()
        clone_image = cv2.cvtColor(clone_image, cv2.COLOR_BGR2RGB)
        clone_image = cv2.resize(clone_image, self.size)
        clone_image = (clone_image - 127.5 ) /127.5
        clone_image = clone_image.astype(np.float32)
        
        input_var = expr.placeholder([1, 300, 300, 3], expr.NHWC)
        input_var.write(clone_image)
        input_var = expr.convert(input_var, expr.NC4HW4)
        output_var = self.lite_net.forward([input_var])
        raw_image_height = image.shape[0]
        raw_image_width = image.shape[1]
        detection_boxes = expr.convert(output_var[0], expr.NHWC)
        detection_scores = expr.convert(output_var[1], expr.NHWC)
        num_boxes = expr.convert(output_var[2], expr.NHWC)
        detection_classes= expr.convert(output_var[3], expr.NHWC)
        # result = []
        for i in range(int(num_boxes[0])):
            confidence = detection_scores[0][i]
            if(detection_scores[0][i]>0.55):
                y1 =int(detection_boxes[0][0][i] * raw_image_height)
                x1 = int(detection_boxes[0][1][i]* raw_image_width)
                y2 = int(detection_boxes[0][2][i]* raw_image_height)
                x2 = int(detection_boxes[0][3][i]* raw_image_width)
                class_id = int(detection_classes[0][i])
                # if class_id == 1:
                #     result.append([x1, y1, x2 - x1, y2 - y1])
        # with open('anchors.txt', 'w') as f:
        #     f.write("float anchors[{}][4] = {{".format(anchors_var.shape[0]))
        #     for i in range(anchors_var.shape[0]):
        #         f.write(" {")
        #         for j in range(anchors_var.shape[1]):
        #             f.write("{:.8f}".format(anchors_var[i][j]))
        #             if j < anchors_var.shape[1] - 1:
        #                 f.write(", ")
        #         f.write("}")
        #         if i < anchors_var.shape[0] - 1:
        #             f.write(",")
        #         f.write("\n")
        #     f.write("};")
        box_img = self.darw_lite(image, output_var)
        return box_img
    def darw_lite(self, image, output_var):
        clone_image = image.copy()
        raw_image_height = image.shape[0]
        raw_image_width = image.shape[1]
        detection_boxes = expr.convert(output_var[0], expr.NHWC)
        detection_scores = expr.convert(output_var[1], expr.NHWC)
        num_boxes = expr.convert(output_var[2], expr.NHWC)
        detection_classes= expr.convert(output_var[3], expr.NHWC)
        for i in range(int(num_boxes[0])):
            confidence = detection_scores[0][i]
            if(detection_scores[0][i]>0.3):
                y1 =int(detection_boxes[0][0][i] * raw_image_height)
                x1 = int(detection_boxes[0][1][i]* raw_image_width)
                y2 = int(detection_boxes[0][2][i]* raw_image_height)
                x2 = int(detection_boxes[0][3][i]* raw_image_width)
                center_x = x1 + 5
                center_y = (y1 + y2) / 2
                class_id = int(detection_classes[0][i])
                cv2.rectangle(clone_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(clone_image, f'{self.class_name[class_id+1]}: {confidence:.2f}', (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.putText(clone_image, "LITE", (int(raw_image_width/2), int(100)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
        return clone_image
    def darw_boxes(self, image, output_var):
        clone_image =  image.copy()
        cols = image.shape[1] 
        rows = image.shape[0]
        for i in range(output_var.shape[1]):
            confidence = output_var[0, i, 1, 0]
            if confidence > self.thr:
                class_id = int(output_var[0, i, 0, 0])  
                xLeftBottom = int(output_var[0, i, 2, 0]* cols)
                yLeftBottom = int(output_var[0, i, 3, 0]* rows)
                xRightTop = int(output_var[0, i, 4, 0] * cols)
                yRightTop = int(output_var[0, i, 5, 0] *  rows)
                center_x = xLeftBottom + 5
                center_y = (yLeftBottom + yRightTop) / 2
                cv2.rectangle(clone_image, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (255, 0, 0), 2)
                cv2.putText(clone_image, f'{self.class_name[class_id]}: {confidence:.2f}', (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.putText(clone_image, "MNN", (int(cols/2), int(100)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
        return clone_image
        
    def infer_video(self, video_path, save_path='output_video.mp4'):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width , frame_height))
        cap_flap = 0
        if not (video_path.endswith('.mp4') or video_path.endswith('.avi') or video_path.endswith('.mov')):
            filename = os.path.basename(save_path)
            filedir = os.path.dirname(save_path)
            new_filename = os.path.join(filedir, f"0_{filename}")
            src_out = cv2.VideoWriter(new_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            cap_flap = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # img_mnn = self.infer_image(frame)
            img_lite = self.infer_lite(frame)
            # combined_image = np.vstack((img_lite,img_mnn))
            out.write(img_lite)
            if(cap_flap == 1):
                src_out.write(frame)

            # screen_width = 600
            # scale_factor = screen_width / combined_image.shape[1]  
            # height = int(combined_image.shape[0] * scale_factor)  
            # resized_image = cv2.resize(combined_image, (screen_width, height))  
            # cv2.imshow('output', img_lite)   
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        cap.release()
        out.release()
        if (cap_flap == 1):
            src_out.release()
        cv2.destroyAllWindows()

parser = argparse.ArgumentParser(description='Script to run MobileNet-SSD object detection network of MNN ')
parser.add_argument("--video", help="path to video file. ")
parser.add_argument("--img",help='source img path ' )
parser.add_argument("--img_dir",help='source img path ' )
parser.add_argument("--save_dir", default="./test/save_mnn/",help='save img path ')
parser.add_argument("--model", default="/home/morgan/ubt/alg/deploy/MNN/build-x86/model/mobilenetssd_fp16.mnn",help='Path to model')
parser.add_argument("--lite_model", default="/home/morgan/ubt/program/project/ai-4-qcm2290-no-snpe/file-copy/data/ai/grass/tflite_mssdint8_grass.mnn",help='Path to model')
parser.add_argument("--prototxt", default="/home/morgan/ubt/alg/cv/export/MobileNet-SSD/example/MobileNetSSD_deploy.prototxt",help='Path to prototxt')
parser.add_argument("--weights", default="/home/morgan/ubt/alg/cv/export/MobileNet-SSD/snapshot_5/mobilenet_iter_110000.caffemodel",help='Path to weights')
parser.add_argument("--thr", default=0.35, type=float, help="confidence threshold to filter out weak detections")

def main(args):
    now = datetime.now()
    date_part = str(now.strftime("%Y-%m-%d"))
    src_path = args.save_dir + date_part
    os.makedirs(src_path, exist_ok=True)
    time_part = str(now.strftime("%H-%M-%S"))
    detector = ObjectDetector(args.model,args.lite_model,args.prototxt , args.weights,thr=args.thr)
    if args.img:
        img = cv2.imread(args.img)
        img_mnn = detector.infer_image(img)
        img_lite = detector.infer_lite(img)
        save_path = os.path.join(src_path,time_part + ".jpg")
        print(save_path)
        combined_image = np.hstack((img_lite,img_mnn))
        cv2.imwrite(save_path, combined_image)
    elif args.img_dir:
        dir_path = src_path + "/dir"
        os.makedirs(dir_path, exist_ok=True)
        for img_name in os.listdir(args.img_dir):
            img_path = os.path.join(args.img_dir, img_name)
            cv2_img = cv2.imread(img_path)
            start = time.time()
            img_mnn = detector.infer_image(cv2_img)
            img_lite = detector.infer_lite(cv2_img)
            end = time.time()
            print("mnn time: ", (end - start)*1000,"ms")
            save_path = os.path.join(dir_path,img_name)
            combined_image = np.hstack((img_lite,img_mnn))
            cv2.imwrite(save_path, combined_image)
    elif args.video:
        save_path = os.path.join(src_path, time_part + ".mp4")
        detector.infer_video(args.video, save_path)

if __name__ == "__main__":

    args = parser.parse_args()
    main(args)
