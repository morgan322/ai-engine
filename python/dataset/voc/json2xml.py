import cv2
import os
import numpy as np
import math
import platform
import json
sys = platform.system()
path = "/media/ai/AI/technology/algorithm/data/1/2"   # json文件路径
img_path = "/media/ai/AI/technology/algorithm/data/1/images"  # 图片文件路径
dst = "/media/ai/AI/technology/algorithm/data/1/xml"           # 保存的xml文件路径

def xml_write(im,image_path,labels,xml_path):
    height=im[0]
    width = im[1]
    if sys == "Windows":
        floder, filename = image_path.rsplit('\\', 1)
    else:
        floder, filename = image_path.rsplit('/', 1)
    xml_file = open(xml_path, 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>' +floder  +'</folder>\n')
    xml_file.write('    <filename>' + filename + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')
    for i in range(len(labels)):
        # for spt in labels[i]:
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + str(labels[i][4]) + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(labels[i][0]) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(labels[i][1]) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(labels[i][2]) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(labels[i][3]) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')
    xml_file.write('</annotation>')

listDir = os.listdir(path)

for cz in listDir:
    if cz.endswith(".json"):
        labels = []
        xml_path = os.path.join(dst, cz[:-5]+'.xml')
        for tail in [".jpg", ".bmp", ".png", "jpeg"]:
            image_path = os.path.join(img_path, cz[:-5] + tail)
            if os.path.exists(image_path):
                # with open(os.path.join(path,cz),encoding='utf-8') as f:
                with open(os.path.join(path,cz), 'r', encoding='latin-1', errors='ignore') as f:
                    jsDict = json.load(f)
                    # jsDict = json.loads(f.read())
                    shape = jsDict['shapes']
                    h = jsDict['imageHeight']
                    w = jsDict['imageWidth']
                    
                    shapes = [h,w,3]
                    for i in range(len(shape)):
                        bboxes = []
                        lab = shape[i]['label']
                        point = shape[i]['points']
                        x_min = point[0][0]
                        x_max = point[1][0]
                        y_min = point[1][0]
                        y_max = point[1][1]
                        if x_min<0:
                            x_min = 0
                        if y_min<0:
                            y_min = 0                                   
                        bboxes.append(x_min)
                        bboxes.append(y_min)
                        bboxes.append(x_max)
                        bboxes.append(y_max)
                        bboxes.append(lab)
                        labels.append(bboxes)
                        print(labels)
                    xml_write(shapes,image_path,labels,xml_path)
                #    print(point)
