# PASCAL Visual Object Classes Challenge
import os
import cv2
import datetime
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image,ImageEnhance

src_dir = '/home/morgan/ubt/data/0722'

frame_flag = 0
plot_flag = 1
gamma_flag = 0
rename_flag = 0
color_flag = 0
EXTRACT_FREQUENCY = 40

image_dir = src_dir + '/JPEGImages/'
annotation_dir = src_dir + '/Annotations/'
VIDEO_PATH = src_dir + '/saveImgs/'
save_dir =src_dir +  "/plot/"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(annotation_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

current_date = datetime.datetime.now()
date_string = current_date.strftime("%Y-%m-%d")

if rename_flag:

    for index, file in enumerate(os.listdir(image_dir)):
        
        old_path = os.path.join(image_dir, file)
        new_name = date_string + '_'+ '{:05d}'.format(index) + '.jpg'
        new_path = os.path.join(src_dir+'/img', new_name)
        
        os.rename(old_path, new_path)


if frame_flag :
    
    count = 1
    cc=len(os.listdir(VIDEO_PATH))
    print("total files==" + str(cc))
    filecount =0
    ev_pic=1
    index = 1
    while True:
        video = cv2.VideoCapture()
        for fn in os.listdir(VIDEO_PATH):
            video_path_and_name = VIDEO_PATH+fn
            print("video_path_and_name="+video_path_and_name)
            if not video.open(video_path_and_name):
                print("can not open the video")
                exit(1)
            while 1:
                _, frame = video.read()
                if frame is None:
                    break
                if count % EXTRACT_FREQUENCY == 0:
                    save_path = image_dir + date_string + '_'+ '{:05d}'.format(index) + '.jpg'
                    # rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    cv2.imwrite(save_path, frame)
                    index += 1
                    ev_pic += 1
                count += 1
            video.release()
            filecount += 1
            print("filecount==" + str(filecount))
            print("extract===>>>"+video_path_and_name)
            print("this file get frame number===>>>"+str(ev_pic))
            ev_pic = 1
            # 打印出所提取帧的总数
            print("Totally save {:d} pics".format(index - 1))
            if filecount == cc:
                break
        break

if plot_flag :
    color_map = {
    'grass': (0, 255, 0),    # 绿色
    'object': (255, 0, 0),   # 红色
    'ground': (0, 0, 255),   # 蓝色
    'cmark': (255, 255, 0),  # 黄色
    'non-grass': (255, 0, 255),  # 洋红色
    '1': (0, 255, 255)    # 青色
    }
    class_count = {}  # 存储每个类别出现的次数
    # 遍历目录中的所有文件
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            # 读取图像
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            # 构建标注文件路径
            annotation_filename = os.path.splitext(filename)[0] + '.xml'
            annotation_path = os.path.join(annotation_dir, annotation_filename)
            # 解析 XML 标注文件
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            # 迭代处理每个对象
            num_index = 0
            for obj in root.findall('object'):
                num_index += 1
                name = obj.find('name').text
                if name == 'non-grass':
                    print(filename)
                if name in class_count:
                    class_count[name] += 1
                else:
                    class_count[name] = 1
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)
                # 在图像上绘制 bounding box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color_map[name], 2)  
                cv2.putText(image, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_map[name], 2)
            # 保存带有 bounding box 和类别名称的图像
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, image)
            if num_index == 0 :
                os.remove(image_path)
                os.remove(annotation_path)
    # 输出每个类别出现的次数
    for class_name, count in class_count.items():
        print(f'{class_name}: {count}')

if gamma_flag:
    for filename in os.listdir(annotation_dir):
        jpg_name = os.path.splitext(filename)[0] + '_' +  str(1) + '.jpg'
        new_name =  os.path.splitext(filename)[0] + '_' +  str(1) + '.xml'
        # 读取 XML 文件
        tree = ET.parse(annotation_dir + filename)
        root = tree.getroot()
        #调节亮度
        if os.path.exists(image_dir +filename.replace('xml','png')):
            exname = 'png'
        else:
            exname = 'jpg'
        image = Image.open(image_dir +filename.replace('xml',exname) )
        arr = np.array(image)
        arr=arr * 1.4
        arr = np.clip(arr, 0, 255)  # 确保像素值在 0 到 255 之间
        processed_image = Image.fromarray(np.uint8(arr))
        processed_image.save(os.path.join(image_dir,jpg_name))
        # 找到 filename 元素并修改文件名
        for filename_elem in root.iter('filename'):
            filename_elem.text = jpg_name
        # 保存修改后的 XML 文件
        tree.write(annotation_dir + new_name )


def adjust_white_balance(image, temp=5000):
    # 将图像转换为 RGB
    img_array = np.array(image.convert("RGB"))

    # 设定色温的加权比例，这里假设色温越高，红色通道更强，蓝色通道更弱
    r, g, b = img_array[..., 0], img_array[..., 1], img_array[..., 2]

    # 色温和 RGB 通道的关系，5000K 作为基准
    scale_r = temp / 5500
    scale_b = 5500 / temp
    
    # 调整 R 和 B 通道
    r = np.clip(r * scale_r, 0, 255)
    b = np.clip(b * scale_b, 0, 255)
    
    # 合并 RGB 通道
    img_array[..., 0] = r
    img_array[..., 2] = b

    # 返回调整后的图像
    return Image.fromarray(img_array.astype(np.uint8))


if color_flag:
    for filename in os.listdir(annotation_dir):
        tree = ET.parse(annotation_dir + filename)
        root = tree.getroot()
        if os.path.exists(image_dir +filename.replace('xml','png')):
            exname = 'png'
        else:
            exname = 'jpg'
    
        image = Image.open(image_dir +filename.replace('xml',exname) )
        contrast = ImageEnhance.Contrast(image)
        image_contrast = contrast.enhance(1.6)  
        image_low = contrast.enhance(0.7)
        saturation = ImageEnhance.Color(image)
        image_saturation = saturation.enhance(1.4)  
        hsv_image = image.convert("HSV")
        hsv_array = np.array(hsv_image)
        hsv_array[..., 0] = (hsv_array[..., 0] + 50) % 256 
        hsv_image = Image.fromarray(hsv_array, "HSV")
        image_hue = hsv_image.convert("RGB") 
        image_white_balance = adjust_white_balance(image, temp=5000)

        new_low_name  = os.path.splitext(filename)[0] + '_' +  str(9) + '.jpg'
        new_contrast_name = os.path.splitext(filename)[0] + '_' +  str(10) + '.jpg'
        new_saturation_name = os.path.splitext(filename)[0] + '_' +  str(11) + '.jpg'
        new_hue_name = os.path.splitext(filename)[0] + '_' +  str(12) + '.jpg'
        new_white_name = os.path.splitext(filename)[0] + '_' +  str(13) + '.jpg'
        new_xml_low_name  = os.path.splitext(filename)[0] + '_' +  str(9) + '.xml'
        new_xml_contrast_name =  os.path.splitext(filename)[0] + '_' +  str(10) + '.xml'
        new_xml_saturation_name =  os.path.splitext(filename)[0] + '_' +  str(11) + '.xml'
        new_xml_hue_name =  os.path.splitext(filename)[0] + '_' +  str(12) + '.xml'
        new_xml_white_name =  os.path.splitext(filename)[0] + '_' +  str(13) + '.xml'

        image_low.save(os.path.join(image_dir,new_low_name))
        image_contrast.save(os.path.join(image_dir,new_contrast_name))
        image_saturation.save(os.path.join(image_dir,new_saturation_name))
        image_hue.save(os.path.join(image_dir,new_hue_name))
        image_white_balance.save(os.path.join(image_dir,new_white_name))
        for filename_elem in root.iter('filename'):
            filename_elem.text = new_low_name
        tree.write(annotation_dir + new_xml_low_name)
        for filename_elem in root.iter('filename'):
            filename_elem.text = new_contrast_name
        tree.write(annotation_dir + new_xml_contrast_name)
        for filename_elem in root.iter('filename'):
            filename_elem.text = new_saturation_name
        tree.write(annotation_dir + new_xml_saturation_name)
        for filename_elem in root.iter('filename'):
            filename_elem.text = new_hue_name
        tree.write(annotation_dir + new_xml_hue_name)
        for filename_elem in root.iter('filename'):
            filename_elem.text = new_white_name
        tree.write(annotation_dir + new_xml_white_name)

        print("图像已调整并保存:",annotation_dir + filename)
