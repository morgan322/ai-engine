import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import random
import os

# 假设 config 中定义了必要参数
import config


class SelectiveSearch:
    """Selective Search 候选区域生成（简化实现）"""
    def __init__(self, scale=180, sigma=0.8, min_size=50):
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size
    
    def generate_proposals(self, img):
        """生成候选区域（简化版本，实际项目中可使用skimage等库的实现）"""
        # 实际项目中可调用skimage的selective_search函数
        # 这里为演示，随机生成一些候选框
        h, w = img.shape[:2]
        proposals = []
        
        # 简化实现：随机生成2000个候选框
        for _ in range(2000):
            # 确保候选框尺寸合理
            min_side = min(h, w) // 20
            max_side = min(h, w) // 2
            
            w_size = random.randint(min_side, max_side)
            h_size = random.randint(min_side, max_side)
            
            x = random.randint(0, w - w_size)
            y = random.randint(0, h - h_size)
            
            proposals.append([x, y, x + w_size, y + h_size])
        
        return np.array(proposals)


class RCNAlexNet(nn.Module):
    """基于AlexNet的特征提取网络"""
    def __init__(self):
        super(RCNAlexNet, self).__init__()
        # 加载预训练的AlexNet
        alexnet = models.alexnet(pretrained=True)
        # 保留特征提取部分（去掉最后的全连接层）
        self.features = alexnet.features
        # 自适应平均池化将特征图转为固定尺寸
        self.avgpool = alexnet.avgpool
        # 输出特征维度为4096，与AlexNet的fc1层一致
        self.feature_dim = 4096
    
    def forward(self, x):
        # 输入: [N, 3, 227, 227]
        x = self.features(x)
        x = self.avgpool(x)
        # 展平特征
        x = torch.flatten(x, 1)
        return x


class RCNNSVM(nn.Module):
    """R-CNN的SVM分类器（PyTorch中用线性层实现）"""
    def __init__(self, feature_dim, num_classes):
        super(RCNNSVM, self).__init__()
        # 每个类别对应一个SVM分类器
        self.classifiers = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        # x: [num_proposals, feature_dim]
        # 输出: [num_proposals, num_classes]
        return self.classifiers(x)


class BBoxRegression(nn.Module):
    """边界框回归模型"""
    def __init__(self, feature_dim, num_classes):
        super(BBoxRegression, self).__init__()
        # 每个类别对应4个回归参数(dx, dy, dw, dh)
        self.regressors = nn.Linear(feature_dim, num_classes * 4)
    
    def forward(self, x):
        # x: [num_proposals, feature_dim]
        # 输出: [num_proposals, num_classes, 4]
        out = self.regressors(x)
        return out.view(out.size(0), -1, 4)


class RCNN(nn.Module):
    """R-CNN整体模型"""
    def __init__(self, num_classes=21):  # 20类+背景
        super(RCNN, self).__init__()
        # 1. 候选区域生成（Selective Search）
        self.selective_search = SelectiveSearch()
        
        # 2. 特征提取网络（AlexNet）
        self.feature_extractor = RCNAlexNet()
        
        # 3. SVM分类器
        self.svm = RCNNSVM(self.feature_extractor.feature_dim, num_classes)
        
        # 4. 边界框回归
        self.bbox_reg = BBoxRegression(self.feature_extractor.feature_dim, num_classes)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_proposal(self, img, proposal, mode='anisotropic'):
        """候选区域预处理：各向异性或各向同性缩放"""
        x1, y1, x2, y2 = proposal
        h, w = y2 - y1, x2 - x1
        
        # 提取候选区域
        proposal_img = img[y1:y2, x1:x2]
        
        if mode == 'anisotropic':
            # 各向异性缩放：直接缩放到227x227，可能变形
            resized = cv2.resize(proposal_img, (227, 227))
        else:
            # 各向同性缩放：先填充再裁剪或先裁剪再填充
            # 方法1：先扩充后裁剪
            size = max(h, w)
            padded = np.zeros((size, size, 3), dtype=np.uint8)
            # 计算填充偏移
            dy = (size - h) // 2
            dx = (size - w) // 2
            padded[dy:dy+h, dx:dx+w] = proposal_img
            # 中心裁剪
            resized = cv2.resize(padded, (227, 227))
        
        return resized
    
    def forward(self, img):
        """
        前向传播流程：
        1. 生成候选区域
        2. 预处理候选区域
        3. 提取特征
        4. 分类预测
        5. 边界框回归
        """
        if isinstance(img, str):
            # 如果输入是图像路径，读取图像
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        
        # 1. 生成候选区域（Selective Search）
        proposals = self.selective_search.generate_proposals(img)
        
        # 2. 预处理候选区域并提取特征
        num_proposals = proposals.shape[0]
        features = torch.zeros(num_proposals, self.feature_extractor.feature_dim)
        
        for i, proposal in enumerate(proposals):
            # 预处理候选区域
            processed = self.preprocess_proposal(img, proposal)
            # 转换为tensor并归一化
            processed_tensor = self.transform(Image.fromarray(processed))[None, :]  # [1, 3, 227, 227]
            
            # 提取特征
            with torch.no_grad():  # 测试时不需要梯度
                feature = self.feature_extractor(processed_tensor)
                features[i] = feature.squeeze()
        
        # 3. SVM分类
        cls_scores = self.svm(features)  # [num_proposals, num_classes]
        
        # 4. 边界框回归
        bbox_deltas = self.bbox_reg(features)  # [num_proposals, num_classes, 4]
        
        return {
            'proposals': proposals,
            'cls_scores': cls_scores,
            'bbox_deltas': bbox_deltas,
            'image_info': (h, w)
        }
    
    def postprocess(self, results, score_threshold=0.5, nms_threshold=0.3):
        """后处理：非极大值抑制和边界框回归应用"""
        proposals = results['proposals']
        cls_scores = results['cls_scores']
        bbox_deltas = results['bbox_deltas']
        h, w = results['image_info']
        
        num_classes = cls_scores.size(1)
        outputs = []
        
        for cls_idx in range(1, num_classes):  # 跳过背景类
            cls_name = config.CLASS_NAMES[cls_idx - 1]  # 假设config中有类别名称
            scores = cls_scores[:, cls_idx]
            
            # 过滤低分数候选框
            keep_idx = scores > score_threshold
            if keep_idx.sum() == 0:
                continue
            
            # 应用边界框回归
            proposals_reg = proposals[keep_idx].astype(np.float32)
            deltas = bbox_deltas[keep_idx, cls_idx].numpy()
            
            # 边界框回归公式:
            # G = P * d_g, 其中 d_g = (dx, dy, dw, dh)
            # dx = (G_x - P_x)/P_w, dy = (G_y - P_y)/P_h
            # dw = log(G_w/P_w), dh = log(G_h/P_h)
            # 因此: G_x = P_w * dx + P_x, G_y = P_h * dy + P_y
            # G_w = P_w * exp(dw), G_h = P_h * exp(dh)
            
            px = (proposals_reg[:, 0] + proposals_reg[:, 2]) / 2.0
            py = (proposals_reg[:, 1] + proposals_reg[:, 3]) / 2.0
            pw = proposals_reg[:, 2] - proposals_reg[:, 0]
            ph = proposals_reg[:, 3] - proposals_reg[:, 1]
            
            gx = deltas[:, 0] * pw[:, np.newaxis] + px[:, np.newaxis]
            gy = deltas[:, 1] * ph[:, np.newaxis] + py[:, np.newaxis]
            gw = np.exp(deltas[:, 2]) * pw[:, np.newaxis]
            gh = np.exp(deltas[:, 3]) * ph[:, np.newaxis]
            
            # 转换为边界框坐标 [x1, y1, x2, y2]
            x1 = gx - gw / 2.0
            y1 = gy - gh / 2.0
            x2 = gx + gw / 2.0
            y2 = gy + gh / 2.0
            
            # 确保边界框在图像内
            x1 = np.clip(x1, 0, w - 1)
            y1 = np.clip(y1, 0, h - 1)
            x2 = np.clip(x2, 0, w - 1)
            y2 = np.clip(y2, 0, h - 1)
            
            # 组合结果
            bboxes = np.concatenate([x1, y1, x2, y2], axis=1)
            scores_filtered = scores[keep_idx].numpy()
            
            # 非极大值抑制
            keep = self.nms(bboxes, scores_filtered, nms_threshold)
            
            if len(keep) > 0:
                outputs.append({
                    'class': cls_name,
                    'bboxes': bboxes[keep],
                    'scores': scores_filtered[keep]
                })
        
        return outputs
    
    def nms(self, bboxes, scores, threshold):
        """非极大值抑制"""
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # 计算重叠区域
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            overlap = w * h / (areas[i] + areas[order[1:]] - overlap)
            
            # 保留重叠小于阈值的索引
            inds = np.where(overlap <= threshold)[0]
            order = order[inds + 1]  # +1是因为order[1:]
            
        return keep


# 示例使用
if __name__ == "__main__":
    # 初始化模型
    model = RCNN()
    
    # 假设config中定义了类别名称
    config.CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                         'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                         'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor']
    
    # 测试图像
    img_path = 'test.jpg'
    results = model(img_path)
    detections = model.postprocess(results)
    
    # 可视化结果
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    ax = plt.gca()
    
    for det in detections:
        cls_name = det['class']
        bboxes = det['bboxes']
        scores = det['scores']
        
        for bbox, score in zip(bboxes, scores):
            x1, y1, x2, y2 = bbox.astype(int)
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                      edgecolor='red', facecolor='none', linewidth=2))
            ax.text(x1, y1 - 10, f'{cls_name}: {score:.3f}', 
                   bbox=dict(facecolor='blue', alpha=0.5), color='white', fontsize=10)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()