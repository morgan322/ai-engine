import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import RoIPool
import torch.nn.functional as F


NUM_CLASSES = 21  # 分类数（含背景）
ROI_SIZE = 7      # RoI Pooling 输出尺寸
SPATIAL_SCALE = 1/16  # 特征图与原图空间缩放比例



class FastRCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(FastRCNN, self).__init__()
        # 1. 骨干网络（以 ResNet50 为例，可替换为 VGG 等）
        backbone = models.resnet50(pretrained=True)
        # 取 ResNet50 的 layer4 之前部分，输出 stride=16 的特征图
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        self.backbone_out_channels = 2048  # ResNet50 layer4 输出通道数

        # 2. RoI Pooling 层
        self.roi_pool = RoIPool(
            (ROI_SIZE, ROI_SIZE), 
            spatial_scale=SPATIAL_SCALE
        )

        # 3. 检测头（分类 + 回归）
        # 先将 RoI Pooling 输出展平后的维度：2048 * 7 * 7
        self.fc1 = nn.Linear(self.backbone_out_channels * ROI_SIZE * ROI_SIZE, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        # 分类头：预测类别（含背景）
        self.cls_head = nn.Linear(1024, num_classes)
        # 回归头：预测边界框偏移（每个类别对应 4 个参数）
        self.reg_head = nn.Linear(1024, num_classes * 4)

    def forward(self, images, rois):
        """
        :param images: 输入图像，形状 [B, 3, H, W]
        :param rois: 感兴趣区域，形状 [N, 5]（格式：[batch_idx, x1, y1, x2, y2]）
        :return: 分类预测、回归预测
        """
        # 1. 骨干网络提取特征
        features = self.backbone(images)  # [B, 2048, H/16, W/16]

        # 2. RoI Pooling
        # 将 RoI 映射到特征图上，提取固定尺寸特征
        roi_features = self.roi_pool(features, rois)  
        # 展平特征：[N, 2048 * 7 * 7]
        roi_features = roi_features.view(roi_features.size(0), -1)  

        # 3. 全连接层处理
        x = F.relu(self.fc1(roi_features), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        # 4. 分类与回归预测
        cls_pred = self.cls_head(x)  # [N, num_classes]
        reg_pred = self.reg_head(x)  # [N, num_classes * 4]
        reg_pred = reg_pred.view(reg_pred.size(0), -1, 4)  # [N, num_classes, 4]

        return cls_pred, reg_pred