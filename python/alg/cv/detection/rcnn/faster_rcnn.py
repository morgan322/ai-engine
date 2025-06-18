import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import RoIPool

# 假设 config 中定义了必要参数，如：
# config.NUM_CLASSES = 21  # 含背景的类别数（VOC 数据集 20 类 + 1 背景）
# config.ROI_SIZE = 7     # RoI Pooling 输出尺寸
# config.RPN_ANCHOR_SCALES = [8, 16, 32]  # 锚框尺度（示例）
# config.RPN_ANCHOR_RATIOS = [0.5, 1, 2]  # 锚框长宽比（示例）
import config  


class RPN(nn.Module):
    """区域提议网络（Region Proposal Network）"""
    def __init__(self, in_channels, mid_channels=512, num_anchors=9):
        super(RPN, self).__init__()
        # 3x3 卷积提取特征
        self.conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        # 分类头：预测锚框是否为前景（二分类）
        self.cls_head = nn.Conv2d(mid_channels, num_anchors * 2, 1, 1, 0)  
        # 回归头：预测锚框坐标偏移（dx, dy, dw, dh）
        self.reg_head = nn.Conv2d(mid_channels, num_anchors * 4, 1, 1, 0)  

    def forward(self, x):
        # x: 骨干网络输出的特征图，形状 [B, C, H, W]
        x = self.conv(x)
        x = nn.functional.relu(x, inplace=True)
        
        # 分类预测：reshape 为 [B, 2*num_anchors, H, W] → 后处理为概率
        cls_pred = self.cls_head(x)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 2*num_anchors]
        cls_pred = cls_pred.view(cls_pred.shape[0], -1, 2)     # [B, H*W*num_anchors, 2]
        
        # 回归预测：reshape 为 [B, 4*num_anchors, H, W] → 坐标偏移
        reg_pred = self.reg_head(x)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 4*num_anchors]
        reg_pred = reg_pred.view(reg_pred.shape[0], -1, 4)     # [B, H*W*num_anchors, 4]
        
        return cls_pred, reg_pred


class RoIHead(nn.Module):
    """检测头（RoI 分类 + 回归）"""
    def __init__(self, in_features, num_classes):
        super(RoIHead, self).__init__()
        # RoI Pooling：将不同尺寸 RoI 映射到固定尺寸
        self.roi_pool = RoIPool((config.ROI_SIZE, config.ROI_SIZE), spatial_scale=1/16)  # 假设骨干网络下采样 16 倍
        # 全连接层
        self.fc1 = nn.Linear(in_features * config.ROI_SIZE * config.ROI_SIZE, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        # 分类头：预测类别
        self.cls_head = nn.Linear(1024, num_classes)
        # 回归头：预测边界框精细调整
        self.reg_head = nn.Linear(1024, num_classes * 4)

    def forward(self, x, rois):
        # x: 骨干网络输出的特征图；rois: 候选区域，形状 [num_rois, 5]（格式：[batch_idx, x1, y1, x2, y2]）
        roi_features = self.roi_pool(x, rois)
        # 展平特征
        roi_features = roi_features.view(roi_features.size(0), -1)
        
        # 全连接层
        x = nn.functional.relu(self.fc1(roi_features), inplace=True)
        x = nn.functional.relu(self.fc2(x), inplace=True)
        
        # 分类预测（含背景）
        cls_pred = self.cls_head(x)
        # 回归预测（每个类别对应边界框偏移）
        reg_pred = self.reg_head(x)
        reg_pred = reg_pred.view(reg_pred.size(0), -1, 4)  # [num_rois, num_classes, 4]
        
        return cls_pred, reg_pred


class FasterRCNN(nn.Module):
    """Faster R-CNN 整体模型"""
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(FasterRCNN, self).__init__()
        # 1. 骨干网络（示例用 ResNet50，可替换为其他）
        backbone = models.resnet50(pretrained=True)
        # 取 ResNet50 的 layer4 之前的层作为特征提取（输出 stride=16 的特征图）
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
        # 骨干网络输出通道数（ResNet50 layer4 输出 2048 维）
        self.backbone_out_channels = 2048  

        # 2. 区域提议网络（RPN）
        self.rpn = RPN(self.backbone_out_channels, num_anchors=len(config.RPN_ANCHOR_SCALES)*len(config.RPN_ANCHOR_RATIOS))

        # 3. 检测头（RoI Head）
        self.roi_head = RoIHead(self.backbone_out_channels, num_classes)

    def forward(self, images, rois=None):
        # images: 输入图像，形状 [B, 3, H, W]
        # rois: 训练时是 RPN 输出的候选框；测试时可动态生成，这里简化处理
        
        # 1. 骨干网络提取特征
        features = self.backbone(images)  # [B, 2048, H/16, W/16]

        # 2. RPN 生成候选区域和预测
        if self.training:
            # 训练阶段：RPN 输出分类、回归预测，用于生成 RoI
            rpn_cls_pred, rpn_reg_pred = self.rpn(features)
            # （实际需结合锚框、NMS 等生成 RoI，这里简化，假设 rois 已处理好传入）
            # 测试阶段：可通过 rpn 输出的预测生成候选框
            pass

        # 3. RoI Head 检测（训练/测试逻辑需完善，这里示意流程）
        if rois is not None:
            roi_cls_pred, roi_reg_pred = self.roi_head(features, rois)
            return rpn_cls_pred, rpn_reg_pred, roi_cls_pred, roi_reg_pred
        else:
            # 测试阶段需先通过 RPN 生成 rois，再走检测头，这里简化返回特征
            return features