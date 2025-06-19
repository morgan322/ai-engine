import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Dict, Optional

# 标签名称（COCO数据集80类）
label_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]
num_classes = len(label_names)

# YOLOv2默认锚框（基于COCO数据集聚类得到）
anchors = torch.tensor([
    [1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892],
    [9.47112, 4.84053], [11.2364, 10.0071]
], dtype=torch.float32)
num_anchors = len(anchors)

# 边界框操作工具函数
def bbox_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """计算两组边界框之间的IOU"""
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = torch.max(right_down - left_up, torch.zeros_like(right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / (union_area + 1e-6)

def bbox_wh_iou(wh1: torch.Tensor, wh2: torch.Tensor) -> torch.Tensor:
    """计算宽高的IOU（用于锚框匹配）"""
    wh2 = wh2.t().unsqueeze(2)  # [2, num_anchors, 1]
    wh1 = wh1.t().unsqueeze(1)  # [2, 1, num_boxes]
    
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + (w2 * h2 + 1e-16) - inter_area
    
    return inter_area / union_area

def build_targets(predictions: torch.Tensor, targets: torch.Tensor, anchors: torch.Tensor,
                  num_classes: int, ignore_thresh: float = 0.5) -> Tuple[torch.Tensor, ...]:
    """构建训练目标（计算损失所需的掩码和目标值）"""
    batch_size, num_anchors, grid_size, _, _ = predictions.shape
    anchor_wh = anchors.to(predictions.device)
    
    # 创建目标张量
    obj_mask = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.bool, device=predictions.device)
    noobj_mask = torch.ones((batch_size, num_anchors, grid_size, grid_size), dtype=torch.bool, device=predictions.device)
    tx = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.float32, device=predictions.device)
    ty = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.float32, device=predictions.device)
    tw = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.float32, device=predictions.device)
    th = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.float32, device=predictions.device)
    tcls = torch.zeros((batch_size, num_anchors, grid_size, grid_size, num_classes), dtype=torch.float32, device=predictions.device)
    
    # 处理每个样本
    for b in range(batch_size):
        if len(targets[b]) == 0:
            continue
            
        # 获取当前样本的预测和目标
        target_boxes = targets[b][:, 2:6]  # [x, y, w, h] (归一化)
        target_labels = targets[b][:, 1].long()
        
        # 将目标边界框从归一化坐标转换为网格坐标
        gxy = target_boxes[:, :2] * grid_size
        gwh = target_boxes[:, 2:4] * grid_size
        
        # 计算每个目标与锚框的IOU
        ious = bbox_wh_iou(gwh.t(), anchor_wh.t())
        best_ious, best_n = ious.max(0)  # 每个目标的最佳锚框
        
        # 确定每个目标的网格位置
        gi, gj = gxy.long().t()
        
        # 设置掩码和目标值
        for i, (n, gi, gj) in enumerate(zip(best_n, gi, gj)):
            if ious[i, n] < ignore_thresh:
                continue
                
            # 更新目标掩码
            obj_mask[b, n, gj, gi] = 1
            noobj_mask[b, n, gj, gi] = 0
            
            # 边界框中心坐标（相对于网格）
            tx[b, n, gj, gi] = gxy[i, 0] - gi.float()
            ty[b, n, gj, gi] = gxy[i, 1] - gj.float()
            
            # 边界框宽高（对数空间）
            tw[b, n, gj, gi] = torch.log(gwh[i, 0] / anchor_wh[n, 0] + 1e-16)
            th[b, n, gj, gi] = torch.log(gwh[i, 1] / anchor_wh[n, 1] + 1e-16)
            
            # 类别标签
            tcls[b, n, gj, gi, target_labels[i]] = 1
            
        # 对于IOU大于阈值但不是最佳匹配的锚框，不计算无目标损失
        for iou, n, gi, gj in zip(ious.t(), range(num_anchors), gi, gj):
            if iou.max() > ignore_thresh:
                noobj_mask[b, n, gj, gi] = 0
                
    return obj_mask, noobj_mask, tx, ty, tw, th, tcls

# Reorg层实现（替代原Cython版本）
class ReorgLayer(nn.Module):
    def __init__(self, stride: int = 2):
        super(ReorgLayer, self).__init__()
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()
        stride = self.stride
        
        # 重新排列张量实现空间到通道的转换
        new_height, new_width = height // stride, width // stride
        new_channels = channels * stride * stride
        
        x = x.view(batch_size, channels, new_height, stride, new_width, stride)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(batch_size, new_channels, new_height, new_width)
        
        return x

# 基本卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, activation: bool = True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True) if activation else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

# YOLOv2模型（基于Darknet19）
class YOLOv2(nn.Module):
    def __init__(self, num_classes: int = 80, anchors: Optional[torch.Tensor] = None):
        super(YOLOv2, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors if anchors is not None else torch.tensor([
            [1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892],
            [9.47112, 4.84053], [11.2364, 10.0071]
        ])
        self.num_anchors = len(self.anchors)
        
        # 构建Darknet19骨干网络
        self.darknet19 = self._build_darknet19()
        
        # 额外的卷积层
        self.extra_conv = nn.Sequential(
            ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1)
        )
        
        # 从中间层提取特征（用于passthrough连接）
        self.passthrough = ReorgLayer(stride=2)  # 将26x26特征图重组为13x13但通道数x4
        
        # 最终检测层
        self.detection = nn.Sequential(
            ConvBlock(1024 + 512*4, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, self.num_anchors * (5 + self.num_classes), kernel_size=1, stride=1, padding=0)
        )
        
    def _build_darknet19(self) -> nn.Sequential:
        """构建Darknet19骨干网络"""
        layers = nn.Sequential(
            # 第一组
            ConvBlock(3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二组
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三组
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 64, kernel_size=1, stride=1, padding=0),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第四组
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 128, kernel_size=1, stride=1, padding=0),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第五组（最后一个MaxPool2d被移除，保留512x26x26特征图）
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
        )
        
        # 保存第五组的输出用于passthrough连接
        self.feature_layer = 18  # 第五组最后一个卷积层的索引
        
        # 第六组和第七组
        layers.add_module('19', nn.MaxPool2d(kernel_size=2, stride=2))
        layers.add_module('20', ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1))
        layers.add_module('21', ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0))
        layers.add_module('22', ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1))
        layers.add_module('23', ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0))
        layers.add_module('24', ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1))
        
        return layers
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 特征提取
        for i, layer in enumerate(self.darknet19):
            x = layer(x)
            if i == self.feature_layer:
                passthrough = x  # 保存26x26特征图用于passthrough连接
                
        # 额外卷积层
        x = self.extra_conv(x)
        
        # Passthrough连接（特征融合）
        passthrough = self.passthrough(passthrough)
        x = torch.cat([passthrough, x], dim=1)
        
        # 检测层
        x = self.detection(x)
        
        # 重塑输出以匹配YOLO格式
        batch_size, _, height, width = x.size()
        x = x.view(batch_size, self.num_anchors, 5 + self.num_classes, height, width)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # [B, A, H, W, 5+C]
        
        return x

# YOLOv2损失函数
class YOLOv2Loss(nn.Module):
    def __init__(self, num_classes: int = 80, anchors: Optional[torch.Tensor] = None, 
                 ignore_thresh: float = 0.5, coord_scale: float = 5.0, 
                 noobj_scale: float = 1.0, obj_scale: float = 5.0, 
                 class_scale: float = 1.0):
        super(YOLOv2Loss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors if anchors is not None else torch.tensor([
            [1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892],
            [9.47112, 4.84053], [11.2364, 10.0071]
        ])
        self.num_anchors = len(self.anchors)
        self.ignore_thresh = ignore_thresh
        self.coord_scale = coord_scale
        self.noobj_scale = noobj_scale
        self.obj_scale = obj_scale
        self.class_scale = class_scale
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size, num_anchors, grid_size, _, _ = predictions.shape
        
        # 提取预测值
        pred_x = torch.sigmoid(predictions[..., 0])  # 中心x坐标（sigmoid激活）
        pred_y = torch.sigmoid(predictions[..., 1])  # 中心y坐标（sigmoid激活）
        pred_w = predictions[..., 2]                  # 宽（对数空间）
        pred_h = predictions[..., 3]                  # 高（对数空间）
        pred_conf = torch.sigmoid(predictions[..., 4])  # 置信度（sigmoid激活）
        pred_cls = torch.sigmoid(predictions[..., 5:])  # 类别预测（sigmoid激活）
        
        # 构建训练目标
        obj_mask, noobj_mask, tx, ty, tw, th, tcls = build_targets(
            predictions, targets, self.anchors.to(predictions.device), 
            self.num_classes, self.ignore_thresh
        )
        
        # 创建网格坐标
        grid_x = torch.arange(grid_size, dtype=torch.float32, device=predictions.device).reshape(1, 1, 1, grid_size)
        grid_y = torch.arange(grid_size, dtype=torch.float32, device=predictions.device).reshape(1, 1, grid_size, 1)
        
        # 扩展锚框维度
        anchor_w = self.anchors[:, 0].to(predictions.device).reshape(1, self.num_anchors, 1, 1)
        anchor_h = self.anchors[:, 1].to(predictions.device).reshape(1, self.num_anchors, 1, 1)
        
        # 计算预测的边界框（相对于图像）
        pred_boxes = torch.zeros_like(predictions[..., :4], device=predictions.device)
        pred_boxes[..., 0] = pred_x + grid_x
        pred_boxes[..., 1] = pred_y + grid_y
        pred_boxes[..., 2] = torch.exp(pred_w) * anchor_w
        pred_boxes[..., 3] = torch.exp(pred_h) * anchor_h
        
        # 计算目标边界框（相对于图像）
        target_boxes = torch.zeros_like(predictions[..., :4], device=predictions.device)
        target_boxes[..., 0] = tx + grid_x
        target_boxes[..., 1] = ty + grid_y
        target_boxes[..., 2] = torch.exp(tw) * anchor_w
        target_boxes[..., 3] = torch.exp(th) * anchor_h
        
        # 计算边界框IOU（用于置信度目标）
        iou = bbox_iou(pred_boxes, target_boxes)
        
        # 计算各项损失
        # 边界框坐标损失
        loss_x = self.coord_scale * F.mse_loss(pred_x[obj_mask], tx[obj_mask], reduction='sum')
        loss_y = self.coord_scale * F.mse_loss(pred_y[obj_mask], ty[obj_mask], reduction='sum')
        loss_w = self.coord_scale * F.mse_loss(pred_w[obj_mask], tw[obj_mask], reduction='sum')
        loss_h = self.coord_scale * F.mse_loss(pred_h[obj_mask], th[obj_mask], reduction='sum')
        
        # 置信度损失
        loss_conf_obj = self.obj_scale * F.mse_loss(pred_conf[obj_mask], iou[obj_mask], reduction='sum')
        loss_conf_noobj = self.noobj_scale * F.mse_loss(pred_conf[noobj_mask], torch.zeros_like(pred_conf[noobj_mask]), reduction='sum')
        
        # 类别损失
        loss_cls = self.class_scale * F.binary_cross_entropy(
            pred_cls[obj_mask], tcls[obj_mask], reduction='sum'
        )
        
        # 总损失
        total_loss = (loss_x + loss_y + loss_w + loss_h + loss_conf_obj + loss_conf_noobj + loss_cls) / batch_size
        
        return total_loss

# 示例训练代码
def train_yolov2():
    # 创建模型
    model = YOLOv2(num_classes=num_classes, anchors=anchors)
    
    # 创建损失函数
    criterion = YOLOv2Loss(num_classes=num_classes, anchors=anchors)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 模拟训练数据
    batch_size = 4
    input_size = 416
    dummy_input = torch.randn(batch_size, 3, input_size, input_size)
    
    # 模拟目标数据（格式：[batch_idx, class_idx, x_center, y_center, width, height]）
    dummy_targets = [
        torch.tensor([[0, 5, 0.5, 0.5, 0.8, 0.6]]),  # 批次0中的一个目标（类别5，边界框）
        torch.tensor([[1, 2, 0.3, 0.4, 0.4, 0.7]]),  # 批次1中的一个目标
        torch.tensor([]),  # 批次2中没有目标
        torch.tensor([[3, 10, 0.7, 0.2, 0.2, 0.3], [3, 20, 0.4, 0.6, 0.5, 0.5]])  # 批次3中的两个目标
    ]
    
    # 训练循环
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(dummy_input)
        
        # 计算损失
        loss = criterion(outputs, dummy_targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    # 创建模型并打印结构
    model = YOLOv2(num_classes=num_classes)
    print(model)
    
    # 示例训练
    # train_yolov2()