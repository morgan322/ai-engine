import math
import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision
import numpy as np

# ---------------------------------------------------------------------#
#   残差结构（Darknet53 内部使用）
# ---------------------------------------------------------------------#
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out += residual
        return out

# ---------------------------------------------------------------------#
#   Darknet53（Backbone）
# ---------------------------------------------------------------------#
class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        # 输入：416x416x3 -> 416x416x32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        # 构建 Darknet53 的 5 个阶段
        self.layer1 = self._make_layer([32, 64], layers[0])   # 416->208
        self.layer2 = self._make_layer([64, 128], layers[1])  # 208->104
        self.layer3 = self._make_layer([128, 256], layers[2]) # 104->52
        self.layer4 = self._make_layer([256, 512], layers[3]) # 52->26
        self.layer5 = self._make_layer([512, 1024], layers[4])# 26->13

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        layers = []
        # 下采样（步长 2，3x3 卷积）
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 堆叠残差块
        self.inplanes = planes[1]
        for i in range(blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)   # 208x208x64
        x = self.layer2(x)   # 104x104x128
        out3 = self.layer3(x)# 52x52x256（对应 YOLOv3 的 middle 尺度）
        out4 = self.layer4(out3)# 26x26x512（对应 YOLOv3 的 large 尺度）
        out5 = self.layer5(out4)# 13x13x1024（对应 YOLOv3 的 small 尺度）

        return out3, out4, out5

# ---------------------------------------------------------------------#
#   改进的检测头，包含自适应锚框机制
# ---------------------------------------------------------------------#
class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, anchor_softmax=False):
        super(YOLOHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.anchor_softmax = anchor_softmax  # 是否使用锚框softmax机制
        
        # 3 次卷积提取特征
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels * 2)
        self.relu1 = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels * 2)
        self.relu2 = nn.LeakyReLU(0.1)
        
        self.conv3 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.relu3 = nn.LeakyReLU(0.1)
        
        # 改进的预测层：分离锚框尺寸预测
        if anchor_softmax:
            # 锚框尺寸使用softmax归一化
            self.anchor_pred = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=1, stride=1, padding=0)
            self.cls_conf_pred = nn.Conv2d(in_channels, num_anchors * (num_classes + 1), kernel_size=1, stride=1, padding=0)
        else:
            # 传统锚框预测
            self.prediction = nn.Conv2d(in_channels, num_anchors * (num_classes + 5), kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        if self.anchor_softmax:
            # 分离预测锚框尺寸和类别置信度
            anchor_pred = self.anchor_pred(x)
            cls_conf_pred = self.cls_conf_pred(x)
            return anchor_pred, cls_conf_pred
        else:
            x = self.prediction(x)
            return x

# ---------------------------------------------------------------------#
#   自适应锚框生成模块
# ---------------------------------------------------------------------#
class AdaptiveAnchorGenerator(nn.Module):
    def __init__(self, strides, initial_anchors, anchor_softmax=False, num_anchors=3):
        super(AdaptiveAnchorGenerator, self).__init__()
        self.strides = strides
        self.initial_anchors = initial_anchors
        self.anchor_softmax = anchor_softmax
        self.num_anchors = num_anchors
        
        # 注册初始锚框为缓冲区
        self.register_buffer("base_anchors", torch.tensor(initial_anchors).float())
        
        if anchor_softmax:
            # 可学习的锚框尺寸偏移
            self.anchor_offset = nn.Parameter(torch.zeros(1, num_anchors, 2))
    
    def forward(self, feature_maps, anchor_preds=None):
        anchor_grids = []
        for i, (feat_map, stride) in enumerate(zip(feature_maps, self.strides)):
            bs, _, h, w = feat_map.shape
            device = feat_map.device
            
            if self.anchor_softmax and anchor_preds is not None:
                # 自适应锚框生成（带softmax）
                anchor_pred = anchor_preds[i]
                anchor_pred = anchor_pred.view(bs, self.num_anchors, 2, h, w)
                anchor_pred = torch.softmax(anchor_pred, dim=2)  # 归一化到(0,1)
                
                # 基于初始锚框和预测偏移生成自适应锚框
                base_anchors = self.base_anchors[i].to(device).view(1, self.num_anchors, 2, 1, 1)
                adaptive_anchors = base_anchors * anchor_pred * stride
                
                # 调整锚框维度以匹配网格
                adaptive_anchors = adaptive_anchors.permute(0, 1, 3, 4, 2)  # (bs, na, h, w, 2)
            else:
                # 传统锚框生成
                base_anchors = self.base_anchors[i].to(device)
                adaptive_anchors = base_anchors.view(1, self.num_anchors, 2, 1, 1) * stride
                adaptive_anchors = adaptive_anchors.permute(0, 1, 3, 4, 2)  # (bs, na, h, w, 2)
            
            # 生成网格
            y, x = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij"
            )
            grid = torch.stack((x, y), 2).view(1, 1, h, w, 2) * stride
            grid = grid.expand(bs, self.num_anchors, h, w, 2)  # 扩展以匹配锚框数量
            
            anchor_grids.append((grid, adaptive_anchors))
        
        return anchor_grids

# ---------------------------------------------------------------------#
#   改进的YOLOv3检测头，包含自适应锚框机制
# ---------------------------------------------------------------------#
class YOLOv3(nn.Module):
    def __init__(self, num_classes, anchors, anchor_softmax=False):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors[0])
        self.anchor_softmax = anchor_softmax
        
        # 1. Backbone: Darknet53
        self.backbone = DarkNet([1, 2, 8, 8, 4])
        
        # 2. 定义特征图步长
        self.strides = [8, 16, 32]  # 52x52, 26x26, 13x13特征图对应的步长
        
        # 3. Neck: 特征金字塔（FPN）
        # （1）13x13 -> 26x26（上采样 + 融合 26x26 特征）
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_bridge1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv_fuse1 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )
        
        # （2）26x26 -> 52x52（上采样 + 融合 52x52 特征）
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_bridge2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv_fuse2 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        
        # 4. Prediction: 3 个检测头
        if anchor_softmax:
            self.head_small = YOLOHead(1024, self.num_anchors, self.num_classes, anchor_softmax=True)
            self.head_mid = YOLOHead(256, self.num_anchors, self.num_classes, anchor_softmax=True)
            self.head_large = YOLOHead(128, self.num_anchors, self.num_classes, anchor_softmax=True)
        else:
            self.head_small = YOLOHead(1024, self.num_anchors, self.num_classes)
            self.head_mid = YOLOHead(256, self.num_anchors, self.num_classes)
            self.head_large = YOLOHead(128, self.num_anchors, self.num_classes)
        
        # 5. 自适应锚框生成器
        self.anchor_generator = AdaptiveAnchorGenerator(
            strides=self.strides,
            initial_anchors=anchors,
            anchor_softmax=anchor_softmax,
            num_anchors=self.num_anchors
        )

    def forward(self, x):
        # 1. Backbone 提取特征
        out3, out4, out5 = self.backbone(x)  # out3:52x52x256, out4:26x26x512, out5:13x13x1024

        # 2. Neck: 特征融合（FPN）
        # （1）处理 13x13 特征 -> 26x26
        out5_bridge = self.conv_bridge1(out5)  # 13x13x1024 -> 13x13x256
        out5_upsampled = self.upsample1(out5_bridge)  # 13x13x256 -> 26x26x256
        out4_fused = torch.cat([out5_upsampled, out4], dim=1)  # 26x26x(256+512)=26x26x768
        out4_fused = self.conv_fuse1(out4_fused)  # 26x26x768 -> 26x26x256
        
        # （2）处理 26x26 特征 -> 52x52
        out4_bridge = self.conv_bridge2(out4_fused)  # 26x26x256 -> 26x26x128
        out4_upsampled = self.upsample2(out4_bridge)  # 26x26x128 -> 52x52x128
        out3_fused = torch.cat([out4_upsampled, out3], dim=1)  # 52x52x(128+256)=52x52x384
        out3_fused = self.conv_fuse2(out3_fused)  # 52x52x384 -> 52x52x128
        
        # 3. Prediction: 3 个检测头
        if self.anchor_softmax:
            # 自适应锚框模式
            anchor_pred_small, cls_conf_small = self.head_small(out5)
            anchor_pred_mid, cls_conf_mid = self.head_mid(out4_fused)
            anchor_pred_large, cls_conf_large = self.head_large(out3_fused)
            
            # 生成锚框和网格
            feature_maps = [out3_fused, out4_fused, out5]
            anchor_preds = [anchor_pred_large, anchor_pred_mid, anchor_pred_small]
            anchor_grids = self.anchor_generator(feature_maps, anchor_preds)
            
            # 解码预测结果
            preds = self._decode_predictions(
                [cls_conf_large, cls_conf_mid, cls_conf_small],
                anchor_grids
            )
        else:
            # 传统锚框模式
            pred_small = self.head_small(out5)
            pred_mid = self.head_mid(out4_fused)
            pred_large = self.head_large(out3_fused)
            
            # 生成锚框和网格
            feature_maps = [out3_fused, out4_fused, out5]
            anchor_grids = self.anchor_generator(feature_maps)
            
            # 解码预测结果
            preds = self._decode_predictions(
                [pred_large, pred_mid, pred_small],
                anchor_grids
            )
        
        return preds

    def _decode_predictions(self, predictions, anchor_grids):
        decoded_preds = []
        for i, (pred, (grid, anchors)) in enumerate(zip(predictions, anchor_grids)):
            bs, _, h, w = pred.shape
            na = self.num_anchors
            nc = self.num_classes
            
            if self.anchor_softmax:
                # 自适应锚框解码
                pred = pred.view(bs, na, nc + 1, h, w)
                pred = pred.permute(0, 1, 3, 4, 2)  # (bs, na, h, w, nc+1)
                conf = pred[..., 0].sigmoid()
                cls = pred[..., 1:].softmax(dim=-1)
                xy = grid  # (bs, na, h, w, 2)
                wh = anchors  # (bs, na, h, w, 2)
            else:
                # 传统锚框解码
                pred = pred.view(bs, na, 5 + nc, h, w)
                pred = pred.permute(0, 1, 3, 4, 2)  # (bs, na, h, w, 5+nc)
                xy = pred[..., :2].sigmoid() * 2.0 + grid - 0.5
                wh = (pred[..., 2:4].sigmoid() * 2.0) ** 2 * anchors
                conf = pred[..., 4:5].sigmoid()
                cls = pred[..., 5:].softmax(dim=-1)
            
            # 组合预测结果 [x, y, w, h, conf, cls1, cls2, ...]
            xywh = torch.cat([xy, wh], dim=-1)  # (bs, na, h, w, 4)
            pred = torch.cat([xywh, conf.unsqueeze(-1), cls], dim=-1)  # (bs, na, h, w, 4+1+nc)
            decoded_preds.append(pred.view(bs, na*h*w, -1))
        
        return torch.cat(decoded_preds, dim=1)

# ---------------------------------------------------------------------#
#   测试代码
# ---------------------------------------------------------------------#
if __name__ == "__main__":
    # 配置：类别数、 anchors（3 个尺度，每个尺度 3 个 anchors）
    num_classes = 80  # COCO 数据集类别数
    anchors = [[(10,13), (16,30), (33,23)],  # 小目标（13x13）
               [(30,61), (62,45), (59,119)], # 中目标（26x26）
               [(116,90), (156,198), (373,326)]] # 大目标（52x52）
    
    # 构建改进的YOLOv3
    model = YOLOv3(num_classes, anchors, anchor_softmax=True)
    print("改进后的YOLOv3模型结构：")
    print(model)
    
    # 测试前向传播
    input_tensor = torch.randn(1, 3, 416, 416)  # 输入：1 张 416x416 的图像
    outputs = model(input_tensor)
    print("输出尺寸：", outputs.shape)