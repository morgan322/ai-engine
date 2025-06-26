import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision

# ResNet50骨干网络实现
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, c2, c3, c4

# 特征金字塔网络(FPN)实现
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        
        # 用于处理最高层特征的额外层
        self.extra_blocks = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        # 获取各层特征
        c2, c3, c4, c5 = x
        
        # 构建特征金字塔
        p5 = self.inner_blocks[3](c5)
        p4 = self.inner_blocks[2](c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.inner_blocks[1](c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.inner_blocks[0](c2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        
        # 应用3x3卷积以获得最终特征图
        p2 = self.layer_blocks[0](p2)
        p3 = self.layer_blocks[1](p3)
        p4 = self.layer_blocks[2](p4)
        p5 = self.layer_blocks[3](p5)
        
        # 额外的下采样层
        p6 = self.extra_blocks(p5)
        
        return {"0": p2, "1": p3, "2": p4, "3": p5, "pool": p6}

# 多尺度RoIAlign实现
class MultiScaleRoIAlign(nn.Module):
    def __init__(self, output_size, sampling_ratio):
        super(MultiScaleRoIAlign, self).__init__()
        # 确保 output_size 是元组 (height, width)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        
    def forward(self, x, boxes, image_shapes):
        """
        x: 字典，包含不同层级的特征图
        boxes: 列表，每个元素是一个batch的boxes [N, 4]
        image_shapes: 列表，原始图像尺寸
        """
        if not isinstance(x, dict):
            x = {"0": x}
            
        rois = []
        rois_indices = []
        
        # 将所有batch的boxes合并，并添加batch索引
        for i, boxes_per_image in enumerate(boxes):
            if boxes_per_image.numel() == 0:
                continue
            rois.append(boxes_per_image)
            rois_indices.append(torch.full((boxes_per_image.shape[0], 1), i,
                                          dtype=torch.float32, device=boxes_per_image.device))
        
        if not rois:
            return torch.zeros((0, list(x.values())[0].shape[1], 
                               self.output_size[0], self.output_size[1]),
                              device=list(x.values())[0].device)
        
        rois = torch.cat(rois, dim=0)
        rois_indices = torch.cat(rois_indices, dim=0)
        
        # 将boxes与batch索引连接 [batch_idx, x1, y1, x2, y2]
        rois = torch.cat([rois_indices, rois], dim=1)
        
        # 确定每个roi应该分配到哪个特征层级
        levels = self._assign_levels(rois, x)
        
        # 对每个层级的特征进行RoIAlign
        result = []
        for level_name, level_features in x.items():
            level = int(level_name)
            level_mask = (levels == level)
            
            if level_mask.sum() == 0:
                continue
                
            level_rois = rois[level_mask]
            
            # 应用RoIAlign
            aligned_features = self._roi_align(level_features, level_rois, 
                                              self.output_size, self.sampling_ratio)
            result.append(aligned_features)
        
        # 按原始顺序合并结果
        if not result:
            return torch.zeros((0, list(x.values())[0].shape[1], 
                               self.output_size[0], self.output_size[1]),
                              device=list(x.values())[0].device)
        
        # 确保结果按原始RoI顺序排列
        order = torch.argsort(torch.cat([torch.where(levels == i)[0] 
                                         for i in sorted(x.keys())]))
        return torch.cat(result, dim=0)[order]
    
    def _assign_levels(self, rois, featmap_names):
        """将RoI分配到合适的特征层级"""
        x1, y1, x2, y2 = rois[:, 1:2], rois[:, 2:3], rois[:, 3:4], rois[:, 4:5]
        w = x2 - x1
        h = y2 - y1
        
        # 计算RoI的面积的平方根
        roi_level = 4 + torch.log2(torch.sqrt(w * h) / 224.0)
        roi_level = torch.clamp(roi_level, min=2, max=5)
        levels = torch.floor(roi_level).to(torch.int64) - 2  # 映射到0-3
        
        return levels
    
    def _roi_align(self, features, rois, output_size, sampling_ratio):
        """对单个特征层级执行RoIAlign"""
        # 计算空间尺度（基于特征图的步长）
        spatial_scale = 1.0 / (2 ** (2 + int(min(int(k) for k in features.keys()))))
        
        # 执行RoIAlign
        return F.roi_align(
            features, rois, 
            output_size=output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio
        )

# 区域提议网络(RPN)
class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        
        for l in self.children():
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        logits = self.cls_logits(x)
        bbox_reg = self.bbox_pred(x)
        return logits, bbox_reg

# Anchor生成器
class AnchorGenerator(nn.Module):
    def __init__(self, sizes, aspect_ratios):
        super(AnchorGenerator, self).__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}
    
    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device="cpu"):
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()
    
    def set_cell_anchors(self, dtype, device):
        if self.cell_anchors is not None:
            return self.cell_anchors
        
        cell_anchors = []
        for size, aspect_ratio in zip(self.sizes, self.aspect_ratios):
            base_anchor = self.generate_anchors(size, aspect_ratio, dtype, device)
            cell_anchors.append(base_anchor)
        
        self.cell_anchors = cell_anchors
        return cell_anchors
    
    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]
    
    def grid_anchors(self, grid_sizes, strides):
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None
        
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device
            
            # 生成网格中心
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            
            # 将基础anchors与网格中心相加
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
        
        return anchors
    
    def forward(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                   torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
        
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        
        anchors = []
        for _ in range(len(image_list.image_sizes)):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(torch.cat(anchors_in_image, dim=0))
        
        return anchors

# 边界框回归与分类头
class BoxHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        super(BoxHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

class BoxPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(BoxPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        
    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        
        return scores, bbox_deltas

# 掩码预测头
class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        Args:
            in_channels (int): 输入通道数
            layers (list): 每层的输出通道数
            dilation (int): 卷积的扩张率
        """
        d = []
        next_feature = in_channels
        for layer_channels in layers:
            d.append(nn.Conv2d(next_feature, layer_channels, kernel_size=3,
                              stride=1, padding=dilation, dilation=dilation))
            d.append(nn.ReLU(inplace=True))
            next_feature = layer_channels
        super(MaskRCNNHeads, self).__init__(*d)
        self.out_channels = next_feature

class MaskRCNNPredictor(nn.Module):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super(MaskRCNNPredictor, self).__init__()
        self.conv5_mask = nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)

# 图像列表类，用于处理批量图像
class ImageList:
    def __init__(self, tensors, image_sizes):
        """
        Args:
            tensors (tensor): 批量图像的张量
            image_sizes (list[tuple]): 每幅图像的原始尺寸
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)

# 完整的Mask R-CNN模型
class MaskRCNN(nn.Module):
    def __init__(self, num_classes, backbone=None, min_size=800, max_size=1333, 
                image_mean=None, image_std=None):
        super(MaskRCNN, self).__init__()
        
        # 构建骨干网络（如果未提供）
        if backbone is None:
            resnet = ResNet(Bottleneck, [3, 4, 6, 3])
            # 移除最后的全连接层
            self.backbone = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4
            )
            in_channels_list = [256, 512, 1024, 2048]  # ResNet50各层输出通道数
            out_channels = 256
        else:
            self.backbone = backbone
            # 需要根据实际骨干网络设置
            in_channels_list = [256, 512, 1024, 2048]
            out_channels = 256
        
        # 构建FPN
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels)
        
        # 构建RPN
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        self.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        self.rpn_head = RPNHead(out_channels, self.anchor_generator.num_anchors_per_location()[0])
        
        # RPN参数
        self.rpn_pre_nms_top_n_train = 2000
        self.rpn_pre_nms_top_n_test = 1000
        self.rpn_post_nms_top_n_train = 2000
        self.rpn_post_nms_top_n_test = 1000
        self.rpn_nms_thresh = 0.7
        self.rpn_fg_iou_thresh = 0.7
        self.rpn_bg_iou_thresh = 0.3
        self.rpn_batch_size_per_image = 256
        self.rpn_positive_fraction = 0.5
        
        # 构建RoIAlign
        self.roi_align = MultiScaleRoIAlign(output_size=7, sampling_ratio=2)
        
        # 构建Box Head
        resolution = self.roi_align.output_size[0]
        in_channels_box = out_channels * resolution ** 2
        representation_size = 1024
        self.box_head = BoxHead(in_channels_box, representation_size)
        self.box_predictor = BoxPredictor(representation_size, num_classes)
        
        # Box参数
        self.box_score_thresh = 0.05
        self.box_nms_thresh = 0.5
        self.box_detections_per_img = 100
        self.box_fg_iou_thresh = 0.5
        self.box_bg_iou_thresh = 0.5
        self.box_batch_size_per_image = 512
        self.box_positive_fraction = 0.25
        
        # 构建Mask Head
        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        self.mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)
        self.mask_predictor = MaskRCNNPredictor(mask_layers[-1], 256, num_classes)
        
        # 图像变换参数
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean if image_mean is not None else [0.485, 0.456, 0.406]
        self.image_std = image_std if image_std is not None else [0.229, 0.224, 0.225]
        
        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, images, targets=None):
        """
        Args:
            images (list[Tensor]): 输入图像列表，每幅图像的尺寸可以不同
            targets (list[Dict[Tensor]]): 目标字典列表，包含边界框和掩码等信息（训练时需要）
        """
        # 确保模型在正确的设备上
        self.to(self.device)
        
        # 图像预处理
        original_image_sizes = [img.shape[-2:] for img in images]
        images = self.preprocess(images)
        
        # 特征提取
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}
        else:
            features = self.fpn(features)
        
        # RPN前向传播，生成候选区域
        proposals, proposal_losses = self.rpn(images, features, targets)
        
        # RoI heads前向传播，包含box分类和回归
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        
        # 掩码预测
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            detections = self.forward_mask(features, detections, targets)
        else:
            detections = self.forward_mask(features, detections)
        
        # 后处理，调整检测结果到原始图像尺寸
        detections = self.postprocess(detections, images.image_sizes, original_image_sizes)
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        
        if self.training:
            return losses
        
        return detections
    
    def preprocess(self, images):
        """图像预处理：归一化和尺寸调整"""
        # 归一化
        images = [img.to(self.device) for img in images]
        images = [img / 255.0 for img in images]
        images = [self.normalize(img) for img in images]
        
        # 调整图像大小，保持宽高比
        images = self.resize(images)
        
        # 将图像堆叠成批量
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        batch_shape = (len(images),) + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
        
        image_sizes = [img.shape[-2:] for img in images]
        return ImageList(batched_imgs, image_sizes)
    
    def normalize(self, image):
        """图像归一化"""
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]
    
    def resize(self, images):
        """调整图像大小，保持宽高比"""
        # 计算调整后的尺寸
        sizes = [img.shape[-2:] for img in images]
        new_sizes = [self._get_size_with_aspect_ratio(
            size, self.min_size, self.max_size) for size in sizes]
        
        # 调整图像大小
        resized_images = []
        for img, size in zip(images, new_sizes):
            img = F.interpolate(img.unsqueeze(0), size=size, mode='bilinear', 
                               align_corners=False).squeeze(0)
            resized_images.append(img)
        
        return resized_images
    
    def _get_size_with_aspect_ratio(self, image_size, smallest_side, largest_side):
        """计算调整后的尺寸，保持宽高比"""
        height, width = image_size
        size = smallest_side
        max_size = largest_side
        
        if height < width:
            scale = size / height
            if width * scale > max_size:
                scale = max_size / width
            new_height = int(height * scale)
            new_width = int(width * scale)
        else:
            scale = size / width
            if height * scale > max_size:
                scale = max_size / height
            new_height = int(height * scale)
            new_width = int(width * scale)
        
        return (new_height, new_width)
    
    def rpn(self, images, features, targets=None):
        """区域提议网络前向传播"""
        # 提取特征图
        feature_maps = list(features.values())
        
        # 生成anchors
        anchors = self.anchor_generator(images, feature_maps)
        
        # RPN前向传播
        objectness, pred_bbox_deltas = self.rpn_head(feature_maps)
        
        # 根据训练/推理模式选择不同的NMS参数
        pre_nms_top_n = self.rpn_pre_nms_top_n_train if self.training else self.rpn_pre_nms_top_n_test
        post_nms_top_n = self.rpn_post_nms_top_n_train if self.training else self.rpn_post_nms_top_n_test
        
        # 应用NMS并生成最终的proposals
        proposals = self.select_training_proposals(
            anchors, objectness, pred_bbox_deltas, images.image_sizes,
            pre_nms_top_n, post_nms_top_n, self.rpn_nms_thresh
        )
        
        # 计算RPN损失（训练模式下）
        proposal_losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(
                anchors, targets, self.rpn_fg_iou_thresh, self.rpn_bg_iou_thresh
            )
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_rpn_losses(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            proposal_losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        
        return proposals, proposal_losses
    
    def select_training_proposals(self, anchors, objectness, pred_bbox_deltas, image_sizes,
                                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        """选择训练用的proposals，应用NMS"""
        # 提取每个特征图的objectness和bbox预测
        num_images = len(image_sizes)
        device = objectness[0].device
        
        # 将所有特征图的预测结果收集起来
        objectnesses = []
        bbox_deltas = []
        for o, b in zip(objectness, pred_bbox_deltas):
            objectnesses.append(o.permute(0, 2, 3, 1).reshape(num_images, -1))
            bbox_deltas.append(b.permute(0, 2, 3, 1).reshape(num_images, -1, 4))
        
        objectnesses = torch.cat(objectnesses, dim=1)
        bbox_deltas = torch.cat(bbox_deltas, dim=1)
        
        # 应用边界框回归
        proposals = self.box_coder.decode(bbox_deltas, anchors)
        
        # 对每个图像分别处理
        final_boxes = []
        final_scores = []
        for i in range(num_images):
            boxes = proposals[i]
            scores = objectnesses[i]
            
            # 裁剪边界框到图像尺寸
            boxes = self.clip_boxes_to_image(boxes, image_sizes[i])
            
            # 移除小尺寸的边界框
            keep = self.remove_small_boxes(boxes, min_size=1e-3)
            boxes, scores = boxes[keep], scores[keep]
            
            # 排序并选择前pre_nms_top_n个
            scores, order = scores.sort(descending=True)
            order = order[:pre_nms_top_n]
            boxes, scores = boxes[order], scores[order]
            
            # 应用NMS
            keep = torchvision.ops.nms(boxes, scores, nms_thresh)
            keep = keep[:post_nms_top_n]
            boxes, scores = boxes[keep], scores[keep]
            
            final_boxes.append(boxes)
            final_scores.append(scores)
        
        return final_boxes
    
    def clip_boxes_to_image(self, boxes, size):
        """裁剪边界框到图像尺寸范围内"""
        height, width = size
        boxes[:, 0].clamp_(min=0, max=width)
        boxes[:, 1].clamp_(min=0, max=height)
        boxes[:, 2].clamp_(min=0, max=width)
        boxes[:, 3].clamp_(min=0, max=height)
        return boxes
    
    def remove_small_boxes(self, boxes, min_size):
        """移除宽度或高度小于min_size的边界框"""
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        return keep
    
    def roi_heads(self, features, proposals, image_shapes, targets=None):
        """RoI heads前向传播，包含box分类和回归"""
        # 提取特征
        box_features = self.roi_align(features, proposals, image_shapes)
        
        # 通过Box Head和Box Predictor
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        
        # 处理检测结果
        detections, detector_losses = self.postprocess_detections(
            class_logits, box_regression, proposals, image_shapes
        )
        
        return detections, detector_losses
    
    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        """处理检测结果，应用分类和边界框回归"""
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        
        pred_scores = F.softmax(class_logits, -1)
        
        # 分割批量
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for boxes, scores, image_shape in zip(pred_boxes, pred_scores, image_shapes):
            boxes = self.clip_boxes_to_image(boxes, image_shape)
            
            # 创建标签列表（0是背景，所以从1开始）
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            
            # 移除背景类
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            
            # 重新组织为一维张量
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            
            # 移除低置信度的检测
            keep = torch.where(scores > self.box_score_thresh)[0]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            
            # 按置信度排序
            keep = torch.argsort(scores, descending=True)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            
            # 应用NMS
            keep = torchvision.ops.batched_nms(boxes, scores, labels, self.box_nms_thresh)
            keep = keep[:self.box_detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]


if __name__ == "__main__":
    # 创建一个示例Mask R-CNN模型
    model = MaskRCNN(num_classes=2)
    print("Mask R-CNN Model:", model)

    # image = torch.randn(1, 3, 224, 224)
    # targets = [torch.tensor([[0, 0, 100, 100], [50, 50, 150, 150]])]
    # detections, losses = model(image, targets)
    # print("Detections:", detections)
    # print("Losses:", losses)