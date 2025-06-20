import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.Dataset
# --------------------------- 2. 损失函数定义 ---------------------------
class YOLOv3Loss(nn.Module):
    def __init__(self, num_classes, anchors, strides, anchor_softmax=False, 
                 obj_scale=1.0, noobj_scale=100.0, cls_scale=1.0, box_scale=5.0):
        super(YOLOv3Loss, self).__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.anchor_softmax = anchor_softmax
        
        # 损失权重
        self.obj_scale = obj_scale      # 有目标的置信度权重
        self.noobj_scale = noobj_scale  # 无目标的置信度权重
        self.cls_scale = cls_scale      # 分类损失权重
        self.box_scale = box_scale      # 边界框回归损失权重
        
        # 损失函数
        self.mse_loss = nn.MSELoss(reduction='none')  # 回归损失
        self.bce_loss = nn.BCELoss(reduction='none')  # 二分类损失（置信度、类别）
        
        # 锚框处理
        self.anchors = torch.tensor(anchors).float()
        self.num_anchors = len(anchors[0])
        
    def forward(self, predictions, targets):
        """
        计算YOLOv3总损失
        参数:
            predictions: 模型预测输出 [batch_size, num_priors, 5+num_classes]
            targets: 真实标签 [batch_size, max_objects, 6] (x, y, w, h, conf, cls)
        返回:
            总损失和各部分损失的元组
        """
        device = predictions.device
        batch_size = predictions.size(0)
        
        # 初始化损失
        total_loss = 0
        box_loss = 0
        obj_loss = 0
        noobj_loss = 0
        cls_loss = 0
        
        # 对每个样本单独计算损失
        for b in range(batch_size):
            # 获取当前样本的预测和目标
            pred = predictions[b]  # [num_priors, 5+num_classes]
            target = targets[b]    # [max_objects, 6]
            
            # 过滤掉填充的目标 (conf=0)
            valid_targets = target[target[:, 4] > 0]
            if len(valid_targets) == 0:
                # 没有目标，只计算noobj损失
                noobj_mask = torch.ones_like(pred[:, 4])
                noobj_loss += self.noobj_scale * self.bce_loss(
                    pred[:, 4].sigmoid(), noobj_mask
                ).sum()
                continue
            
            # 1. 计算锚框与目标的匹配
            anchors = self.anchors.to(device)
            anchor_boxes = torch.zeros_like(anchors)
            anchor_boxes[..., 2:] = anchors
            
            # 计算目标与锚框的IoU
            target_boxes = valid_targets[:, :4].clone()
            target_boxes[:, 2:] = target_boxes[:, 2:] * 1000  # 放大以匹配锚框尺寸
            
            # 计算IoU矩阵 [num_anchors, num_targets]
            iou_matrix = torch.zeros((self.num_anchors, len(valid_targets)), device=device)
            for a in range(self.num_anchors):
                anchor_area = anchor_boxes[a, 2] * anchor_boxes[a, 3]
                for t in range(len(valid_targets)):
                    target_area = target_boxes[t, 2] * target_boxes[t, 3]
                    
                    # 计算交集
                    w_inter = torch.min(anchor_boxes[a, 2], target_boxes[t, 2])
                    h_inter = torch.min(anchor_boxes[a, 3], target_boxes[t, 3])
                    area_inter = w_inter * h_inter
                    
                    # 计算并集
                    area_union = anchor_area + target_area - area_inter
                    
                    # 计算IoU
                    iou_matrix[a, t] = area_inter / (area_union + 1e-16)
            
            # 为每个目标分配最佳锚框
            best_ious, best_anchor_indices = iou_matrix.max(0)
            
            # 2. 构建掩码和目标张量
            num_priors = pred.size(0)
            obj_mask = torch.zeros(num_priors, device=device)
            noobj_mask = torch.ones(num_priors, device=device)
            tx = torch.zeros(num_priors, device=device)
            ty = torch.zeros(num_priors, device=device)
            tw = torch.zeros(num_priors, device=device)
            th = torch.zeros(num_priors, device=device)
            tcls = torch.zeros((num_priors, self.num_classes), device=device)
            
            # 为每个目标分配到对应的预测位置
            for t in range(len(valid_targets)):
                target_box = valid_targets[t, :4]  # x, y, w, h (归一化)
                target_cls = int(valid_targets[t, 5])
                
                # 确定目标所在的网格
                grid_x = int(target_box[0] * 13)  # 假设最大特征图尺寸为13x13
                grid_y = int(target_box[1] * 13)
                
                # 确定对应的锚框
                anchor_idx = best_anchor_indices[t]
                
                # 计算对应的先验框索引
                prior_idx = grid_y * 13 * self.num_anchors + grid_x * self.num_anchors + anchor_idx
                
                # 设置掩码
                obj_mask[prior_idx] = 1
                noobj_mask[prior_idx] = 0
                
                # 设置目标值
                tx[prior_idx] = target_box[0] * 13 - grid_x
                ty[prior_idx] = target_box[1] * 13 - grid_y
                tw[prior_idx] = torch.log(target_box[2] * 13 / self.anchors[anchor_idx // 3, anchor_idx % 3, 0] + 1e-16)
                th[prior_idx] = torch.log(target_box[3] * 13 / self.anchors[anchor_idx // 3, anchor_idx % 3, 1] + 1e-16)
                tcls[prior_idx, target_cls] = 1
            
            # 3. 计算损失
            
            # 边界框回归损失
            pred_boxes = pred[:, :4]
            tx = tx[obj_mask > 0]
            ty = ty[obj_mask > 0]
            tw = tw[obj_mask > 0]
            th = th[obj_mask > 0]
            
            if len(tx) > 0:
                px = pred_boxes[obj_mask > 0, 0]
                py = pred_boxes[obj_mask > 0, 1]
                pw = pred_boxes[obj_mask > 0, 2]
                ph = pred_boxes[obj_mask > 0, 3]
                
                box_loss += self.box_scale * (
                    self.mse_loss(px, tx) + 
                    self.mse_loss(py, ty) + 
                    self.mse_loss(pw, tw) + 
                    self.mse_loss(ph, th)
                ).sum()
            
            # 置信度损失
            pred_conf = pred[:, 4].sigmoid()
            obj_loss += self.obj_scale * self.bce_loss(
                pred_conf[obj_mask > 0], obj_mask[obj_mask > 0]
            ).sum()
            
            noobj_loss += self.noobj_scale * self.bce_loss(
                pred_conf[noobj_mask > 0], obj_mask[noobj_mask > 0]
            ).sum()
            
            # 分类损失
            pred_cls = pred[:, 5:][obj_mask > 0]
            tcls = tcls[obj_mask > 0]
            
            if len(pred_cls) > 0:
                cls_loss += self.cls_scale * self.bce_loss(
                    pred_cls.sigmoid(), tcls
                ).sum()
        
        # 平均损失
        batch_size = max(1, batch_size)
        box_loss = box_loss / batch_size
        obj_loss = obj_loss / batch_size
        noobj_loss = noobj_loss / batch_size
        cls_loss = cls_loss / batch_size
        
        # 总损失
        total_loss = box_loss + obj_loss + noobj_loss + cls_loss
        
        return total_loss, box_loss, obj_loss, noobj_loss, cls_loss

# --------------------------- 3. 数据集和数据加载器 ---------------------------
class YOLODataset(Dataset):
    def __init__(self, images, labels, img_size=416, augment=True):
        self.images = images  # 图像路径列表
        self.labels = labels  # 标签列表 (每个标签是一个numpy数组: [x, y, w, h, cls])
        self.img_size = img_size
        self.augment = augment
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 加载标签
        label = self.labels[idx].copy()  # [x, y, w, h, cls]
        
        # 图像增强 (简化版)
        if self.augment:
            # 随机缩放
            scale = np.random.uniform(0.8, 1.2)
            img = cv2.resize(img, (int(self.img_size * scale), int(self.img_size * scale)))
            
            # 随机裁剪
            h, w = img.shape[:2]
            max_dx = w - self.img_size
            max_dy = h - self.img_size
            dx = np.random.randint(0, max_dx) if max_dx > 0 else 0
            dy = np.random.randint(0, max_dy) if max_dy > 0 else 0
            img = img[dy:dy+self.img_size, dx:dx+self.img_size]
            
            # 调整标签坐标
            if len(label) > 0:
                label[:, 0] = (label[:, 0] * w - dx) / self.img_size
                label[:, 1] = (label[:, 1] * h - dy) / self.img_size
                label[:, 2] *= w / self.img_size
                label[:, 3] *= h / self.img_size
        
        # 调整图像大小
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # 归一化
        img = img / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        
        # 转换为张量
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        return img, label

# --------------------------- 4. 训练函数 ---------------------------
def train_yolov3(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    """训练YOLOv3模型"""
    model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_box_loss = 0
        train_obj_loss = 0
        train_noobj_loss = 0
        train_cls_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(images)
            
            # 计算损失
            loss, box_loss, obj_loss, noobj_loss, cls_loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录损失
            train_loss += loss.item()
            train_box_loss += box_loss.item()
            train_obj_loss += obj_loss.item()
            train_noobj_loss += noobj_loss.item()
            train_cls_loss += cls_loss.item()
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, Box: {box_loss.item():.4f}, "
                      f"Obj: {obj_loss.item():.4f}, NoObj: {noobj_loss.item():.4f}, "
                      f"Cls: {cls_loss.item():.4f}")
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        train_box_loss /= len(train_loader)
        train_obj_loss /= len(train_loader)
        train_noobj_loss /= len(train_loader)
        train_cls_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss, _, _, _, _ = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # 打印本轮训练结果
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'yolov3_best.pth')
            print(f"Model saved at epoch {epoch+1} with val loss: {val_loss:.4f}")
    
    print("Training completed!")
