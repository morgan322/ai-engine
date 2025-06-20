# --------------------------- 5. 主函数（示例训练流程） ---------------------------
def main():
    # 设置参数
    num_classes = 80
    anchors = [[(10,13), (16,30), (33,23)],
               [(30,61), (62,45), (59,119)],
               [(116,90), (156,198), (373,326)]]
    strides = [8, 16, 32]
    img_size = 416
    batch_size = 8
    epochs = 100
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 构建模型
    model = YOLOv3(num_classes, anchors, anchor_softmax=True)
    
    # 初始化损失函数
    criterion = YOLOv3Loss(
        num_classes=num_classes,
        anchors=anchors,
        strides=strides,
        anchor_softmax=True
    )
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 准备数据集（这里需要替换为实际数据）
    # train_images, train_labels = load_coco_dataset('path/to/coco/train')
    # val_images, val_labels = load_coco_dataset('path/to/coco/val')
    
    # 为了示例，使用随机数据
    train_images = [f'dummy_{i}.jpg' for i in range(100)]
    train_labels = [torch.rand(5, 5) for _ in range(100)]  # 随机标签
    val_images = [f'dummy_val_{i}.jpg' for i in range(20)]
    val_labels = [torch.rand(5, 5) for _ in range(20)]
    
    # 创建数据加载器
    train_dataset = YOLODataset(train_images, train_labels, img_size=img_size)
    val_dataset = YOLODataset(val_images, val_labels, img_size=img_size, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 训练模型
    train_yolov3(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        device=device
    )

if __name__ == "__main__":
    main()