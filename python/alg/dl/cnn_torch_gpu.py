import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8 * 13 * 13, 128)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.view(-1, 8 * 13 * 13)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x

import argparse
parser = argparse.ArgumentParser(description='MNIST Dataset Path')
parser.add_argument('--data_root', type=str, default='/home/morgan/ubt/data/ml', help='Path to MNIST dataset')
args = parser.parse_args()

# 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root=args.data_root, train=True, transform=transform, download=True)
test_dataset = MNIST(root=args.data_root, train=False, transform=transform, download=True)

# 减少数据集规模
train_dataset.data = train_dataset.data[:5000]
train_dataset.targets = train_dataset.targets[:5000]
test_dataset.data = test_dataset.data[:1000]
test_dataset.targets = test_dataset.targets[:1000]

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练参数
num_epochs = 100
losses = []

# 训练循环
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
        images, labels = images.to(device), labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        losses.append(loss.item())

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss over Batches')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.show()

# 测试模型
correct_predictions = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

accuracy = correct_predictions / total
print(f'Test accuracy: {accuracy * 100:.2f}%')

# 可视化部分测试结果
images, labels = next(iter(test_loader))
images = images[:9].to(device)
labels = labels[:9].cpu().numpy()
outputs = model(images)
_, predicted = torch.max(outputs.data, 1)
predicted = predicted.cpu().numpy()

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].cpu().numpy().reshape(28, 28), cmap='gray')
    plt.title(f'Pred: {predicted[i]}, True: {labels[i]}')
    plt.axis('off')
plt.show()