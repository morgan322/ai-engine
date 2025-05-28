import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

'''定义参数'''
batch_size = 64
lr = 0.001
num_classes = 10

'''获取数据集（修改为MNIST）'''
import argparse
parser = argparse.ArgumentParser(description='MNIST Dataset Path')
parser.add_argument('--data_root', type=str, default='/home/morgan/ubt/data/ml', help='Path to MNIST dataset')
args = parser.parse_args()

# MNIST为单通道灰度图像，转换时需保持单通道
train_dataset = torchvision.datasets.MNIST(
    root=args.data_root,
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),  # 自动将像素值从[0,255]归一化到[0,1]
        # 可选：添加数据增强（如随机旋转、平移）
        # transforms.RandomRotation(10),
        # transforms.RandomCrop(28, padding=2),
    ])
)

test_dataset = torchvision.datasets.MNIST(
    root=args.data_root,
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

'''装载数据'''
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2  # 可选：根据硬件性能调整数据加载线程数
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

'''设计适配MNIST的LeNet-5（关键修改：输入通道=1，尺寸调整）'''
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        '''第一层卷积：输入通道1，输出通道6，卷积核5x5，步距1'''
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0) 
        
        '''第一层池化：核大小2x2，步距2'''
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        '''第二层卷积：输入通道6，输出通道16，卷积核5x5，步距1'''
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        
        '''第二层池化：核大小2x2，步距2'''
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        '''全连接层：根据MNIST尺寸计算输入维度'''
        # 计算池化后尺寸：28 -> (28-5+1)=24 -> 池化后12；12 -> (12-5+1)=8 -> 池化后4
        # 因此，Flatten尺寸为 16 * 4 * 4 = 256
        self.linear1 = nn.Linear(16 * 4 * 4, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        # 输入形状：(N, 1, 28, 28)
        out = F.sigmoid(self.conv1(x))  # 卷积后：(N, 6, 24, 24)
        out = self.pool1(out)           # 池化后：(N, 6, 12, 12)
        
        out = F.sigmoid(self.conv2(out))  # 卷积后：(N, 16, 8, 8)
        out = self.pool2(out)            # 池化后：(N, 16, 4, 4)
        
        out = out.reshape(-1, 16 * 4 * 4)  # Flatten: (N, 256)
        
        out = F.sigmoid(self.linear1(out))  # (N, 120)
        out = F.sigmoid(self.linear2(out))  # (N, 84)
        out = self.linear3(out)            # (N, 10)
        
        return out

model = LeNet(num_classes)

'''设置损失函数与优化器'''
criterion = nn.CrossEntropyLoss()  # 内部包含Softmax，无需手动添加
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

'''开始训练'''
total_step = len(train_loader)
for epoch in range(10):
    model.train()  # 开启训练模式（如启用Dropout/BatchNorm）
    for i, (images, labels) in enumerate(train_loader):
        # MNIST数据为单通道，无需修改维度，直接使用
        images = Variable(images)
        labels = Variable(labels)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/10], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

'''可选：测试集评估'''
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')