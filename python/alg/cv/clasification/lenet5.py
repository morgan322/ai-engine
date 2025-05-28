import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
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




import argparse
parser = argparse.ArgumentParser(description='Dataset Path')
parser.add_argument('--data_root', type=str, default='/home/morgan/ubt/data/ml', help='Path to MNIST dataset')
args = parser.parse_args()


if __name__ == "__main__":

    train_dataset = torchvision.datasets.MNIST(
        root=args.data_root,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),  # 自动将像素值从[0,255]归一化到[0,1]
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


    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    batch_size = 64
    lr = 0.001
    num_classes = 10

    model = LeNet(num_classes)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_step = len(train_loader)
    for epoch in range(10):
        model.train()  
        for i, (images, labels) in enumerate(train_loader):
    
            images = Variable(images)
            labels = Variable(labels)
        
            outputs = model(images)
            loss = criterion(outputs, labels)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/10], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

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