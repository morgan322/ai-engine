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
        out = F.sigmoid(self.conv1(x))  # input: (N, 1, 28, 28) out:(N, 6, 24, 24)
        out = self.pool1(out)           # out:(N, 6, 12, 12)
        out = F.sigmoid(self.conv2(out))  # out:(N, 16, 8, 8)
        out = self.pool2(out)            # out:(N, 16, 4, 4)
        out = out.reshape(-1, 16 * 4 * 4)  # Flatten: (N, 256)
        out = F.sigmoid(self.linear1(out))  # (N, 120)
        out = F.sigmoid(self.linear2(out))  # (N, 84)
        out = self.linear3(out)            # (N, 10)
        
        return out

def main(args):

    train_dataset = torchvision.datasets.MNIST(
        root=args.data_root,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = LeNet()
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.train:
        acc = 0
        for epoch in range(args.epochs):
            model.train()  
            for i, (images, labels) in enumerate(train_loader):
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (i + 1) % 100 == 0:
                    accuracy = 100 * (outputs.argmax(dim=1) == labels).sum().item() / labels.size(0)
                    print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f},Accuracy: {accuracy:.2f}%')
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            if accuracy > acc:
                acc = accuracy
                print(f'Test Accuracy: {100 * correct / total:.2f}%')
                torch.save(model.state_dict(), args.model_path)
    else:
        print('Loaded model from', args.model_path)
        model.load_state_dict(torch.load(args.model_path))
        for i in range(10):
            img, label = next(iter(test_loader))
            img = img[i].unsqueeze(0)
            model.eval()
            with torch.no_grad():
                output = model(img)
            pred = output.argmax(dim=1).item()
            print(f'Predicted class: {pred}, actual value: {label[i]}')

import argparse
parser = argparse.ArgumentParser(description='Script Description')
parser.add_argument('--data_root', type=str, default='/home/morgan/ubt/data/ml', help='Path to MNIST dataset')
parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--model_path', type=str, default='/home/morgan/ubt/data/ml/weights/classification/lenet.pth', help='Path to model path')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
args = parser.parse_args()

if __name__ == "__main__":
    main(args)
   