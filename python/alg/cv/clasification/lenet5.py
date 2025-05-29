import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 输入形状：(N, 1, 32, 32)
        x = self.conv1(x)  # 卷积后：(N, 6, 28, 28)
        x = self.pool1(torch.relu(x))  # 池化后：(N, 6, 14, 14)
        x = self.conv2(x) # 卷积后：(N, 16, 10, 10)
        x = self.pool2(torch.relu(x))  # 池化后：(N, 16, 5, 5)
        x = x.view(-1, 16 * 5 * 5) # Flatten: (N, 400)
        x = self.fc1(x) # (N, 120)
        x = torch.relu(x)
        x = self.fc2(x)  # (N, 84)
        x = torch.relu(x)
        x = self.fc3(x)  # (N, 10)
        return x

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # CIFAR-10
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=args.data_root, train=True,
                                            download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False,
                                        download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=2)

    model = LeNet5()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    if not args.train:
        model.load_state_dict(torch.load(args.model_path))
        print('Loaded model from', args.model_path)
        for i in range(10):
            img, label = next(iter(test_loader))
            img = img[i].unsqueeze(0)
            model.eval()
            with torch.no_grad():
                output = model(img.to(device))
            pred = output.argmax(dim=1).item()
            print(f'Predicted class: {pred}, actual value: {label[i]}')
    else:
        acc = 0.0
        for epoch in range(args.epochs):
            model.train()
            for images, labels in train_loader:
                outputs = model(images.to(device))
                loss = criterion(outputs, labels.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images.to(device))
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels.to(device)).sum().item()

            accuracy = 100 * correct / len(testset)
            if accuracy > acc:
                acc = accuracy
                torch.save(model.state_dict(), args.model_path)
            print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, args.epochs, loss.item(), accuracy))

        print('finished training at accuracy:', acc)

import argparse
parser = argparse.ArgumentParser(description='Script Description')
parser.add_argument('--data_root', type=str, default='/home/morgan/ubt/data/ml', help='Path to MNIST dataset')
parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--model_path', type=str, default='/home/morgan/ubt/data/ml/weights/classification/lenet5.pth', help='Path to model path')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
args = parser.parse_args()

if __name__ == "__main__":
    main(args)
    