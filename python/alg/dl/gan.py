import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR

# 定义生成器
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1)
        return self.gen(x)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, kernel_size=7, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

# 超参数设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0002
z_dim = 100
batch_size = 64
num_epochs = 50
beta1 = 0.5
beta2 = 0.999

# 数据加载
transform = transforms.Compose([
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='/home/morgan/ubt/data/ml', train=True,
                         transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化生成器和判别器
gen = Generator(z_dim).to(device)
disc = Discriminator().to(device)

# 定义优化器和损失函数
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, beta2))
criterion = nn.BCELoss()

# 学习率调度器
scheduler_gen = StepLR(opt_gen, step_size=10, gamma=0.1)
scheduler_disc = StepLR(opt_disc, step_size=10, gamma=0.1)

# 训练循环
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        batch_size = real.shape[0]

        ### 训练判别器
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        ### 训练生成器
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

    # 更新学习率
    scheduler_gen.step()
    scheduler_disc.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")

# 生成一些样本进行可视化
num_samples = 16
noise = torch.randn(num_samples, z_dim).to(device)
generated_images = gen(noise).cpu().detach().squeeze(1).numpy()

# 显示生成的图像
fig, axes = plt.subplots(4, 4, figsize=(4, 4))
axes = axes.flatten()
for i in range(num_samples):
    axes[i].imshow(generated_images[i], cmap='gray')
    axes[i].axis('off')
plt.show()    