import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 图像像素值在[0,1]之间
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # 编码
        h = self.encoder(x.view(-1, 784))
        mu, logvar = self.mu(h), self.logvar(h)
        
        # 重参数化采样
        z = self.reparameterize(mu, logvar)
        
        # 解码
        return self.decoder(z), mu, logvar

# 训练函数
def train(epoch, model, train_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data.view(-1, 784), mu, logvar)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch: {epoch}, Average loss: {train_loss / len(train_loader.dataset):.4f}')

# 损失函数
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.BCELoss(reduction='sum')(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


def calculate_reconstruction_accuracy(model, data_loader, device):
    model.eval()
    total_mse = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            batch_size = data.size(0)
            
            # 获取重构结果
            recon_batch, _, _ = model(data)
            
            # 计算均方误差(MSE)
            mse = nn.MSELoss(reduction='sum')(recon_batch, data.view(-1, 784))
            total_mse += mse.item()
            total_samples += batch_size
    
    # 返回平均重构误差
    return total_mse / total_samples


# 初始化模型和训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 数据加载
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('/media/ai/AI/package/data/ml', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 训练模型
for epoch in range(1, 100):
    train(epoch, model, train_loader, optimizer)

     # 每10个epoch评估一次
    if epoch % 10 == 0:
        recon_error = calculate_reconstruction_accuracy(model, train_loader, device)
        print(f"Epoch {epoch}, 重构误差: {recon_error:.6f}")
        
        # # 保存模型
        # torch.save(model.state_dict(), f'vae_epoch_{epoch}.pth')

# 生成样本示例
with torch.no_grad():
    z = torch.randn(16, 20).to(device)
    sample = model.decoder(z).cpu()

# 可视化生成的样本
plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(sample[i].view(28, 28), cmap='gray')
    plt.axis('off')
plt.show()