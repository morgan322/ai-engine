import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class ManualGRU(nn.Module):
    """手动实现的GRU单元"""
    def __init__(self, input_size, hidden_size):
        super(ManualGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 重置门参数
        self.W_r = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.U_r = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        
        # 更新门参数
        self.W_z = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.U_z = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        
        # 候选隐藏状态参数
        self.W_h = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.U_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, input, hx=None):
        """
        input shape: [seq_len, batch_size, input_size]
        hx shape: [batch_size, hidden_size]
        """
        seq_len, batch_size, _ = input.size()
        device = input.device
        
        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=device)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = input[t]
            
            # 计算重置门
            r_t = torch.sigmoid(F.linear(x_t, self.W_r, self.b_r) + 
                              F.linear(hx, self.U_r, self.b_r))
            
            # 计算更新门
            z_t = torch.sigmoid(F.linear(x_t, self.W_z, self.b_z) + 
                              F.linear(hx, self.U_z, self.b_z))
            
            # 计算候选隐藏状态
            h_tilde = torch.tanh(F.linear(x_t, self.W_h, self.b_h) + 
                              F.linear(r_t * hx, self.U_h, self.b_h))
            
            # 更新隐藏状态
            hx = (1 - z_t) * h_tilde + z_t * hx
            outputs.append(hx)
        
        return torch.stack(outputs, dim=0), hx

class CharGRU(nn.Module):
    """基于GRU的字符级语言模型"""
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super(CharGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = ManualGRU(embed_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hx=None):
        """
        x shape: [batch_size, seq_len]
        """
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        
        outputs, hx = self.gru(x, hx)  # outputs: [seq_len, batch_size, hidden_size]
        
        # 全连接层预测下一个字符
        logits = self.fc(outputs)  # [seq_len, batch_size, vocab_size]
        logits = logits.transpose(0, 1)  # [batch_size, seq_len, vocab_size]
        
        return logits, hx
    
    def generate(self, seed, length, char_to_idx, idx_to_char, device='cpu'):
        """生成文本序列"""
        self.eval()
        
        # 初始化隐藏状态
        hx = None
        
        # 处理种子文本
        generated = list(seed)
        input_idx = torch.tensor([char_to_idx[c] for c in seed], device=device).unsqueeze(0)
        
        # 基于种子文本生成隐藏状态
        with torch.no_grad():
            _, hx = self(input_idx, hx)
        
        # 从最后一个字符继续生成
        current_char = seed[-1]
        current_idx = char_to_idx[current_char]
        
        for i in range(length):
            # 输入当前字符索引
            input_tensor = torch.tensor([[current_idx]], device=device)
            
            # 预测下一个字符
            with torch.no_grad():
                logits, hx = self(input_tensor, hx)
            
            # 获取预测的字符分布（使用softmax转换为概率）
            probs = torch.softmax(logits[0, 0], dim=0)
            
            # 采样下一个字符（使用温度参数控制随机性）
            temperature = 0.8
            probs = probs / temperature
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = idx_to_char[next_idx]
            
            # 添加到生成的文本中
            generated.append(next_char)
            current_idx = next_idx
        
        return ''.join(generated)

def train_model():
    # 示例数据：莎士比亚十四行诗片段
    text = """
    Shall I compare thee to a summer's day?
    Thou art more lovely and more temperate:
    Rough winds do shake the darling buds of May,
    And summer's lease hath all too short a date:
    """
    
    # 构建字符词典
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)
    
    # 超参数
    embed_dim = 50
    hidden_size = 128
    seq_len = 40
    batch_size = 32
    learning_rate = 0.001
    epochs = 100
    
    # 创建训练数据
    data = []
    targets = []
    for i in range(0, len(text) - seq_len):
        data.append([char_to_idx[c] for c in text[i:i+seq_len]])
        targets.append([char_to_idx[c] for c in text[i+1:i+seq_len+1]])
    
    # 转换为张量
    data = torch.tensor(data, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    
    # 创建模型
    model = CharGRU(vocab_size, embed_dim, hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        
        # 按批次训练
        for i in range(0, len(data), batch_size):
            batch_data = data[i:min(i+batch_size, len(data))]
            batch_targets = targets[i:min(i+batch_size, len(data))]
            
            optimizer.zero_grad()
            
            # 前向传播
            logits, _ = model(batch_data)
            
            # 计算损失（展平后计算）
            loss = criterion(
                logits.reshape(-1, vocab_size),
                batch_targets.reshape(-1)
            )
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 记录平均损失
        avg_loss = total_loss / (len(data) // batch_size + 1)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            
            # 生成一些文本示例
            seed = "Shall I"
            generated = model.generate(seed, 50, char_to_idx, idx_to_char)
            print(f"Generated: {generated}\n")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    return model, char_to_idx, idx_to_char

# 运行训练并生成文本
if __name__ == "__main__":
    model, char_to_idx, idx_to_char = train_model()
    
    # 最终生成示例
    seed = "Shall I"
    print("\n=== Final Generation ===")
    print(model.generate(seed, 200, char_to_idx, idx_to_char, device='cpu'))