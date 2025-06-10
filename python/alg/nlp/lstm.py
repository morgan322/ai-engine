import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def make_batch():
    input_batch, target_batch = [], []

    for seq in seq_data:
        input_ids = [word_dict[n] for n in seq[:-1]]  # 输入序列（去掉最后一个字符）
        target = word_dict[seq[-1]]  # 目标字符（最后一个字符）
        input_onehot = [np.eye(n_class)[idx] for idx in input_ids]  # 转换为one-hot向量
        input_batch.append(input_onehot)
        target_batch.append(target)
    
    # 转换为NumPy数组提高效率
    input_batch = np.array(input_batch, dtype=np.float32)
    target_batch = np.array(target_batch, dtype=np.int64)
    
    return input_batch, target_batch

class ManualLSTM(nn.Module):
    """手动实现的单层LSTM，模拟nn.LSTM接口"""
    def __init__(self, input_size, hidden_size):
        super(ManualLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 定义LSTM各门将的权重和偏置（按input, forget, cell, output顺序）
        # 输入门 (W_ii, W_hi), 遗忘门 (W_if, W_hf), 细胞门 (W_ig, W_hg), 输出门 (W_io, W_ho)
        self.weight_ih = nn.Parameter(torch.randn(4*hidden_size, input_size) * 0.01)
        self.weight_hh = nn.Parameter(torch.randn(4*hidden_size, hidden_size) * 0.01)
        self.bias_ih = nn.Parameter(torch.zeros(4*hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(4*hidden_size))
    
    def forward(self, input, hx=None):
        """
        input shape: [seq_len, batch_size, input_size]
        hx: (h0, c0)，形状均为[1, batch_size, hidden_size]
        """
        seq_len, batch_size, _ = input.size()
        device = input.device
        
        # 初始化隐藏状态和细胞状态
        if hx is None:
            h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
            c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        else:
            h0, c0 = hx
        
        h_prev = h0.squeeze(0)  # [batch_size, hidden_size]
        c_prev = c0.squeeze(0)  # [batch_size, hidden_size]
        outputs = []
        
        for t in range(seq_len):
            x_t = input[t]  # [batch_size, input_size]
            
            # 计算门控信号（合并为一个矩阵运算）
            # 输入门控：i = sigmoid(W_ii x + W_hi h_prev + b_ii)
            # 遗忘门控：f = sigmoid(W_if x + W_hf h_prev + b_if)
            # 细胞门控：g = tanh(W_ig x + W_hg h_prev + b_ig)
            # 输出门控：o = sigmoid(W_io x + W_ho h_prev + b_io)
            gates = F.linear(x_t, self.weight_ih, self.bias_ih) + \
                    F.linear(h_prev, self.weight_hh, self.bias_hh)
            
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, dim=1)
            
            # 应用激活函数
            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            c_gate = torch.tanh(c_gate)
            o_gate = torch.sigmoid(o_gate)
            
            # 更新细胞状态和隐藏状态
            c_current = f_gate * c_prev + i_gate * c_gate
            h_current = o_gate * torch.tanh(c_current)
            
            outputs.append(h_current)
            h_prev, c_prev = h_current, c_current
        
        # 堆叠输出并调整隐藏状态维度
        outputs = torch.stack(outputs, dim=0)  # [seq_len, batch_size, hidden_size]
        hx = (h_prev.unsqueeze(0), c_prev.unsqueeze(0))  # [1, batch_size, hidden_size]
        
        return outputs, hx

class TextLSTM(nn.Module):
    """使用手动LSTM的文本分类模型"""
    def __init__(self, n_class, n_hidden):
        super(TextLSTM, self).__init__()
        self.lstm = ManualLSTM(input_size=n_class, hidden_size=n_hidden)
        # self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self, X):
        """
        X: [batch_size, seq_len, n_class]
        """
        X = X.transpose(0, 1)  # 转换为[seq_len, batch_size, n_class]
        batch_size = X.size(1)
        
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=X.device)
        c0 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=X.device)
        
        # 前向传播
        outputs, (hn, cn) = self.lstm(X, (h0, c0))
        last_hidden = outputs[-1]  # 取最后一个时间步的隐藏状态
        
        # 全连接层计算输出
        model = self.W(last_hidden) + self.b
        return model

if __name__ == '__main__':
    n_step = 3       # 序列长度（每个单词的前3个字符）
    n_hidden = 128   # 隐藏层维度
    lr = 0.001       # 学习率

    char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
    word_dict = {n: i for i, n in enumerate(char_arr)}
    number_dict = {i: w for i, w in enumerate(char_arr)}
    n_class = len(word_dict)
    seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

    model = TextLSTM(n_class, n_hidden)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 生成批次数据
    input_batch, target_batch = make_batch()
    input_tensor = torch.from_numpy(input_batch)
    target_tensor = torch.from_numpy(target_batch)

    # 训练模型
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch: {epoch+1:04d}, Loss: {loss.item():.6f}')
        
        loss.backward()
        optimizer.step()

    # 预测结果
    with torch.no_grad():
        predict = model(input_tensor).argmax(dim=1)
    
    print("\n输入序列:", [seq[:3] for seq in seq_data])
    print("预测结果:", [number_dict[n.item()] for n in predict])
    print("真实标签:", [seq[-1] for seq in seq_data])