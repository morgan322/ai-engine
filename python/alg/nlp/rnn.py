import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()  # space tokenizer
        input_ids = [word_dict[n] for n in word[:-1]]  # 输入词ID
        target = word_dict[word[-1]]  # 目标词ID
        
        # 转换为one-hot向量并添加到列表
        input_onehot = [np.eye(n_class)[idx] for idx in input_ids]
        input_batch.append(input_onehot)
        target_batch.append(target)
    
    # 合并为NumPy数组提高转换效率
    input_batch = np.array(input_batch, dtype=np.float32)
    target_batch = np.array(target_batch, dtype=np.int64)
    
    return input_batch, target_batch

class ManualRNN(nn.Module):
    """手动实现的单层层RNN，继承nn.Module并实现__call__"""
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(ManualRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        # 定义可训练参数
        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.weight_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bias_ih = nn.Parameter(torch.zeros(hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(hidden_size))
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, input, hx=None):
        """
        input shape: [seq_len, batch_size, input_size]
        hx shape: [1, batch_size, hidden_size]
        """
        seq_len, batch_size, _ = input.size()
        
        # 初始化隐藏状态
        if hx is None:
            hx = torch.zeros(1, batch_size, self.hidden_size, device=input.device)
        
        hx = hx.squeeze(0)  # [batch_size, hidden_size]
        output = []
        
        for t in range(seq_len):
            x_t = input[t]  # [batch_size, input_size]
            
            # 计算隐藏状态
            h_t = torch.tanh(
                F.linear(x_t, self.weight_ih, self.bias_ih) + 
                F.linear(hx, self.weight_hh, self.bias_hh)
            )
            
            # 应用Dropout
            h_t = self.dropout_layer(h_t)
            output.append(h_t)
            hx = h_t  # 更新隐藏状态
        
        # 堆叠输出并调整隐藏状态维度
        outputs = torch.stack(output, dim=0)  # [seq_len, batch_size, hidden_size]
        hx = hx.unsqueeze(0)  # [1, batch_size, hidden_size]
        
        return outputs, hx

class TextRNN(nn.Module):
    """使用手动RNN的文本分类模型"""
    def __init__(self, n_class, n_hidden, dropout=0.0):
        super(TextRNN, self).__init__()
        self.rnn = ManualRNN(input_size=n_class, hidden_size=n_hidden, dropout=dropout)
        # self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.fc = nn.Linear(n_hidden, n_class)  # 全连接层（包含偏置）
        self.b = nn.Parameter(torch.ones([n_class]))
    def forward(self, hx, X):
        """
        X: [batch_size, n_step, n_class]
        hx: [1, batch_size, hidden_size]
        """
        X = X.transpose(0, 1)  # 转换为[seq_len, batch_size, n_class]
        outputs, hx = self.rnn(X, hx)  # 直接调用ManualRNN对象
        last_hidden = outputs[-1]  # 取最后一个时间步输出
        logits = self.fc(last_hidden) + self.b # 全连接层计算logits
        return logits

if __name__ == '__main__':
    n_step = 2          # 序列长度
    n_hidden = 64       # 隐藏层维度
    dropout = 0.1       # Dropout率
    lr = 0.005          # 学习率
    weight_decay = 1e-5 # 权重衰减
    
    sentences = ["i like dog", "i love coffee", "i hate milk"]
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    n_class = len(word_dict)
    batch_size = len(sentences)

    # 创建模型
    model = TextRNN(n_class, n_hidden, dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1000)

    # 生成批次数据（已优化转换方式）
    input_batch, target_batch = make_batch()
    input_tensor = torch.from_numpy(input_batch)
    target_tensor = torch.from_numpy(target_batch)

    # 训练
    total_epochs = 10000
    for epoch in range(total_epochs):
        optimizer.zero_grad()
        hx = torch.zeros(1, batch_size, n_hidden)  # 初始隐藏状态
        logits = model(hx, input_tensor)
        
        loss = criterion(logits, target_tensor)
        scheduler.step(loss)
        
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch: {epoch+1:04d}, Loss: {loss.item():.6f}')
        
        loss.backward()
        optimizer.step()

    # 预测
    with torch.no_grad():
        hx = torch.zeros(1, batch_size, n_hidden)
        logits = model(hx, input_tensor)
        predicts = logits.argmax(dim=1)
    
    print("\n预测结果:")
    for i, sen in enumerate(sentences):
        print(f"输入: {sen.split()[:-1]}, 预测: {word_list[predicts[i]]}, 真实: {sen.split()[-1]}")
    
    # 计算准确率
    accuracy = (predicts == target_tensor).sum().item() / batch_size * 100
    print(f"准确率: {accuracy:.2f}%")