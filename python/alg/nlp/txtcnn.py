import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 检测并配置GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_classes, filter_sizes, num_filters):
        super(TextCNN, self).__init__()
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_filters_total = num_filters * len(filter_sizes)
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_size).to(device)
        
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1, 
                out_channels=num_filters, 
                kernel_size=(fs, embedding_size)
            ).to(device) 
            for fs in filter_sizes
        ])
        
        # 全连接层
        self.fc = nn.Linear(self.num_filters_total, num_classes).to(device)
    
    def forward(self, X):
        # X shape: [batch_size, sequence_length]
        batch_size, seq_len = X.size()
        
        # 嵌入层：[batch_size, seq_len, embedding_size] → [batch_size, 1, seq_len, embedding_size]
        embedded = self.embedding(X).unsqueeze(1)
        
        pooled_outputs = []
        for conv in self.convs:
            # 卷积 + ReLU
            conv_out = F.relu(conv(embedded))
            
            # 动态池化层尺寸
            pool_height = conv_out.size(2)
            mp = nn.MaxPool2d((pool_height, 1))  # 高度方向池化
            
            # 池化并调整维度：[batch_size, num_filters, 1, 1] → [batch_size, num_filters]
            pooled = mp(conv_out).squeeze(3).squeeze(2)
            pooled_outputs.append(pooled)
        
        # 拼接所有滤波器的输出
        if len(pooled_outputs) > 1:
            h_pool = torch.cat(pooled_outputs, dim=1)  # 多滤波器拼接
        else:
            h_pool = pooled_outputs[0]  # 单滤波器直接使用
        
        # 全连接层分类
        logits = self.fc(h_pool)
        return logits

if __name__ == '__main__':
    # 超参数
    embedding_size = 2
    sequence_length = 3
    num_classes = 2
    filter_sizes = [2, 3]  # 使用不同的n-gram窗口（修复原代码中重复的[2,2,2]）
    num_filters = 3
    learning_rate = 0.001
    
    # 样本数据
    sentences = [
        "i love you", "he loves me", "she likes baseball",
        "i hate you", "sorry for that", "this is awful"
    ]
    labels = [1, 1, 1, 0, 0, 0]
    
    # 构建词汇表
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)
    
    # 转换为张量并移动到GPU
    inputs = torch.LongTensor([[word_dict[n] for n in sen.split()] for sen in sentences]).to(device)
    targets = torch.LongTensor(labels).to(device)
    
    # 初始化模型、损失函数和优化器
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        num_classes=num_classes,
        filter_sizes=filter_sizes,
        num_filters=num_filters
    ).to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    num_epochs = 5000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch: {epoch+1:04d}, Loss: {loss.item():.6f}')
    
    # 测试示例
    test_text = "sorry hate you"
    test_input = torch.LongTensor([[word_dict[n] for n in test_text.split()]]).to(device)
    
    # 确保测试输入长度与训练时一致
    if test_input.size(1) != sequence_length:
        print(f"Error: Test sequence length ({test_input.size(1)}) does not match training sequence length ({sequence_length})")
    else:
        model.eval()
        with torch.no_grad():
            logits = model(test_input)
            predict = logits.argmax(dim=1).item()
        
        print(f"\nTest text: {test_text}")
        print("Prediction:", "Good Mean!!" if predict == 1 else "Bad Mean...")