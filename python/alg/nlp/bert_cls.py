import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
import re

# 固定随机种子
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# -------------------------- 数据加载 --------------------------
class MRPCDataset(Dataset):
    def __init__(self, file_path, word_dict, maxlen, is_test=False):
        self.sentence1 = []
        self.sentence2 = []
        self.labels = []
        self.word_dict = word_dict
        self.maxlen = maxlen
        self.is_test = is_test
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # 跳过表头
            for row in reader:
                if len(row) < 5:
                    continue
                # 处理标签（测试集可能无标签或标签格式不同）
                if not self.is_test:
                    try:
                        label = int(row[0])
                        if label not in {0, 1}:
                            continue  # 过滤无效标签
                        self.labels.append(label)
                    except:
                        continue
                else:
                    self.labels.append(0)  # 测试集用0占位（不用于计算损失）
                
                # 文本清洗与分词
                s1 = self.clean_text(row[3])
                s2 = self.clean_text(row[4])
                self.sentence1.append(s1)
                self.sentence2.append(s2)
    
    def clean_text(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        s1 = self.sentence1[idx]
        s2 = self.sentence2[idx]
        label = self.labels[idx]
        
        # 转换为token ID
        tokens1 = [self.word_dict.get(w, self.word_dict['[UNK]']) for w in s1]
        tokens2 = [self.word_dict.get(w, self.word_dict['[UNK]']) for w in s2]
        
        # 构造输入序列
        input_ids = [self.word_dict['[CLS]']] + tokens1 + [self.word_dict['[SEP]']] + tokens2 + [self.word_dict['[SEP]']]
        segment_ids = [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1)
        
        # 截断或填充
        if len(input_ids) > self.maxlen:
            input_ids = input_ids[:self.maxlen]
            segment_ids = segment_ids[:self.maxlen]
        else:
            pad_len = self.maxlen - len(input_ids)
            input_ids += [self.word_dict['[PAD]']] * pad_len
            segment_ids += [0] * pad_len
        
        return {
            'input_ids': torch.LongTensor(input_ids),
            'segment_ids': torch.LongTensor(segment_ids),
            'label': torch.LongTensor([label])
        }

# -------------------------- 模型结构 --------------------------
def get_attn_pad_mask(seq_q, seq_k):
    pad_mask = seq_k.eq(0).unsqueeze(1)  # [B, 1, Lk]
    return pad_mask.expand(-1, seq_q.size(1), -1)  # [B, Lq, Lk]

class Embedding(nn.Module):
    def __init__(self, vocab_size, maxlen, d_model, n_segments):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(maxlen, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, seg):
        B, L = x.size()
        pos = torch.arange(L, dtype=torch.long, device=x.device).unsqueeze(0).repeat(B, 1)
        emb = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(emb)

class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.functional.softmax(scores, dim=-1)
        return torch.matmul(attn, V), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, n_heads * d_k)
        self.W_K = nn.Linear(d_model, n_heads * d_k)
        self.W_V = nn.Linear(d_model, n_heads * d_v)
        self.out_proj = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, Q, K, V, attn_mask):
        B, Lq, D = Q.size()
        B, Lk, D = K.size()
        
        q = self.W_Q(Q).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(K).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(V).view(B, Lk, self.n_heads, self.d_v).transpose(1, 2)
        
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(q, k, v, attn_mask)
        
        context = context.transpose(1, 2).contiguous().view(B, Lq, -1)
        output = self.norm(Q + self.out_proj(context))
        return output, attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, attn_mask):
        attn_output, _ = self.self_attn(x, x, x, attn_mask)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class BERTClassifier(nn.Module):
    def __init__(self, vocab_size, maxlen, d_model, n_layers, n_heads, d_k, d_v, d_ff, n_segments, num_classes=2):
        super().__init__()
        self.embedding = Embedding(vocab_size, maxlen, d_model, n_segments)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)
        ])
        self.cls_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.LayerNorm(d_model)
        )
        self.dropout = nn.Dropout(0.1)  # 添加dropout防止过拟合
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, input_ids, segment_ids):
        x = self.embedding(input_ids, segment_ids)
        attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.enc_layers:
            x = layer(x, attn_mask)
        
        cls_feat = self.cls_fc(x[:, 0])
        cls_feat = self.dropout(cls_feat)  # 应用dropout
        logits = self.classifier(cls_feat)
        return logits

# -------------------------- 训练与评估 --------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        labels = batch['label'].squeeze().to(device)
        
        logits = model(input_ids, segment_ids)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * input_ids.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += input_ids.size(0)
    
    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc

def evaluate(model, dataloader, criterion, device, is_test=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            logits = model(input_ids, segment_ids)
            
            # 测试集可能无真实标签，跳过损失计算
            if not is_test:
                loss = criterion(logits, labels)
                total_loss += loss.item() * input_ids.size(0)
            
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += input_ids.size(0)
    
    avg_loss = total_loss / total_samples if not is_test else 0.0
    acc = total_correct / total_samples
    return avg_loss, acc

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 超参数（降低复杂度以减少过拟合）
    maxlen = 128
    batch_size = 64  # 增大批次大小
    n_layers = 2  # 减少层数
    n_heads = 4  # 减少注意力头数
    d_model = 128  # 减小隐藏层维度
    d_ff = d_model * 4
    d_k = d_v = 32
    n_segments = 2
    epochs = 10  # 减少训练轮次
    lr = 5e-6  # 降低学习率
    weight_decay = 1e-5  # 添加权重衰减
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # -------------------------- 构建词表 --------------------------
    train_path = '/media/ai/AI/package/data/ml/glue_data/MRPC/train.tsv'
    all_words = set()
    
    with open(train_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) < 5:
                continue
            s1 = re.sub(r'[^\w\s]', '', row[3].lower()).split()
            s2 = re.sub(r'[^\w\s]', '', row[4].lower()).split()
            all_words.update(s1)
            all_words.update(s2)
    
    # 构建词表
    special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[UNK]']
    word_list = special_tokens + list(all_words)
    word_dict = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)
    print(f"词表大小: {vocab_size}")
    
    # -------------------------- 加载数据 --------------------------
    train_dataset = MRPCDataset(
        file_path=train_path,
        word_dict=word_dict,
        maxlen=maxlen,
        is_test=False
    )
    dev_dataset = MRPCDataset(
        file_path='/media/ai/AI/package/data/ml/glue_data/MRPC/dev.tsv',
        word_dict=word_dict,
        maxlen=maxlen,
        is_test=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    
    # -------------------------- 初始化模型 --------------------------
    model = BERTClassifier(
        vocab_size=vocab_size,
        maxlen=maxlen,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        d_ff=d_ff,
        n_segments=n_segments
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay  # 添加L2正则化
    )
    
    # -------------------------- 训练循环（含早停） --------------------------
    best_dev_acc = 0.0
    patience = 3
    no_improve_epochs = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device)
        
        print(f"Epoch {epoch+1:02d}/{epochs}")
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"Dev:   Loss={dev_loss:.4f}, Acc={dev_acc:.4f}")
        
        # 早停逻辑
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), 'mrpc_best_model.pt')
            no_improve_epochs = 0
            print(f"保存最佳模型 (Dev Acc: {best_dev_acc:.4f})")
        else:
            no_improve_epochs += 1
            print(f"验证准确率未提升，计数: {no_improve_epochs}/{patience}")
            if no_improve_epochs >= patience:
                print(f"早停于第{epoch+1}轮")
                break
        
        print("-" * 50)
    
    # -------------------------- 测试 --------------------------
    print(f"最佳验证准确率: {best_dev_acc:.4f}")
    model.load_state_dict(torch.load('mrpc_best_model.pt'))
    
    # 加载测试集（使用is_test=True忽略标签问题）
    test_dataset = MRPCDataset(
        file_path='/media/ai/AI/package/data/ml/glue_data/MRPC/test.tsv',
        word_dict=word_dict,
        maxlen=maxlen,
        is_test=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 测试集评估（仅计算准确率，不计算损失）
    _, test_acc = evaluate(model, test_loader, criterion, device, is_test=True)
    print(f"测试集准确率: {test_acc:.4f}")
    
    # # 保存预测结果
    # def predict(model, dataloader, device):
    #     model.eval()
    #     all_preds = []
    #     with torch.no_grad():
    #         for batch in dataloader:
    #             input_ids = batch['input_ids'].to(device)
    #             segment_ids = batch['segment_ids'].to(device)
    #             logits = model(input_ids, segment_ids)
    #             preds = logits.argmax(dim=1).cpu().tolist()
    #             all_preds.extend(preds)
    #     return all_preds
    
    # predictions = predict(model, test_loader, device)
    # with open('mrpc_predictions.txt', 'w') as f:
    #     f.write('index\tprediction\n')
    #     for i, pred in enumerate(predictions):
    #         f.write(f'{i}\t{pred}\n')
    # print(f"预测结果已保存至 mrpc_predictions.txt")