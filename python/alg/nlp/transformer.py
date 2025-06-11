import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 符号定义
PAD = 0  # 填充符
SOS = 5  # 解码起始符（Start of Sentence）
EOS = 6  # 解码结束符（End of Sentence）

def make_batch(sentences):
    """生成批次数据，包含源序列、目标输入和目标标签"""
    src_vocab = {'P': PAD, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    tgt_vocab = {'P': PAD, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': SOS, 'E': EOS}
    
    # 源序列：ich mochte ein bier P → [1,2,3,4,0]
    enc_inputs = [[src_vocab[word] for word in sentences[0].split()]]
    # 目标输入：S i want a beer → [5,1,2,3,4]
    dec_inputs = [[tgt_vocab[word] for word in sentences[1].split()]]
    # 目标标签：i want a beer E → [1,2,3,4,6]
    target_batch = [[tgt_vocab[word] for word in sentences[2].split()]]
    
    return (torch.LongTensor(enc_inputs), 
            torch.LongTensor(dec_inputs), 
            torch.LongTensor(target_batch))

def get_sinusoid_encoding_table(n_position, d_model):
    """生成位置编码表"""
    table = np.zeros((n_position, d_model), dtype=np.float32)
    for pos in range(n_position):
        for i in range(d_model // 2):
            angle = pos / np.power(10000, 2 * i / d_model)
            table[pos, 2*i] = np.sin(angle)
            table[pos, 2*i+1] = np.cos(angle)
    return torch.from_numpy(table)

def get_attn_pad_mask(seq_q, seq_k, pad_idx=PAD):
    """生成填充掩码"""
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_mask = seq_k.eq(pad_idx).unsqueeze(1)  # [B, 1, Lk]
    return pad_mask.expand(batch_size, len_q, len_k)  # [B, Lq, Lk]

def get_attn_subsequent_mask(seq):
    """生成后续位置掩码（防止未来信息泄露）"""
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte().to(seq.device)  # 移至与seq相同的设备
    return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.size(-1))  # [B, H, Lq, Lk]
        scores = scores.masked_fill(attn_mask, -1e9)
        attn = nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # [B, H, Lq, Dv]
        return context, attn

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, d_model, n_heads, d_k, d_v):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        
        self.W_Q = nn.Linear(d_model, n_heads * d_k)
        self.W_K = nn.Linear(d_model, n_heads * d_k)
        self.W_V = nn.Linear(d_model, n_heads * d_v)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, Q, K, V, attn_mask):
        B, Lq, D = Q.size()
        B, Lk, D = K.size()
        B, Lv, D = V.size()
        
        # 线性投影并拆分为多头
        q = self.W_Q(Q).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, Lq, Dk]
        k = self.W_K(K).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, Lk, Dk]
        v = self.W_V(V).view(B, Lv, self.n_heads, self.d_v).transpose(1, 2)  # [B, H, Lv, Dv]
        
        # 扩展掩码维度
        attn_mask = attn_mask.unsqueeze(1)  # [B, 1, Lq, Lk]
        
        # 计算注意力
        context, attn = ScaledDotProductAttention()(q, k, v, attn_mask)  # [B, H, Lq, Dv]
        context = context.transpose(1, 2).contiguous().view(B, Lq, -1)  # [B, Lq, H*Dv]
        
        # 多头合并和残差连接
        output = self.norm(Q + self.fc(context))  # [B, Lq, D]
        return output, attn

class PoswiseFeedForwardNet(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        residual = x  # [B, L, D]
        x = nn.functional.relu(self.conv1(x.transpose(1, 2))).transpose(1, 2)  # [B, L, Dff]
        x = self.norm(self.conv2(x.transpose(1, 2)).transpose(1, 2) + residual)
        return x

class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)
    
    def forward(self, x, attn_mask):
        x, attn = self.self_attn(x, x, x, attn_mask)
        x = self.ffn(x)
        return x, attn

class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)
    
    def forward(self, x, enc_outputs, self_attn_mask, cross_attn_mask):
        # 自注意力
        x, self_attn = self.self_attn(x, x, x, self_attn_mask)
        # 交叉注意力
        x, cross_attn = self.cross_attn(x, enc_outputs, enc_outputs, cross_attn_mask)
        x = self.ffn(x)
        return x, self_attn, cross_attn

class Encoder(nn.Module):
    """编码器"""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_k, d_v, d_ff, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_len, d_model),
            freeze=True
        )
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)
        ])
    
    def forward(self, x):
        B, L = x.size()
        # 位置编码（动态生成，适配输入长度）
        pos = torch.arange(L, dtype=torch.long, device=x.device).unsqueeze(0)  # [1, L]
        emb = self.embedding(x) + self.pos_emb(pos)  # [B, L, D]
        attn_mask = get_attn_pad_mask(x, x)  # [B, L, L]
        attns = []
        for layer in self.layers:
            emb, attn = layer(emb, attn_mask)
            attns.append(attn)
        return emb, attns

class Decoder(nn.Module):
    """解码器"""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_k, d_v, d_ff, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_len, d_model),
            freeze=True
        )
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)
        ])
    
    def forward(self, x, enc_outputs, enc_inputs):
        B, L = x.size()
        # 位置编码（动态生成，适配输入长度）
        pos = torch.arange(L, dtype=torch.long, device=x.device).unsqueeze(0)  # [1, L]
        emb = self.embedding(x) + self.pos_emb(pos)  # [B, L, D]
        
        # 自注意力掩码（填充 + 后续位置）
        self_attn_pad_mask = get_attn_pad_mask(x, x)  # [B, L, L]
        self_attn_sub_mask = get_attn_subsequent_mask(x)  # [1, 1, L, L]
        self_attn_mask = self_attn_pad_mask | self_attn_sub_mask  # [B, L, L]
        
        # 交叉注意力掩码（仅填充）
        cross_attn_mask = get_attn_pad_mask(x, enc_inputs)  # [B, L, L_enc]
        
        attns_self = []
        attns_cross = []
        for layer in self.layers:
            emb, self_attn, cross_attn = layer(emb, enc_outputs, self_attn_mask, cross_attn_mask)
            attns_self.append(self_attn)
            attns_cross.append(cross_attn)
        return emb, attns_self, attns_cross

class Transformer(nn.Module):
    """Transformer整体模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_k, d_v, d_ff, max_len=100):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_k, d_v, d_ff, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_k, d_v, d_ff, max_len)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
    
    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_cross_attns = self.decoder(dec_inputs, enc_outputs, enc_inputs)
        logits = self.projection(dec_outputs)  # [B, L, T_vocab]
        return logits.view(-1, logits.size(-1)), enc_attns, dec_self_attns, dec_cross_attns

def show_attention(attn, src_sent, tgt_sent, title="Attention Map"):
    """可视化注意力矩阵"""
    B, H, Lq, Lk = attn.size()
    attn = attn[0, 0].detach().cpu().numpy()  # 取第一个批次、第一个头
    
    fig, ax = plt.subplots(figsize=(Lk, Lq))
    ax.imshow(attn, cmap='viridis', aspect='auto')
    
    # 设置轴标签
    ax.set_xticks(np.arange(Lk))
    ax.set_xticklabels(src_sent.split(), rotation=90, ha='right')
    ax.set_yticks(np.arange(Lq))
    ax.set_yticklabels(tgt_sent.split(), va='top')
    
    plt.title(title)
    plt.xlabel("Source")
    plt.ylabel("Target")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    sentences = [
        "ich mochte ein bier P",   # 源序列（德语）
        "S i want a beer",         # 目标输入（英语，以SOS开头）
        "i want a beer E"          # 目标标签（英语，以EOS结尾）
    ]
    
    # 超参数
    d_model = 512       # 嵌入维度
    d_ff = 2048        # 前馈网络维度
    d_k = d_v = 64     # 键/值维度
    n_layers = 2       # 编码器/解码器层数
    n_heads = 8        # 头数
    lr = 0.001         # 学习率
    epochs = 100       # 训练轮次
    
    # 生成批次数据
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_inputs = enc_inputs.to(device)
    dec_inputs = dec_inputs.to(device)
    target_batch = target_batch.to(device)

    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)
    
    # 初始化模型
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        d_ff=d_ff
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)  # 忽略填充符的损失
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        outputs, _, _, _ = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # 预测
    model.eval()
    with torch.no_grad():
        logits, _, _, _ = model(enc_inputs, dec_inputs)
        predict = logits.argmax(dim=-1).view(dec_inputs.size())
    
    # 转换为单词
    tgt_vocab_inv = {v: k for k, v in tgt_vocab.items()}
    predicted_sent = " ".join([tgt_vocab_inv[idx.item()] for idx in predict.squeeze()])
    print(f"预测结果: {predicted_sent}")
    
    # 可视化注意力（以最后一层、第一个头为例）
    _, enc_attns, dec_self_attns, dec_cross_attns = model(enc_inputs, dec_inputs)
    
    print("\n编码器自注意力可视化:")
    show_attention(enc_attns[-1], sentences[0], sentences[0], "Encoder Self-Attention")
    
    print("\n解码器自注意力可视化:")
    show_attention(dec_self_attns[-1], sentences[1], sentences[1], "Decoder Self-Attention")
    
    print("\n解码器-编码器交叉注意力可视化:")
    show_attention(dec_cross_attns[-1], sentences[0], sentences[1], "Decoder-Encoder Cross-Attention")