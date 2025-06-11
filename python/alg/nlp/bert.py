import math
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

random.seed(42)
torch.manual_seed(42)

def make_batch():
    batch = []
    positive = 0
    negative = 0
    batch_size_half = batch_size // 2
    
    while positive < batch_size_half or negative < batch_size_half:
        tokens_a_idx, tokens_b_idx = random.randrange(len(sentences)), random.randrange(len(sentences))
        tokens_a, tokens_b = token_list[tokens_a_idx], token_list[tokens_b_idx]
        is_next = tokens_a_idx + 1 == tokens_b_idx and positive < batch_size_half
        
        # 构造输入序列
        input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        
        # 处理掩码LM
        cand_pos = [i for i, token in enumerate(input_ids) if token not in {word_dict['[CLS]'], word_dict['[SEP]']}]
        n_pred = min(max_pred, len(cand_pos))
        masked_pos = random.sample(cand_pos, n_pred)
        masked_tokens = [input_ids[pos] for pos in masked_pos]
        
        for pos in masked_pos:
            if random.random() < 0.8:
                input_ids[pos] = word_dict['[MASK]']
            elif random.random() < 0.5:
                input_ids[pos] = random.randint(4, vocab_size - 1)  # 随机非特殊token
        
        # 填充到maxlen
        pad_len = maxlen - len(input_ids)
        input_ids += [0] * pad_len
        segment_ids += [0] * pad_len
        
        # 填充masked_tokens到max_pred
        masked_tokens += [0] * (max_pred - n_pred)
        masked_pos += [0] * (max_pred - n_pred)
        
        # 添加到批次
        if is_next and positive < batch_size_half:
            batch.append((input_ids, segment_ids, masked_tokens, masked_pos, 1))
            positive += 1
        elif not is_next and negative < batch_size_half:
            batch.append((input_ids, segment_ids, masked_tokens, masked_pos, 0))
            negative += 1
    
    return batch

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
        
        q = self.W_Q(Q).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, Lq, Dk]
        k = self.W_K(K).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, Lk, Dk]
        v = self.W_V(V).view(B, Lk, self.n_heads, self.d_v).transpose(1, 2)  # [B, H, Lk, Dv]
        
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # [B, H, Lq, Lk]
        context, attn = ScaledDotProductAttention()(q, k, v, attn_mask)
        
        context = context.transpose(1, 2).contiguous().view(B, Lq, -1)  # [B, Lq, H*Dv]
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

class BERT(nn.Module):
    def __init__(self, vocab_size, maxlen, d_model, n_layers, n_heads, d_k, d_v, d_ff, n_segments):
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
        self.classifier = nn.Linear(d_model, 2)
        self.mlm_decoder = nn.Linear(d_model, vocab_size)
        self.mlm_decoder.weight = self.embedding.tok_embed.weight  # 共享权重
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))
    
    def forward(self, input_ids, segment_ids, masked_pos):
        # 嵌入层
        x = self.embedding(input_ids, segment_ids)  # [B, L, D]
        
        # 编码器
        attn_mask = get_attn_pad_mask(input_ids, input_ids)  # [B, L, L]
        for layer in self.enc_layers:
            x = layer(x, attn_mask)
        
        # NSP任务：CLS标记
        cls_feat = self.cls_fc(x[:, 0])  # [B, D]
        logits_clsf = self.classifier(cls_feat)  # [B, 2]
        
        # MLM任务：掩码位置
        masked_pos = masked_pos.unsqueeze(-1).expand(-1, -1, x.size(-1))  # [B, max_pred, D]
        mlm_feat = torch.gather(x, 1, masked_pos)  # [B, max_pred, D]
        logits_lm = self.mlm_decoder(mlm_feat) + self.mlm_bias  # [B, max_pred, V]
        
        return logits_lm, logits_clsf

if __name__ == "__main__":
    # 超参数
    maxlen = 30
    batch_size = 6
    max_pred = 5
    n_layers = 2  # 简化为2层便于测试
    n_heads = 12
    d_model = 768
    d_ff = 768 * 4
    d_k = d_v = 64
    n_segments = 2

    # 数据预处理
    text = (
        'Hello, how are you? I am Romeo.\n'
        'Hello, Romeo My name is Juliet. Nice to meet you.\n'
        'Nice meet you too. How are you today?\n'
        'Great. My baseball team won the competition.\n'
        'Oh Congratulations, Juliet\n'
        'Thanks you Romeo'
    )
    sentences = re.sub(r"[.,!?\-]", "", text.lower()).split('\n')
    word_list = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"] + list(set(" ".join(sentences).split()))
    word_dict = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)
    number_dict = {i: w for i, w in enumerate(word_dict)}
    token_list = [[word_dict[w] for w in s.split()] for s in sentences]

    # 初始化模型
    model = BERT(
        vocab_size=vocab_size,
        maxlen=maxlen,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        d_ff=d_ff,
        n_segments=n_segments
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    for epoch in range(100):
        batch = make_batch()
        input_ids, segment_ids, masked_tokens, masked_pos, is_next = zip(*batch)
        input_ids = torch.LongTensor(input_ids)
        segment_ids = torch.LongTensor(segment_ids)
        masked_tokens = torch.LongTensor(masked_tokens)
        masked_pos = torch.LongTensor(masked_pos)
        is_next = torch.LongTensor(is_next)
        
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
        
        # 计算MLM损失（忽略填充的0）
        active_mask = masked_tokens.ne(0)  # [B, max_pred]
        loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)
        loss_lm = (loss_lm * active_mask.float()).sum() / active_mask.float().sum()
        
        # 计算NSP损失
        loss_clsf = criterion(logits_clsf, is_next)
        loss = loss_lm + loss_clsf
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} Loss: {loss.item():.4f} (LM: {loss_lm.item():.4f}, CLS: {loss_clsf.item():.4f})")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 测试预测
    test_input, test_seg, test_masked_tokens, test_masked_pos, test_is_next = batch[0]
    test_input = torch.LongTensor([test_input])
    test_seg = torch.LongTensor([test_seg])
    test_masked_pos = torch.LongTensor([test_masked_pos])
    
    with torch.no_grad():
        logits_lm, logits_clsf = model(test_input, test_seg, test_masked_pos)
    
    # 解析预测结果
    pred_lm = logits_lm.argmax(dim=-1).squeeze().tolist()
    pred_clsf = logits_clsf.argmax(dim=-1).item()
    
    print("\n输入序列:", [number_dict[i.item()] for i in test_input.squeeze() if i != 0]) 
    print("掩码位置真实标签:", [number_dict[i] for i in test_masked_tokens if i != 0])
    print("掩码位置预测结果:", [number_dict[i] for i in pred_lm if i != 0])
    print("下一句预测:", "是" if pred_clsf == test_is_next else "否")