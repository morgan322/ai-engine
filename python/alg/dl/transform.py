import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import random


class Config(object):
    def __init__(self):
        self.vocab_size = 6
        self.d_model = 20
        self.n_heads = 2
        assert self.d_model % self.n_heads == 0
        
        self.dim_k = self.d_model // self.n_heads
        self.dim_v = self.d_model // self.n_heads
        
        self.padding_size = 30
        self.UNK = 5
        self.PAD = 4
        self.N = 6
        self.p = 0.1

config = Config()


class Embedding(nn.Module):
    def __init__(self,vocab_size):
        super(Embedding, self).__init__()
        # 一个普通的 embedding层，我们可以通过设置padding_idx=config.PAD 来实现论文中的 padding_mask
        self.embedding = nn.Embedding(vocab_size,config.d_model,padding_idx=config.PAD)


    def forward(self,x):
        # 根据每个句子的长度，进行padding，短补长截
        for i in range(len(x)):
            if len(x[i]) < config.padding_size:
                x[i].extend([config.UNK] * (config.padding_size - len(x[i]))) # 注意 UNK是你词表中用来表示oov的token索引，这里进行了简化，直接假设为6
            else:
                x[i] = x[i][:config.padding_size]
        x = self.embedding(torch.tensor(x)) # batch_size * seq_len * d_model
        return x



class Positional_Encoding(nn.Module):

    def __init__(self,d_model):
        super(Positional_Encoding,self).__init__()
        self.d_model = d_model


    def forward(self,seq_len,embedding_dim):
        positional_encoding = np.zeros((seq_len,embedding_dim))
        for pos in range(positional_encoding.shape[0]):
            for i in range(positional_encoding.shape[1]):
                positional_encoding[pos][i] = math.sin(pos/(10000**(2*i/self.d_model))) if i % 2 == 0 else math.cos(pos/(10000**(2*i/self.d_model)))
        return torch.from_numpy(positional_encoding)


class Mutihead_Attention(nn.Module):
    def __init__(self,d_model,dim_k,dim_v,n_heads):
        super(Mutihead_Attention, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads

        self.q = nn.Linear(d_model,dim_k)
        self.k = nn.Linear(d_model,dim_k)
        self.v = nn.Linear(d_model,dim_v)

        self.o = nn.Linear(dim_v,d_model)
        self.norm_fact = 1 / math.sqrt(d_model)

    def generate_mask(self,dim):
        # 此处是 sequence mask ，防止 decoder窥视后面时间步的信息。
        # padding mask 在数据输入模型之前完成。
        matirx = np.ones((dim,dim))
        mask = torch.Tensor(np.tril(matirx))

        return mask==1

    def forward(self,x,y,requires_mask=False):
        assert self.dim_k % self.n_heads == 0 and self.dim_v % self.n_heads == 0
        # size of x : [batch_size * seq_len * batch_size]
        # 对 x 进行自注意力
        Q = self.q(x).reshape(-1,x.shape[0],x.shape[1],self.dim_k // self.n_heads) # n_heads * batch_size * seq_len * dim_k
        K = self.k(x).reshape(-1,x.shape[0],x.shape[1],self.dim_k // self.n_heads) # n_heads * batch_size * seq_len * dim_k
        V = self.v(y).reshape(-1,y.shape[0],y.shape[1],self.dim_v // self.n_heads) # n_heads * batch_size * seq_len * dim_v
        # print("Attention V shape : {}".format(V.shape))
        attention_score = torch.matmul(Q,K.permute(0,1,3,2)) * self.norm_fact
        if requires_mask:
            mask = self.generate_mask(x.shape[1])
            # masked_fill 函数中，对Mask位置为True的部分进行Mask
            attention_score.masked_fill(mask,value=float("-inf")) # 注意这里的小Trick，不需要将Q,K,V 分别MASK,只MASKSoftmax之前的结果就好了
        output = torch.matmul(attention_score,V).reshape(y.shape[0],y.shape[1],-1)
        # print("Attention output shape : {}".format(output.shape))

        output = self.o(output)
        return output


class Feed_Forward(nn.Module):
    def __init__(self,input_dim,hidden_dim=2048):
        super(Feed_Forward, self).__init__()
        self.L1 = nn.Linear(input_dim,hidden_dim)
        self.L2 = nn.Linear(hidden_dim,input_dim)

    def forward(self,x):
        output = nn.ReLU()(self.L1(x))
        output = self.L2(output)
        return output

class Add_Norm(nn.Module):
    def __init__(self):
        super(Add_Norm, self).__init__()
        self.dropout = nn.Dropout(config.p)


    def forward(self,x,sub_layer,**kwargs):
        sub_output = sub_layer(x,**kwargs)
        # print("{} output : {}".format(sub_layer,sub_output.size()))
        x = self.dropout(x + sub_output)

        layer_norm = nn.LayerNorm(x.size()[1:])
        out = layer_norm(x)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.positional_encoding = Positional_Encoding(config.d_model)
        self.muti_atten = Mutihead_Attention(config.d_model,config.dim_k,config.dim_v,config.n_heads)
        self.feed_forward = Feed_Forward(config.d_model)

        self.add_norm = Add_Norm()


    def forward(self,x): # batch_size * seq_len 并且 x 的类型不是tensor，是普通list

        x += self.positional_encoding(x.shape[1],config.d_model)
        # print("After positional_encoding: {}".format(x.size()))
        output = self.add_norm(x,self.muti_atten,y=x)
        output = self.add_norm(output,self.feed_forward)

        return output

# 在 Decoder 中，Encoder的输出作为Query和KEy输出的那个东西。即 Decoder的Input作为V。此时是可行的
# 因为在输入过程中，我们有一个padding操作，将Inputs和Outputs的seq_len这个维度都拉成一样的了
# 我们知道，QK那个过程得到的结果是 batch_size * seq_len * seq_len .既然 seq_len 一样，那么我们可以这样操作
# 这样操作的意义是，Outputs 中的 token 分别对于 Inputs 中的每个token作注意力

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.positional_encoding = Positional_Encoding(config.d_model)
        self.muti_atten = Mutihead_Attention(config.d_model,config.dim_k,config.dim_v,config.n_heads)
        self.feed_forward = Feed_Forward(config.d_model)
        self.add_norm = Add_Norm()

    def forward(self,x,encoder_output): # batch_size * seq_len 并且 x 的类型不是tensor，是普通list
        # print(x.size())
        x += self.positional_encoding(x.shape[1],config.d_model)
        # print(x.size())
        # 第一个 sub_layer
        output = self.add_norm(x,self.muti_atten,y=x,requires_mask=True)
        # 第二个 sub_layer
        output = self.add_norm(x,self.muti_atten,y=encoder_output,requires_mask=True)
        # 第三个 sub_layer
        output = self.add_norm(output,self.feed_forward)
        return output

class Transformer_layer(nn.Module):
    def __init__(self):
        super(Transformer_layer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self,x):
        x_input,x_output = x
        encoder_output = self.encoder(x_input)
        decoder_output = self.decoder(x_output,encoder_output)
        return (encoder_output,decoder_output)

class Transformer(nn.Module):
    def __init__(self,N,vocab_size,output_dim):
        super(Transformer, self).__init__()
        self.embedding_input = Embedding(vocab_size=vocab_size)
        self.embedding_output = Embedding(vocab_size=vocab_size)

        self.output_dim = output_dim
        self.linear = nn.Linear(config.d_model,output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.model = nn.Sequential(*[Transformer_layer() for _ in range(N)])


    def forward(self,x):
        x_input , x_output = x
        x_input = self.embedding_input(x_input)
        x_output = self.embedding_output(x_output)

        _ , output = self.model((x_input,x_output))

        output = self.linear(output)
        output = self.softmax(output)

        return output


# 简单的对话数据集
class ChatDataset(Dataset):
    def __init__(self, data, vocab_size, padding_size):
        self.data = data
        self.vocab_size = vocab_size
        self.padding_size = padding_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        # 简单的文本转数字（实际应用中应使用词表）
        input_ids = [min(token, self.vocab_size-1) for token in input_seq]
        target_ids = [min(token, self.vocab_size-1) for token in target_seq]
        
        # 填充序列
        if len(input_ids) < self.padding_size:
            input_ids += [config.PAD] * (self.padding_size - len(input_ids))
        else:
            input_ids = input_ids[:self.padding_size]
            
        if len(target_ids) < self.padding_size:
            target_ids += [config.PAD] * (self.padding_size - len(target_ids))
        else:
            target_ids = target_ids[:self.padding_size]
            
        return {
            'input_ids': torch.tensor(input_ids),
            'target_ids': torch.tensor(target_ids)
        }

def train_model(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids']
            target_ids = batch['target_ids']
            
            input_seqs = input_ids.tolist()
            target_seqs = [seq[:-1] for seq in target_ids.tolist()]  # 移除最后一个token
            
            optimizer.zero_grad()
            outputs = model((input_seqs, target_seqs))
            
            loss = 0
            for i in range(len(outputs)):
                # 调整预测张量的形状，使其与掩码匹配
                pred = outputs[i][:, :-1, :].reshape(-1, config.vocab_size)  # 移除最后一个时间步
                target = target_ids[i][1:].reshape(-1)  # 移除第一个token
                mask = (target != config.PAD)
                if mask.sum() > 0:
                    loss += criterion(pred[mask], target[mask])
            loss /= len(outputs)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# 文本生成函数
def generate_response(model, input_text, max_length=30):
    model.eval()
    with torch.no_grad():
        # 简单的文本转数字
        input_ids = [min(token, config.vocab_size-1) for token in input_text]
        if len(input_ids) < config.padding_size:
            input_ids += [config.PAD] * (config.padding_size - len(input_ids))
        else:
            input_ids = input_ids[:config.padding_size]
        
        # 初始化输出序列（从<START>标记开始，这里用0表示）
        output_ids = [0]
        
        for _ in range(max_length):
            # 准备输入
            input_seqs = [input_ids]
            target_seqs = [output_ids + [config.PAD] * (config.padding_size - len(output_ids))]
            
            # 生成下一个token
            outputs = model((input_seqs, target_seqs))
            next_token_probs = outputs[0][len(output_ids)-1]
            next_token = torch.argmax(next_token_probs).item()
            
            # 添加到输出序列
            output_ids.append(next_token)
            
            # 如果生成了结束标记（这里用1表示），则停止生成
            if next_token == 1:
                break
    
    # 简单的数字转文本
    return output_ids

# 创建简单的训练数据
def create_dummy_data(num_examples=100):
    data = []
    for _ in range(num_examples):
        # 简单对话模式：A问 -> B答
        input_len = random.randint(3, 10)
        output_len = random.randint(3, 10)
        
        # 确保输入和输出的token在词表范围内
        input_seq = [random.randint(2, config.vocab_size-2) for _ in range(input_len)]
        output_seq = [random.randint(2, config.vocab_size-2) for _ in range(output_len)]
        
        # 添加开始和结束标记
        output_seq = [0] + output_seq + [1]  # 0: <START>, 1: <END>
        
        data.append((input_seq, output_seq))
    return data
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# 主函数
def main():
    # 创建数据
    train_data = create_dummy_data(500)
    dataset = ChatDataset(train_data, config.vocab_size, config.padding_size)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 初始化模型
    model = Transformer(N=config.N, vocab_size=config.vocab_size, output_dim=config.vocab_size)
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 训练模型
    train_model(model, dataloader, optimizer, criterion, epochs=10)
    
    # 测试生成功能
    test_input = [2, 3, 4]  # 简单的测试输入
    response = generate_response(model, test_input)
    print(f"输入: {test_input}")
    print(f"生成的回复: {response}")

if __name__ == "__main__":
    main()    