import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import os
import pickle

# 定义 Elman 网络类，增加层数和调整结构
class ElmanNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=2, dropout_rate=0.3):
        super(ElmanNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size * 2, output_size)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        h0 = torch.zeros(self.rnn.num_layers, inputs.size(0), self.rnn.hidden_size).to(inputs.device)
        out, _ = self.rnn(embedded, h0)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 定义分词器
tokenizer = get_tokenizer('basic_english')

# 构建词汇表的辅助函数
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# 检查是否有缓存的词汇表
vocab_cache_path = 'vocab_cache.pkl'
if os.path.exists(vocab_cache_path):
    print("Loading vocabulary from cache...")
    with open(vocab_cache_path, 'rb') as f:
        vocab = pickle.load(f)
else:
    # 加载 AG_NEWS 数据集
    train_iter = AG_NEWS(split='train')
    # 增加数据量，取更多数据
    sampled_train_iter = list(train_iter)[:30000]
    # 构建词汇表
    vocab = build_vocab_from_iterator(yield_tokens(sampled_train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    # 保存词汇表到缓存
    with open(vocab_cache_path, 'wb') as f:
        pickle.dump(vocab, f)

# 文本转换函数
text_pipeline = lambda x: vocab(tokenizer(x))
# 确保标签从 0 开始
label_pipeline = lambda x: int(x) - 1

# 定义数据处理函数
def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return label_list, text_list

# 重新加载训练集
train_iter = AG_NEWS(split='train')
# 取更多数据作为小数据集
train_data = list(train_iter)[:30000]
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_batch)

# 定义网络参数
vocab_size = len(vocab)
embedding_dim = 200
hidden_size = 256
output_size = 4
# 初始化 Elman 网络
model = ElmanNetwork(vocab_size, embedding_dim, hidden_size, output_size)

# 定义损失函数和优化器，调整学习率
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)

# 训练网络，增加训练轮数
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for labels, texts in train_dataloader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.4f}')

# 测试网络
test_iter = AG_NEWS(split='test')
test_data = list(test_iter)[:2000]
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_batch)
correct = 0
total = 0
with torch.no_grad():
    model.eval()
    for labels, texts in test_dataloader:
        outputs = model(texts)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {correct / total * 100:.2f}%')

# 预测函数
def predict(sentence, model, vocab, tokenizer):
    model.eval()
    processed_text = torch.tensor(text_pipeline(sentence), dtype=torch.int64).unsqueeze(0)
    with torch.no_grad():
        output = model(processed_text)
        predicted = torch.argmax(output, dim=1).item()
    return predicted

# 类别映射字典
category_mapping = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

# 示例输入
input_sentence = "The stock market is booming today."
predicted_class = predict(input_sentence, model, vocab, tokenizer)
predicted_category = category_mapping[predicted_class]
print(f"输入句子: {input_sentence}")
print(f"预测类别: {predicted_category}")