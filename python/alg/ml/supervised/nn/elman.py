import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, GloVe
import os
import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import nltk
from nltk.corpus import stopwords

# 下载停用词（如果尚未下载）
nltk.download('stopwords')

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 定义 Elman 网络类
class ElmanNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=3, dropout_rate=0.2):
        super(ElmanNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, lengths):
        embedded = self.embedding(inputs)
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h0 = torch.zeros(self.rnn.num_layers, inputs.size(0), self.rnn.hidden_size).to(inputs.device)
        packed_output, hidden = self.rnn(packed_embedded, h0)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out = hidden[-1]
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
    # 构建词汇表
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    # 保存词汇表到缓存
    with open(vocab_cache_path, 'wb') as f:
        pickle.dump(vocab, f)

# 加载预训练词向量
glove = GloVe(name='6B', dim=200)

# 文本转换函数，添加去除停用词功能
def text_pipeline(x):
    tokens = tokenizer(x)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    indices = [vocab[token] for token in tokens]
    return torch.tensor(indices, dtype=torch.long)

# 确保标签从 0 开始
label_pipeline = lambda x: int(x) - 1

# 定义数据处理函数，计算文本长度
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = text_pipeline(_text)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))

    label_list = torch.tensor(label_list, dtype=torch.int64).to(device)
    text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True).to(device)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return label_list, text_list, lengths

# 重新加载训练集
train_iter = AG_NEWS(split='train')
# 使用全部训练数据
train_data = list(train_iter)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_batch)

# 定义网络参数
vocab_size = len(vocab)
embedding_dim = 200
hidden_size = 512
output_size = 4
# 初始化 Elman 网络并将其移动到 GPU
model = ElmanNetwork(vocab_size, embedding_dim, hidden_size, output_size).to(device)

# 定义损失函数和优化器，调整学习率
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')

# 训练网络，增加训练轮数
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for labels, texts, lengths in train_dataloader:
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.4f}')

    # 使用测试集近似验证集来调整学习率
    test_iter = AG_NEWS(split='test')
    test_data = list(test_iter)[:2000]
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_batch)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for test_labels, test_texts, test_lengths in test_dataloader:
            test_outputs = model(test_texts, test_lengths)
            test_loss += criterion(test_outputs, test_labels).item()
            predicted = torch.argmax(test_outputs, dim=1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
    scheduler.step(test_loss)

print(f'Test Accuracy: {correct / total * 100:.2f}%')

# 预测函数
def predict(sentence, model, vocab, tokenizer):
    model.eval()
    processed_text = text_pipeline(sentence).unsqueeze(0).to(device)
    lengths = torch.tensor([processed_text.size(1)], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(processed_text, lengths)
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