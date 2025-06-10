import numpy as np
from collections import defaultdict

class NGramModel:
    def __init__(self, n, smoothing=0.0):
        self.n = n  # n-gram的阶数（如n=2表示bigram）
        self.smoothing = smoothing  # 平滑参数（0表示无平滑）
        self.model = defaultdict(lambda: defaultdict(int))  # 存储n-gram的频率
        self.vocab = set()  # 词汇表
    
    def train(self, corpus):
        """
        训练n-gram模型
        参数：
            corpus: 输入语料，格式为句子列表，每个句子是单词列表
        """
        for sentence in corpus:
            # 在句子前后添加特殊标记（<s>表示句首，</s>表示句尾）
            sentence = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            
            # 更新词汇表
            self.vocab.update(sentence)
            
            for i in range(len(sentence) - self.n + 1):
                # 提取n-gram窗口
                history = tuple(sentence[i:i+self.n-1])
                next_word = sentence[i+self.n-1]
                self.model[history][next_word] += 1  # 统计频率
    
    def get_prob(self, history, word):
        """
        计算条件概率P(word | history)，支持平滑处理
        """
        history = tuple(history)
        count_history = sum(self.model[history].values())
        
        if self.smoothing > 0:
            # 拉普拉斯平滑
            count_history_word = self.model[history].get(word, 0)
            return (count_history_word + self.smoothing) / (count_history + self.smoothing * len(self.vocab))
        else:
            # 无平滑（零概率）
            if count_history == 0:
                return 0.0
            return self.model[history].get(word, 0) / count_history
    
    def generate_text(self, start_words=None, max_length=20, use_sampling=False, temperature=1.0):
        """
        生成文本
        参数：
            start_words: 起始单词列表（长度需≥n-1）
            max_length: 生成文本的最大长度
            use_sampling: 是否使用概率采样（而非贪心搜索）
            temperature: 采样温度，控制随机性（仅在use_sampling=True时有效）
        """
        if start_words is None:
            # 随机选择以<s>开头的n-1个词作为起始
            start_history = np.random.choice([h for h in self.model.keys() if h[0] == '<s>'])
        else:
            if len(start_words) != self.n - 1:
                raise ValueError(f"start_words长度需为{self.n-1}")
            start_history = tuple(start_words)
        
        generated = list(start_history)
        for _ in range(max_length):
            # 获取当前历史的所有可能下一个词及其概率
            current_history = tuple(generated[-self.n+1:])
            next_words = self.model.get(current_history, {})
            
            if not next_words:
                break  # 无法继续生成
            
            if use_sampling:
                # 基于概率的采样
                words, probs = zip(*next_words.items())
                
                # 应用温度调整
                if temperature != 1.0:
                    probs = np.array([p ** (1.0 / temperature) for p in probs])
                    probs = probs / np.sum(probs)
                
                next_word = np.random.choice(words, p=probs)
            else:
                # 贪心选择概率最高的词
                next_word = max(next_words, key=lambda w: next_words[w])
            
            generated.append(next_word)
            
            # 遇到句尾标记则结束
            if next_word == '</s>':
                break
        
        # 去除特殊标记
        return ' '.join([word for word in generated if word not in {'<s>', '</s>'}])

# 示例语料（简单英文句子）
corpus = [
    ["i", "like", "dog"],
    ["i", "love", "coffee"],
    ["i", "hate", "milk"],
    ["dog", "likes", "meat"],
    ["coffee", "is", "good"],
    ["milk", "is", "white"]
]

# 训练bigram模型（n=2），使用拉普拉斯平滑（k=0.1）
model = NGramModel(n=2, smoothing=0.1)
model.train(corpus)

# 查看i的下一个词的概率分布
history = ["i"]
print(f"P(word | {' '.join(history)}) 概率分布:")
for word in sorted(model.vocab):
    if word != '<s>':  # 排除<s>，因为它不会出现在i后面
        prob = model.get_prob(history, word)
        if prob > 0:
            print(f"  {word}: {prob:.4f}")

# 生成文本（贪心搜索）
generated_greedy = model.generate_text(start_words=["i"], max_length=10)
print("\n贪心生成:", generated_greedy)

# 生成文本（概率采样，温度=0.7）
generated_sampling = model.generate_text(
    start_words=["i"], 
    max_length=10, 
    use_sampling=True,
    temperature=0.7
)
print("采样生成:", generated_sampling)