import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import string

# -----------------------
# NLTK 资源下载（自动处理路径）
# -----------------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)  # 确保下载 punkt 分词器
nltk.download('punkt_tab', quiet=True)  # 下载 punkt_tab 资源

nltk.data.path.append('/home/morgan/nltk_data')
# -----------------------
# 文本预处理配置（直接使用 nltk.word_tokenize）
# -----------------------
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def preprocess_text(text):
    """文本清洗、分词、去停用词、词形还原"""
    # 1. 转小写并移除标点
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    # 2. 分词（直接使用 nltk.word_tokenize，NLTK 自动加载 punkt 分词器）
    tokens = nltk.word_tokenize(text)
    # 3. 去停用词和非字母词
    tokens = [token for token in tokens if token.isalpha() and token not in STOPWORDS]
    # 4. 词形还原
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens]
    return tokens

# -----------------------
# TF-IDF向量化（手动实现）
# -----------------------
class TfidfVectorizer:
    def __init__(self):
        self.vocab = []  # 词汇表
        self.idf = {}    # 逆文档频率

    def fit(self, corpus):
        """构建词汇表并计算IDF"""
        all_words = defaultdict(int)
        doc_count = 0
        for doc in corpus:
            doc_words = set(doc)  # 去重计算文档出现次数
            for word in doc_words:
                all_words[word] += 1
            doc_count += 1
        self.vocab = list(all_words.keys())
        self.idf = {
            word: np.log(doc_count / (count + 1)) 
            for word, count in all_words.items()
        }

    def transform(self, corpus):
        """将文本转换为TF-IDF矩阵"""
        X = []
        for doc in corpus:
            tf = defaultdict(int)
            for word in doc:
                tf[word] += 1
            vec = [tf[word] * self.idf[word] for word in self.vocab]
            X.append(vec)
        return np.array(X)

# -----------------------
# 示例数据与训练流程
# -----------------------
if __name__ == "__main__":
    # 增加更多多样化的训练数据
    data = [
        ("Great product! Highly recommend.", 1),
        ("Terrible experience. Waste of money.", -1),
        ("Works well, better than expected.", 1),
        ("Worst service ever. Avoid this.", -1),
        ("Good quality, but expensive.", 1),
        ("Not worth the price. Poor performance.", -1),
        ("This is an amazing item! I love it.", 1),
        ("It's a complete disaster. Stay away.", -1),
        ("The product exceeded my expectations.", 1),
        ("The worst thing I've ever bought.", -1),
        ("I'm really satisfied with this purchase.", 1),
        ("I'm extremely disappointed with this product.", -1),
        ("This is the best thing I've bought in a long time.", 1),
        ("This is the most useless product I've ever seen.", -1),
        ("The quality is top-notch. I'll buy again.", 1),
        ("The quality is so poor. Don't even think about it.", -1)
    ]

    # 预处理：文本清洗 + 分词
    corpus = []
    labels = []
    for text, label in data:
        tokens = preprocess_text(text)
        corpus.append(tokens)
        labels.append(label)

    # TF-IDF向量化
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    X = vectorizer.transform(corpus)
    y = np.array(labels)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 使用 sklearn 的 SVC 并通过网格搜索调整参数
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # 获取最佳模型
    best_svm = grid_search.best_estimator_

    # 在测试集上评估模型
    y_pred = best_svm.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_accuracy}")

    # 预测新文本
    new_texts = [
        "Amazing! Best purchase ever.",
        "Horrible. Don't buy this."
    ]
    new_corpus = [preprocess_text(t) for t in new_texts]
    new_X = vectorizer.transform(new_corpus)
    predictions = best_svm.predict(new_X)
    print("Predictions:", ["positive" if p == 1 else "negative" for p in predictions])    