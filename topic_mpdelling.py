import os
import numpy as np
import jieba
import chardet
from gensim import corpora, models
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB  # 这里替换为朴素贝叶斯分类器
from sklearn.metrics import accuracy_score

# ---------------------------
# 1. 自动检测文件编码
# ---------------------------
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # 读取部分内容
    return chardet.detect(raw_data)['encoding']

# ---------------------------
# 2. 读取目录下所有 .txt 文件并合并数据
# ---------------------------
def load_data(directory_path, sample_size=1000, K=100):
    """
    读取指定目录下的所有 .txt 文件，并均匀抽取 sample_size 个段落，每个段落 K 个 token。
    """
    texts, labels = [], []
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            encoding = detect_encoding(file_path)
            print(f"正在读取 {filename}，检测编码为 {encoding}")
            
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                lines = f.readlines()
            
            lines = [line.strip() for line in lines if line.strip()]
            label = filename.replace(".txt", "")
            
            for line in lines:
                texts.append(' '.join(jieba.cut(line)))  # 分词
                labels.append(label)
    
    if len(texts) < sample_size:
        raise ValueError(f'数据量不足，共有 {len(texts)} 条，少于 {sample_size}！')
    
    sampled_indices = np.random.choice(len(texts), size=sample_size, replace=False)
    sampled_texts = [texts[i] for i in sampled_indices]
    sampled_labels = [labels[i] for i in sampled_indices]
    
    return sampled_texts, sampled_labels

# ---------------------------
# 3. 训练LDA模型
# ---------------------------
def train_lda(texts, num_topics=20):
    texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    return lda_model, dictionary, corpus

# ---------------------------
# 4. 获取主题分布向量
# ---------------------------
def get_lda_features(lda_model, dictionary, texts):
    corpus = [dictionary.doc2bow(text.split()) for text in texts]
    features = []
    for doc in corpus:
        topic_dist = lda_model.get_document_topics(doc, minimum_probability=0)
        features.append([prob for _, prob in topic_dist])
    return np.array(features)

# ---------------------------
# 5. 进行分类 & 10折交叉验证
# ---------------------------
def classify_with_cross_validation(features, labels):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    
    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]
        
        # 使用朴素贝叶斯分类器
        classifier = MultinomialNB()  # 这里替换为朴素贝叶斯
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        accuracies.append(acc)
    
    print(f'Average Accuracy: {np.mean(accuracies):.4f}')
    return np.mean(accuracies)

# ---------------------------
# 6. 运行实验
# ---------------------------
if __name__ == "__main__":
    directory_path = "/data3/yanxinrui/DLNLP/Topic Modeling"  # 你的数据集目录
    T_values = [5, 10, 20, 30]  # 主题数量
    K_values = [20, 100, 500, 1000, 3000]  # 段落 token 数
    
    for K in K_values:
        texts, labels = load_data(directory_path, sample_size=1000, K=K)
        
        for T in T_values:
            print(f'Processing: K={K}, T={T}')
            lda_model, dictionary, _ = train_lda(texts, num_topics=T)
            features = get_lda_features(lda_model, dictionary, texts)
            avg_acc = classify_with_cross_validation(features, labels)
            print(f'Result for K={K}, T={T}: {avg_acc:.4f}\n')
