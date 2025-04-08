import os
import jieba
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ========== 1. 加载语料 ==========

def load_corpus(directory_path):
    corpus = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                corpus.append(text)
    return corpus

corpus = load_corpus("/data3/yanxinrui/DLNLP/Corpos")

# ========== 2. 中文分词处理 ==========

def preprocess(corpus):
    sentences = []
    for doc in corpus:
        lines = doc.split('\n')
        for line in lines:
            words = jieba.lcut(line.strip())
            if words:
                sentences.append(words)
    return sentences

tokenized_sentences = preprocess(corpus)

# ========== 3. Word2Vec 训练 ==========

model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=5, workers=4)
model.save("w2v_jinyong.model")

print("模型训练完成，保存为 w2v_jinyong.model")

# ========== 4. 词向量可视化（PCA 降维） ==========

def visualize_words(words, model):
    word_vecs = []
    valid_words = []
    for word in words:
        if word in model.wv:
            valid_words.append(word)
            word_vecs.append(model.wv[word])
    
    pca = PCA(n_components=2)
    result = pca.fit_transform(word_vecs)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(valid_words):
        plt.scatter(result[i, 0], result[i, 1])
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.title("Word2Vec词向量可视化 (PCA)")
    plt.grid()
    plt.savefig("word2vec_pca.png")
    plt.show()


test_words = ["张无忌", "赵敏", "周芷若", "灭绝师太", "倚天剑", "九阳神功", "峨眉", "明教"]
visualize_words(test_words, model)

# ========== 5. 聚类分析 ==========

def cluster_words(words, model, num_clusters=3):
    vectors = [model.wv[word] for word in words if word in model.wv]
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(vectors)

    for i, word in enumerate(words):
        if word in model.wv:
            print(f"{word} 属于第 {clusters[i]} 类")

cluster_words(test_words, model, num_clusters=3)

# ========== 6. 相似词测试 ==========

def show_similar_words(word, model):
    if word in model.wv:
        print(f"\n与『{word}』最相近的词：")
        for similar_word, similarity in model.wv.most_similar(word, topn=10):
            print(f"{similar_word}: {similarity:.4f}")
    else:
        print(f"{word} 不在词汇表中")

show_similar_words("张无忌", model)
