import nltk
import numpy as np
from nltk.corpus import gutenberg
from collections import Counter

# 设置 NLTK 数据集的路径
nltk.data.path.append('/data/yanxinrui/Vocoders_forensics/DLNLP')

def entropy(probabilities):
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

def calculate_entropy(text):
    # 计算字符级别的熵
    char_counts = Counter(text)
    total_chars = sum(char_counts.values())
    char_probs = [count / total_chars for count in char_counts.values()]
    char_entropy = entropy(char_probs)
    
    # 计算单词级别的熵
    words = nltk.word_tokenize(text)
    word_counts = Counter(words)
    total_words = sum(word_counts.values())
    word_probs = [count / total_words for count in word_counts.values()]
    word_entropy = entropy(word_probs)
    
    return char_entropy, word_entropy

# 遍历 Gutenberg 语料库中的所有书籍
book_entropies = {}
for book_id in gutenberg.fileids():
    text = gutenberg.raw(book_id).lower()  # 统一转换为小写
    char_entropy, word_entropy = calculate_entropy(text)
    book_entropies[book_id] = (char_entropy, word_entropy)
    print(f"{book_id}: Character Entropy = {char_entropy:.4f} bits, Word Entropy = {word_entropy:.4f} bits")

# 计算所有书籍的平均信息熵
avg_char_entropy = np.mean([char for char, word in book_entropies.values()])
avg_word_entropy = np.mean([word for char, word in book_entropies.values()])

print(f"\nAverage Character-level Entropy: {avg_char_entropy:.4f} bits")
print(f"Average Word-level Entropy: {avg_word_entropy:.4f} bits")
