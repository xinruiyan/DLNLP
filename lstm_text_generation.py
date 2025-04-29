import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import re

# ========== 超参数 ==========
SEQ_LEN = 60
BATCH_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
NUM_EPOCHS = 100
LR = 0.003
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = "jinyong_lstm_model.pth"
MIN_CHAR_FREQ = 10

# ========== 文本预处理 ==========
def load_corpus_from_folder(folder_path):
    text = ""
    for filepath in sorted(glob.glob(os.path.join(folder_path, "*.txt"))):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            text += '\n' + content
    return text

def clean_chinese_text(text):
    text = re.sub(r'[^\u4e00-\u9fa5。，！？、“”‘’；：——《》（）\n]', '', text)
    text = re.sub(r'\u3000', '', text)  # 清除全角空格
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text

# ========== 载入并清洗语料 ==========
corpus_dir = "/data/yanxinrui/Vocoders_forensics/DLNLP/corpora/jinyong"
text = clean_chinese_text(load_corpus_from_folder(corpus_dir))
text = text[:1000000]  # 可限制语料长度

# 过滤低频字符
char_counts = Counter(text)
valid_chars = set([ch for ch, count in char_counts.items() if count >= MIN_CHAR_FREQ])
text = ''.join([ch for ch in text if ch in valid_chars])

# 构建词典
chars = sorted(list(valid_chars))
char2idx = {ch: idx for idx, ch in enumerate(chars)}
idx2char = {idx: ch for ch, idx in char2idx.items()}
vocab_size = len(chars)

# ========== 编码数据 ==========
def encode_text(text, seq_len):
    inputs, targets = [], []
    for i in range(len(text) - seq_len):
        seq_in = text[i:i+seq_len]
        seq_out = text[i+seq_len]
        inputs.append([char2idx[ch] for ch in seq_in])
        targets.append(char2idx[seq_out])
    return np.array(inputs), np.array(targets)

X, y = encode_text(text, SEQ_LEN)

class TextDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = TextDataset(X, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ========== 模型定义 ==========
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embed(x)
        output, hidden = self.lstm(x, hidden)
        out = self.fc(output[:, -1, :])
        return out, hidden

model = LSTMModel(vocab_size, HIDDEN_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ========== 训练模型 ==========
print("开始训练模型...\n")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} 完成，平均损失: {avg_loss:.4f}")

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'char2idx': char2idx,
        'idx2char': idx2char,
    }, MODEL_SAVE_PATH)
    print(f"模型已保存到: {MODEL_SAVE_PATH}\n")

# ========== 文本生成函数 ==========
def post_process_output(text):
    text = re.sub(r'([壬棬飬仰塣磬]{2,})', '', text)  # 删除重复异常字符
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def generate_text(model, start_text, char2idx, idx2char, max_len=300):
    model.eval()
    input_seq = [char2idx.get(ch, 0) for ch in start_text[-SEQ_LEN:]]
    input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(DEVICE)

    result = list(start_text)
    hidden = None

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            probs = F.softmax(output.squeeze(), dim=0).cpu().numpy()

            # 弱化异常字符的概率
            for bad_ch in ['\n', '壬', '棬', '飬', '磬', '仰', '塣']:
                if bad_ch in char2idx:
                    probs[char2idx[bad_ch]] *= 0.05

            probs = probs / probs.sum()

            next_idx = np.random.choice(len(probs), p=probs)
            next_char = idx2char[next_idx]
            result.append(next_char)

            input_seq = torch.cat([
                input_seq[:, 1:],
                torch.tensor([[next_idx]], dtype=torch.long).to(DEVICE)
            ], dim=1)

    return post_process_output(''.join(result))

# ========== 文本生成示例 ==========
start_text = "郭靖说道："
generated = generate_text(model, start_text, char2idx, idx2char, max_len=300)
print("===== 生成文本如下 =====\n")
print(generated)