"""
Step 6 - 深度学习 & PyTorch 系统教程
===================================
运行: python3 06_pytorch_basics.py

路线图: Phase 5 — 深度学习 & PyTorch（预计 4 周）

内容概览:
  Part 1 — PyTorch 基础: Tensor、运算、索引
  Part 2 — Autograd: 自动求导机制
  Part 3 — 线性回归实战
  Part 4 — Dataset & DataLoader
  Part 5 — 全连接网络 (FNN) 分类手写数字
  Part 6 — CNN 卷积神经网络
  Part 7 — RNN / LSTM 循环神经网络
  Part 8 — Transformer 注意力机制
  Part 9 — 实战：文本情感分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import math

# ============================================================
# Part 1 — PyTorch 基础: Tensor
# ============================================================
print("=" * 60)
print("Part 1: PyTorch Tensor 基础")
print("=" * 60)

# --- 创建 Tensor ---
# 从列表创建
t1 = torch.tensor([1, 2, 3, 4])
print(f"从列表创建: {t1}")

# 从 NumPy 创建
arr = np.array([5, 6, 7, 8])
t2 = torch.from_numpy(arr)
print(f"从 NumPy 创建: {t2}")

# 特殊张量
zeros = torch.zeros(2, 3)          # 全零
ones = torch.ones(2, 3)            # 全一
rand = torch.rand(2, 3)            # 均匀分布 [0,1)
randn = torch.randn(2, 3)          # 标准正态分布
eye = torch.eye(3)                 # 单位矩阵
print(f"zeros(2,3):\n{zeros}")
print(f"randn(2,3):\n{randn}")

# --- Tensor 属性 ---
x = torch.randn(3, 4)
print(f"shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")

# --- Tensor 运算 ---
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(f"加法: {a + b}")
print(f"乘法: {a * b}")
print(f"点积: {torch.dot(a, b)}")

# 矩阵乘法
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = torch.mm(A, B)     # 2x4
print(f"矩阵乘法 shape: {C.shape}")

# --- 索引和切片（和 NumPy 一样） ---
x = torch.randn(4, 5)
print(f"第0行: {x[0]}")
print(f"第0行前3列: {x[0, :3]}")
print(f"所有行第1列: {x[:, 1]}")

# --- 形状操作 ---
x = torch.randn(2, 3, 4)
print(f"原始 shape: {x.shape}")
print(f"reshape(2, 12): {x.reshape(2, 12).shape}")
print(f"展平: {x.view(-1).shape}")

# --- GPU 支持（如果有） ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    x_gpu = x.to(device)
    print(f"已移动到 GPU: {x_gpu.device}")
else:
    print("CPU 模式（无 CUDA）")

print("\n")


# ============================================================
# Part 2 — Autograd: 自动求导
# ============================================================
print("=" * 60)
print("Part 2: Autograd — 自动求导")
print("=" * 60)

# requires_grad = True 跟踪计算图
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1   # y = x² + 3x + 1
print(f"x = {x}, y = {y}")

# 反向传播求导 dy/dx = 2x + 3，在 x=2 处 = 7
y.backward()
print(f"dy/dx (x=2) = {x.grad}")   # 应该是 7

# 多次求导需要清除梯度
x = torch.tensor([2.0], requires_grad=True)
z = x ** 3          # dz/dx = 3x² = 12
z.backward()
print(f"dz/dx (x=2) = {x.grad}")

# 停止梯度追踪
x = torch.tensor([2.0], requires_grad=True)
with torch.no_grad():
    y = x * 2
print(f"no_grad 下 y.requires_grad = {y.requires_grad}")

print("\n")


# ============================================================
# Part 3 — 线性回归实战 (用 Autograd 手动训练)
# ============================================================
print("=" * 60)
print("Part 3: 线性回归实战")
print("=" * 60)

# 生成数据: y = 2 * x + 1 + 噪声
np.random.seed(42)
X_np = np.random.rand(100, 1).astype(np.float32)
y_np = (2 * X_np + 1 + 0.1 * np.random.randn(100, 1)).astype(np.float32)

X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np)

# 初始化参数
w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 训练
lr = 0.1
n_epochs = 500
for epoch in range(n_epochs):
    # 前向传播
    y_pred = X @ w + b
    loss = ((y_pred - y) ** 2).mean()

    # 反向传播
    loss.backward()

    # 梯度下降（手动更新）
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        # 清零梯度
        w.grad.zero_()
        b.grad.zero_()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d}, Loss = {loss.item():.6f}")

print(f"训练结果: w = {w.item():.4f}, b = {b.item():.4f} (真实: w=2, b=1)")
print("用 nn.Linear 实现会更简洁，下面看！")
print("\n")


# ============================================================
# Part 4 — Dataset & DataLoader
# ============================================================
print("=" * 60)
print("Part 4: Dataset & DataLoader")
print("=" * 60)

# 方式 1: TensorDataset（简单场景）
X_data = torch.randn(100, 3)
y_data = torch.randint(0, 2, (100,))
dataset = TensorDataset(X_data, y_data)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch_x, batch_y in loader:
    print(f"批次 shape: X={batch_x.shape}, y={batch_y.shape}")
    break

# 方式 2: 自定义 Dataset（真实场景）
class TextDataset(Dataset):
    """自定义文本数据集示例"""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": self.labels[idx],
            "index": idx,
        }

# 示例使用
texts = ["好评", "差评", "一般", "很好"]
labels = [1, 0, 0, 1]
custom_dataset = TextDataset(texts, labels)
custom_loader = DataLoader(custom_dataset, batch_size=2, shuffle=True)

for batch in custom_loader:
    print(f"自定义 Dataset 批次: {batch}")
    break

print("\n")


# ============================================================
# Part 5 — 全连接网络 (FNN) 分类手写数字
# ============================================================
print("=" * 60)
print("Part 5: 全连接网络 — MNIST 手写数字分类")
print("=" * 60)

# --- 定义模型 ---
class FNN(nn.Module):
    """全连接神经网络"""
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (batch, 1, 28, 28) -> (batch, 784)
        x = x.view(x.size(0), -1)
        return self.net(x)

model = FNN()
print(f"模型结构:\n{model}")

# 用随机数据测试前向传播
dummy_input = torch.randn(4, 1, 28, 28)
dummy_output = model(dummy_input)
print(f"输入 shape: {dummy_input.shape}")
print(f"输出 shape: {dummy_output.shape}")   # (4, 10)
print(f"预测类别: {dummy_output.argmax(dim=1)}")

# --- 训练流程模板 ---
def train_one_epoch(model, loader, criterion, optimizer, device="cpu"):
    """一个 epoch 的训练"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100.0 * correct / total

def evaluate(model, loader, criterion, device="cpu"):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100.0 * correct / total

print("训练模板已定义（train_one_epoch / evaluate）")
print("❗ 想跑完整 MNIST 训练？取消下面注释运行：")
print("""
# from torchvision import datasets, transforms
# train_loader = DataLoader(
#     datasets.MNIST('./data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=64, shuffle=True)
# test_loader = DataLoader(
#     datasets.MNIST('./data', train=False, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=64, shuffle=False)
#
# model = FNN()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# for epoch in range(10):
#     loss, acc = train_one_epoch(model, train_loader, criterion, optimizer)
#     test_loss, test_acc = evaluate(model, test_loader, criterion)
#     print(f"Epoch {epoch}: Train Acc={acc:.2f}%, Test Acc={test_acc:.2f}%")
""")

print("\n")


# ============================================================
# Part 6 — CNN 卷积神经网络
# ============================================================
print("=" * 60)
print("Part 6: CNN 卷积神经网络")
print("=" * 60)

class CNN(nn.Module):
    """卷积神经网络"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (1,28,28) -> (32,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                                # (32,14,14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # (64,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),                                # (64,7,7)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (128,7,7)
            nn.ReLU(),
            nn.MaxPool2d(2),                                # (128,3,3)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)   # 展平
        x = self.classifier(x)
        return x

cnn = CNN()
dummy = torch.randn(4, 1, 28, 28)
print(f"CNN 输入: {dummy.shape}, 输出: {cnn(dummy).shape}")

# --- CNN 核心概念示意 ---
print("\n--- CNN 核心概念 ---")
print("Conv2d: 卷积核提取局部特征")
print("  - kernel_size: 感受野大小（如 3x3）")
print("  - padding: 边缘填充（保持尺寸）")
print("  - stride: 步长（滑动步数）")
print("MaxPool2d: 下采样，保留最显著特征")
print("  - 2x2 池化: 尺寸减半")
print("Dropout: 随机丢弃神经元，防止过拟合")
print("\n")


# ============================================================
# Part 7 — RNN / LSTM
# ============================================================
print("=" * 60)
print("Part 7: RNN / LSTM 循环神经网络")
print("=" * 60)

class LSTMClassifier(nn.Module):
    """LSTM 文本分类器"""
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256,
                 num_layers=2, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len) — token ids
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        # 取最后一个时间步的 hidden state
        last_hidden = h_n[-1]                 # (batch, hidden_dim)
        return self.classifier(last_hidden)

lstm_model = LSTMClassifier()
dummy_text = torch.randint(0, 1000, (4, 20))   # 4个样本，每个20个token
print(f"LSTM 输入: {dummy_text.shape}")
print(f"LSTM 输出: {lstm_model(dummy_text).shape}")

# --- RNN 核心概念 ---
print("\n--- RNN/LSTM 核心概念 ---")
print("RNN 问题: 长程依赖缺失（梯度消失）")
print("LSTM 解决: 遗忘门 + 输入门 + 输出门")
print("  - 遗忘门: 决定丢弃哪些历史信息")
print("  - 输入门: 决定写入哪些新信息")
print("  - 输出门: 决定输出哪些信息")
print("batch_first=True: 输入形状为 (batch, seq_len, feature)")
print("双向 LSTM: 同时看上文和下文 (bidirectional=True)")
print("\n")


# ============================================================
# Part 8 — Transformer 注意力机制
# ============================================================
print("=" * 60)
print("Part 8: Transformer — 注意力机制")
print("=" * 60)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力

    Args:
        Q: Query   (batch, heads, seq_len, d_k)
        K: Key     (batch, heads, seq_len, d_k)
        V: Value   (batch, heads, seq_len, d_v)
        mask: 可选掩码（如 padding mask）

    Returns:
        output: (batch, heads, seq_len, d_v)
        attention_weights: (batch, heads, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    # 注意力分数: Q @ K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 线性投影 + 分头
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        output, attn = scaled_dot_product_attention(Q, K, V, mask)

        # 合并头
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        return self.W_O(output)

class TransformerBlock(nn.Module):
    """一个 Transformer Encoder 块"""
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 多头注意力 + 残差连接 + LayerNorm
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN + 残差连接 + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

# 测试 Transformer Block
batch_size, seq_len, d_model = 2, 10, 128
x = torch.randn(batch_size, seq_len, d_model)
transformer_block = TransformerBlock(d_model=d_model, num_heads=4, d_ff=512)
output = transformer_block(x)
print(f"Transformer 输入: {x.shape}")
print(f"Transformer 输出: {output.shape}")   # (2, 10, 128) — 形状不变，语义变了

# --- Transformer 核心概念 ---
print("\n--- Transformer 核心概念 ---")
print("1. 自注意力 (Self-Attention)")
print("   Q, K, V 都来自同一个输入")
print("   每个位置关注所有位置")
print()
print("2. 多头注意力 (Multi-Head)")
print("   多个注意力头捕捉不同角度的关系")
print("   比如: 一个头关注语法，一个头关注语义")
print()
print("3. 位置编码 (Positional Encoding)")
print("   Transformer 没有顺序信息")
print("   用 sin/cos 函数注入位置信息")
print()
print("4. 残差连接 + LayerNorm")
print("   解决深层网络退化问题")
print("   稳定训练过程")
print()
print("5. Encoder-Decoder 结构")
print("   Encoder: 理解输入")
print("   Decoder: 生成输出")
print()

# --- 面试高频考点：手写 Scaled Dot-Product Attention ---
print("--- 面试高频考点: 手写 Attention ---")
print("""
def attention(Q, K, V):
    # Q, K, V: (batch, seq_len, d_k)
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return weights @ V   # (batch, seq_len, d_v)
""")

print("\n")


# ============================================================
# Part 9 — 实战：文本情感分类（完整流程）
# ============================================================
print("=" * 60)
print("Part 9: 实战 — 文本情感分类完整流程")
print("=" * 60)

# --- 9.1 准备数据 ---
texts = [
    "这个电影太好看了，非常推荐！",
    "很差劲，浪费了两个小时。",
    "中规中矩，没什么亮点。",
    "演技在线，剧情也很棒！",
    "太难看了，完全不值得看。",
    "非常精彩，看了还想看。",
    "一般般吧，不算太好。",
    "烂片，千万别去看！",
    "这部电影真的让人感动。",
    "很无聊，差点睡着。",
]
labels = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]   # 1: 正面, 0: 负面

# 构建词表
def build_vocab(texts):
    """从文本构建词表"""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in texts:
        for char in text:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab

vocab = build_vocab(texts)
print(f"词表大小: {len(vocab)}")

def encode_text(text, vocab, max_len=20):
    """将文本编码为 token ids"""
    ids = [vocab.get(c, vocab["<UNK>"]) for c in text]
    # padding / truncation
    if len(ids) < max_len:
        ids = ids + [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

max_len = 20
X = torch.tensor([encode_text(t, vocab, max_len) for t in texts])
y = torch.tensor(labels)
print(f"编码后: X shape={X.shape}, y shape={y.shape}")

# --- 9.2 定义模型 ---
class SentimentClassifier(nn.Module):
    """情感分类模型（字级别）"""
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 用 CNN 提取 n-gram 特征
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)                      # (batch, seq_len, embed_dim)
        emb = emb.transpose(1, 2)                    # (batch, embed_dim, seq_len) 对 Conv1d

        c1 = self.pool(F.relu(self.conv1(emb))).squeeze(-1)   # (batch, 64)
        c2 = self.pool(F.relu(self.conv2(emb))).squeeze(-1)   # (batch, 64)
        combined = torch.cat([c1, c2], dim=-1)       # (batch, 128)

        return self.classifier(combined)

# --- 9.3 训练 ---
model = SentimentClassifier(len(vocab), embed_dim=32, hidden_dim=64)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print("\n训练情感分类模型...")
for epoch in range(50):
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if epoch % 10 == 0:
        acc = 100.0 * correct / total
        print(f"Epoch {epoch:2d}, Loss={total_loss/len(loader):.4f}, Acc={acc:.1f}%")

# --- 9.4 测试 ---
model.eval()
test_texts = [
    "太好看了！推荐！",
    "非常无聊，不好看。",
]
test_X = torch.tensor([encode_text(t, vocab, max_len) for t in test_texts])

with torch.no_grad():
    outputs = model(test_X)
    probs = F.softmax(outputs, dim=1)
    predictions = outputs.argmax(dim=1)

for text, pred, prob in zip(test_texts, predictions, probs):
    sentiment = "正面 😊" if pred == 1 else "负面 😞"
    confidence = prob[pred].item()
    print(f"\"{text}\" -> {sentiment} (置信度: {confidence:.2%})")

print("\n")


# ============================================================
# 总结 & 下一步
# ============================================================
print("=" * 60)
print("Phase 5 知识点总结")
print("=" * 60)
print("""
✅ Part 1: Tensor 创建、运算、索引
✅ Part 2: Autograd 自动求导
✅ Part 3: 线性回归实战
✅ Part 4: Dataset & DataLoader
✅ Part 5: 全连接网络 (FNN)
✅ Part 6: 卷积神经网络 (CNN)
✅ Part 7: 循环神经网络 (RNN/LSTM)
✅ Part 8: Transformer 注意力机制
✅ Part 9: 文本情感分类实战

下一步 — Phase 6: NLP 项目实战
  ✅ 项目1: 中文情感分析 (PyTorch + TextCNN)
  ✅ 项目2: 智能问答机器人 (RAG + 混合检索)
  项目3: 文本摘要工具 (Seq2Seq / LLM)

🔥 面试重点: 手写 Attention、Transformer 架构、CNN 原理
""")
