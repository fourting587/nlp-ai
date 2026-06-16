# Transformer 架构

## 什么是 Transformer？
Transformer 是 2017 年 Google 提出的"Attention Is All You Need"中的架构。
完全基于注意力机制，没有循环和卷积，是当前 NLP 和 CV 领域的基础架构。

## 核心创新
1. **自注意力机制 (Self-Attention)**: 每个位置关注所有位置
2. **多头注意力 (Multi-Head Attention)**: 多个子空间并行学习
3. **位置编码 (Positional Encoding)**: 注入序列位置信息
4. **残差连接 + LayerNorm**: 解决深层网络退化

## Scaled Dot-Product Attention
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

- Q (Query): 查询向量
- K (Key): 键向量
- V (Value): 值向量
- d_k: 缩放因子，防止内积过大

## 多头注意力
将 Q, K, V 投影到 h 个不同的子空间，分别计算注意力后拼接：
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

每个头关注不同的关系模式（语法、语义、位置等）。

## Transformer Encoder
1. 输入嵌入 + 位置编码
2. N 个 Encoder Block，每个包含:
   - 多头自注意力 (Multi-Head Self-Attention)
   - 残差连接 + LayerNorm
   - 前馈网络 (FFN: Linear → ReLU → Linear)
   - 残差连接 + LayerNorm

## Transformer Decoder
除了 Encoder 的组件外，还有:
- **Masked Self-Attention**: 防止看到未来 token
- **Cross-Attention**: Q 来自 Decoder，K, V 来自 Encoder

## 位置编码
- 使用不同频率的 sin/cos 函数
- PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
- 优点: 可处理任意长度序列，无需额外参数

## BERT (Bidirectional Encoder Representations from Transformers)
- 使用 Transformer Encoder
- **预训练任务1: Masked LM** — 随机掩盖 15% 的 token 并预测
- **预训练任务2: Next Sentence Prediction** — 预测两句话是否连续
- 可以微调用于各种下游任务

## GPT (Generative Pre-trained Transformer)
- 使用 Transformer Decoder
- **预训练任务**: 自回归语言模型（预测下一个 token）
- 单向（从左到右），适合生成任务

## Transformer 的优缺点
**优点**:
- 并行计算（不像 RNN 需要串行）
- 长程依赖捕获能力强
- 可扩展到大规模数据和参数

**缺点**:
- 计算复杂度 O(n^2)（n 为序列长度）
- 需要大量训练数据
- 位置编码不是位置感知的

## 改进版本
- **BERT**: Encoder-only，理解任务
- **GPT**: Decoder-only，生成任务
- **T5**: Encoder-Decoder，文本到文本框架
- **RoBERTa**: BERT 优化版（更多数据、更大 batch、动态 masking）
- **ALBERT**: 参数共享，更轻量
- **XLNet**: 排列语言模型，结合自回归和自编码
