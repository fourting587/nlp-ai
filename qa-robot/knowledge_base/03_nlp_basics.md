# NLP 基础

## 什么是 NLP？
自然语言处理 (Natural Language Processing) 是让计算机理解、生成和处理人类语言的领域。
核心任务: 文本分类、命名实体识别、关系抽取、机器翻译、文本摘要、问答系统等。

## 文本表示方法

### 离散表示
- **One-hot**: 每个词一个维度，维度灾难，无语义信息
- **Bag of Words (BoW)**: 词频向量，忽略词序
- **TF-IDF**: 词频-逆文档频率，降低常见词权重

### 分布式表示
- **Word2Vec**: 用上下文预测词 (CBOW) 或用词预测上下文 (Skip-gram)
- **GloVe**: 基于全局词共现统计
- **FastText**: 考虑子词信息（字符 n-gram）
- **BERT Embedding**: 上下文相关的动态表示

## 子词分词 (Subword Tokenization)
解决 OOV (Out-of-Vocabulary) 问题:
- **BPE (Byte Pair Encoding)**: 迭代合并最频繁的字符对
- **WordPiece**: 基于概率的合并策略，BERT 使用
- **SentencePiece**: 不依赖空格分割，支持多语言
- **Unigram LM**: 基于概率模型的分词

## 序列标注任务
- **命名实体识别 (NER)**: 识别人名、地名、机构名等
- **词性标注 (POS Tagging)**: 标注每个词的词性
- **分词 (Segmentation)**: 中文 NLP 的基础步骤
- 常用模型: BiLSTM-CRF, BERT-CRF

## 文本分类
- **传统方法**: TF-IDF + SVM/朴素贝叶斯
- **深度方法**: TextCNN, TextRNN, HAN, BERT
- TextCNN: 多个卷积核提取 n-gram 特征
- 应用: 情感分析、垃圾邮件检测、主题分类

## 序列到序列 (Seq2Seq)
- **Encoder-Decoder 架构**
- **Encoder**: 将输入序列编码为上下文向量
- **Decoder**: 根据上下文向量生成输出序列
- **Attention**: 解决长序列信息瓶颈
- 应用: 机器翻译、文本摘要、对话系统

## 注意力机制 (Attention)
- **Bahdanau Attention**: 加性注意力，score = v^T tanh(W_h h + W_s s)
- **Luong Attention**: 乘性注意力，score = h^T W s
- **Self-Attention**: Q=K=V，关注自身不同位置
- 注意力本质: 加权求和，权重由相关性决定

## 评价指标
- **BLEU**: 机器翻译评价，基于 n-gram 精确率
- **ROUGE**: 文本摘要评价，基于召回率
- **Perplexity**: 语言模型评价，越小越好
- **F1 Score**: 分类任务，精确率和召回率的调和平均

## 预训练语言模型演进
Word2Vec(2013) → ELMo(2018) → BERT(2018) → GPT-2(2019) → BART/T5(2019) → GPT-3(2020) → ChatGPT(2022) → GPT-4(2023)

趋势: 模型越来越大，从静态词向量到动态上下文表示，从单向到双向，从单任务到多任务。
