# 中文情感分析 (Chinese Sentiment Analysis)

基于 **PyTorch** 从零实现的中文文本情感二分类系统。

本项目使用 **TextCNN** 模型对中文评论进行正面/负面情感判断，完全离线可用，
无需下载任何预训练模型。

## 技术亮点

- 🧠 **TextCNN 模型** — 多尺寸卷积核提取 n-gram 特征（2-gram 到 5-gram）
- 🔤 **字符级词表** — 从训练数据自动构建，无需外部词向量
- 📦 **完全离线** — 所有代码和数据都包含在项目中，clone 即用
- 🔍 **三种推理模式** — 单条、批量、交互式
- 📊 **完整评估** — 准确率 + 梯度裁剪 + 学习率余弦退火

## 技术栈

| 模块 | 技术 |
|------|------|
| 深度学习框架 | PyTorch |
| 模型架构 | TextCNN (Kim, 2014) |
| 训练方式 | 从零训练，自动构建词表 |
| 数据增强 | 梯度裁剪、CosineAnnealingLR |
| 数据集 | ChnSentiCorp 酒店评论子集（96 条训练 / 12 条测试） |

## 快速开始

```bash
# 1. 训练模型
python3 train.py

# 输出示例:
# Epoch  1/20 | Loss=0.3671 | Test Acc=66.7%
# Epoch 10/20 | Loss=0.0093 | Test Acc=91.7%
# ...
# 🎉 训练完成！最佳准确率: 91.7%

# 2. 预测
python3 predict.py --text "环境很好，服务也很周到"
# → 正面 😊 (置信度: 99.8%)

python3 predict.py --text "设施太旧了，隔音很差"
# → 负面 😞 (置信度: 98.5%)

# 3. 交互模式（推荐用来玩）
python3 predict.py --interactive

# 4. 批量分析
python3 predict.py --file reviews.txt
```

## 训练效果

在 96 条训练 / 12 条测试的数据上，**TextCNN** 约 10 个 epoch 可达 **90%+ 准确率**。

## 模型切换：TextCNN → BERT

有网络时可一键升级到 BERT。在 `train.py` 顶部把 `USE_BERT` 改为 `True`：

```python
USE_BERT = True   # 需要网络下载 bert-base-chinese
```

BERT 版本会自动下载预训练模型并微调，准确率可提升到 95%+。

## 项目结构

```
sentiment-analysis/
├── train.py          # 训练脚本（TextCNN + BERT 两种模式）
├── predict.py        # 预测脚本（单条/批量/交互）
├── models/           # 训练好的模型 (gitignored)
│   └── best.pt
├── .gitignore
└── README.md
```

## 面试考点

| 问题 | 答案要点 |
|------|----------|
| TextCNN 如何捕捉 n-gram 特征？ | 不同大小的卷积核（2,3,4,5）→ 不同窗口大小 |
| 为什么用 max pooling？ | 提取每个 feature map 最显著的特征 |
| 梯度裁剪的作用？ | 防止梯度爆炸，稳定训练 |
| 如何处理变长文本？ | padding + attention mask |
| 怎么预防过拟合？ | Dropout + Weight Decay + 早停 |
| 为什么字符级而不是词级？ | OOV 问题少，适合中文（词边界模糊） |
| 如何升级到 BERT？ | 替换 embedding 层为预训练模型，微调最后几层 |

## 扩展方向

- [ ] **升级到 BERT** — 有网络时切换到预训练微调
- [ ] **三分类** — 增加中性类别（正面/中性/负面）
- [ ] **多模型对比** — TextCNN vs BERT vs LSTM
- [ ] **Web 界面** — 用 Gradio 或 Streamlit 搭建 Demo
- [ ] **更多数据** — 接入 S 万级真实评论数据
