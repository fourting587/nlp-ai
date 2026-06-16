# 智能问答机器人 (RAG-based Q&A Bot)

基于 **RAG (Retrieval-Augmented Generation)** 架构的中文智能问答系统。
可回答深度学习、NLP、Transformer、PyTorch 等技术面试问题。

## 项目亮点

- 🧠 **RAG 全流程** — 检索 → 增强 → 生成，工业级架构
- 🔍 **混合检索策略** — TF-IDF（精确匹配）+ Dense Embedding（语义匹配）加权融合
- 📚 **内置知识库** — 4 篇 AI/NLP 面试知识点文档（45+ 小节）
- 💻 **三种使用模式** — 单次问答、批量、交互式 CLI
- 🚫 **完全离线** — 无需下载任何模型或数据

## 技术栈

| 模块 | 技术 |
|------|------|
| 检索器 | TF-IDF (从零实现) + 字符级 Dense Embedding |
| 融合策略 | Hybrid 加权混合检索 |
| 知识库 | Markdown 文档 → 语义分块 → 向量索引 |
| 答案提取 | 相关性评分 + 最佳子节匹配 |
| 序列化 | Pickle 索引持久化 |

## 快速开始

```bash
# 1. 构建索引
python3 build_index.py

# 2. 交互式问答
python3 qa_bot.py

# 3. 单次提问
python3 qa_bot.py --query "什么是Transformer？"
# → Transformer 是 2017 年 Google 提出的"Attention Is All You Need"中的架构...

python3 qa_bot.py --query "过拟合怎么解决"
# → 解决方法: 增加数据、正则化(L1/L2)、Dropout、早停、数据增强

python3 qa_bot.py --query "LSTM怎么解决梯度消失"
# → LSTM 解决: 遗忘门 + 输入门 + 输出门
```

## 交互模式功能

```
💡 你的问题: 什么是自注意力机制？
🤔 思考中...
🤖 回答: 每个头关注不同的关系模式（语法、语义、位置等）。
📊 置信度: 38.9% 🟡 中等置信度

内置命令:
  /history  查看历史记录
  /stats    查看知识库统计
  /mode     切换流式输出
  q         退出
```

## 项目结构

```
qa-robot/
├── qa_bot.py           # 主程序 (RAG 流程 + CLI)
├── retriever.py        # 检索模块 (TF-IDF + Dense + Hybrid)
├── build_index.py      # 索引构建工具
├── knowledge_base/     # 知识库 (Markdown)
│   ├── 01_deep_learning.md
│   ├── 02_transformer.md
│   ├── 03_nlp_basics.md
│   └── 04_pytorch.md
├── models/             # 索引文件 (gitignored)
│   └── index.pkl
└── README.md
```

## RAG 架构详解

```
用户提问 "什么是Transformer？"
        │
        ▼
  ┌─────────────────┐
  │  1. 检索阶段     │  HybridRetriever
  │  - TF-IDF 检索   │  关键词匹配 "Transformer" "注意力"
  │  - Dense 检索    │  语义匹配 "自注意力" "Attention"
  │  - 分数融合      │  score = α * tfidf + (1-α) * dense
  └────────┬────────┘
           │ Top-3 文档
           ▼
  ┌─────────────────┐
  │  2. 增强阶段     │  AnswerExtractor
  │  - 选择最佳文档  │  最高分文档
  │  - 定位最佳子节  │  按 ## 标题评分
  │  - 提取相关内容  │  移除分隔线/格式化
  └────────┬────────┘
           │ 格式化答案
           ▼
  ┌─────────────────┐
  │  3. 生成阶段     │  展示结果
  │  - 答案          │  Transformer 是 2017 年...
  │  - 置信度        │  34.6%
  │  - 来源          │  02_transformer.md
  └─────────────────┘
```

## 扩展方向

- [ ] **接入 LLM** — 用 Claude/GPT 替换 AnswerExtractor，生成更自然的回答
- [ ] **更多知识库** — 添加 PyTorch 实战、LeetCode 题解、项目经验文档
- [ ] **Web 界面** — 用 Gradio/Streamlit 搭建网页 Demo
- [ ] **增量更新** — 知识库新增文档时增量构建索引
- [ ] **语义缓存** — 缓存常见问题的答案
