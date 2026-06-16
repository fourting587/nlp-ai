# 智能问答机器人 (RAG-based Q&A Bot)

基于 **RAG (Retrieval-Augmented Generation)** 架构的中文智能问答系统。
可回答深度学习、NLP、Transformer、PyTorch 等技术面试问题。

## 项目亮点

- 🧠 **RAG 全流程** — 检索 → 增强 → 生成，工业级架构
- 🔍 **混合检索** — TF-IDF（精确匹配）+ Dense Embedding（语义匹配）加权融合
- 🤖 **LLM 接入** — 可选 Claude API 生成更自然的回答
- 💾 **语义缓存** — 相似问题自动缓存命中，秒级响应
- 🌐 **Web 界面** — Gradio 聊天 UI，支持示例问题和 LLM 切换
- 📚 **7 篇知识库** — 深度学习、Transformer、NLP、PyTorch、ML 算法、LeetCode、面试准备
- 🔄 **增量更新** — 知识库新增文档时自动检测并更新索引
- 🚫 **完全离线** — 基础模式无需下载任何模型或数据

## 快速开始

```bash
# 1. 构建索引（含 7 篇知识库文档）
python3 build_index.py

# 2. 交互式 CLI 问答
python3 qa_bot.py

# 3. Web 界面
python3 app.py
# → 浏览器打开 http://127.0.0.1:7860
```

## 使用方式

### CLI 交互模式

```bash
python3 qa_bot.py
```

内置命令:
| 命令 | 功能 |
|------|------|
| `/history` | 查看问答历史 |
| `/stats` | 知识库统计 |
| `/cache` | 缓存状态 |
| `/clearcache` | 清空缓存 |
| `/mode` | 切换流式输出 |
| `q` | 退出 |

### 单次问答

```bash
python3 qa_bot.py --query "什么是Transformer？"
```

### LLM 增强模式

```bash
# 需要设置 ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY=sk-ant-...
python3 qa_bot.py --llm --query "什么是Transformer？"
python3 app.py --llm    # Web 界面启用 LLM
```

### Web 界面

```bash
python3 app.py                # 默认 7860 端口
python3 app.py --port 8080    # 自定义端口
python3 app.py --share        # 生成公网链接
```

### 增量更新

```bash
python3 build_index.py --incremental    # 检测并添加新文档
python3 build_index.py --watch          # 持续监听变化（每 30s）
python3 build_index.py --stats          # 查看知识库统计
```

## 完整功能

### 语义缓存

```
第一次提问: "什么是Transformer？"  → 检索 + 生成  → 存入缓存
第二次提问: "什么是Transformer"   → 缓存命中! → 直接返回 (相似度: 0.95)
第三次提问: "Transformer是什么"   → 缓存命中! → 直接返回 (相似度: 0.88)
```

缓存阈值可调（默认 0.85），同一交互会话内生效。

### LLM 接入架构

```
用户提问 → 语义缓存? → 混合检索 → LLM (Claude) 生成答案 → 返回 + 缓存
                    ↑                          ↑
              命中直接返回              失败回退本地提取
```

### 知识库文档

| 文件 | 内容 | 小节数 |
|------|------|--------|
| `01_deep_learning.md` | 深度学习基础、激活函数、优化器 | ~10 |
| `02_transformer.md` | Transformer、Attention、BERT、GPT | ~12 |
| `03_nlp_basics.md` | NLP 基础、文本表示、预训练模型 | ~9 |
| `04_pytorch.md` | PyTorch 面试、训练技巧、手写题 | ~12 |
| `05_ml_algorithms.md` | 机器学习算法、评估指标、特征工程 | ~10 |
| `06_python_coding.md` | Python 编程、LeetCode 高频题、排序 | ~10 |
| `07_interview_prep.md` | 面试流程、简历模板、高频问题 | ~8 |

## 项目结构

```
qa-robot/
├── qa_bot.py           # 主程序 (RAG + 缓存 + LLM)
├── retriever.py        # 检索模块 (TF-IDF + Dense + Hybrid)
├── build_index.py      # 索引构建 (全量/增量/监听)
├── app.py              # Gradio Web 界面
├── knowledge_base/     # 知识库 (7 篇 Markdown 文档)
│   ├── 01_deep_learning.md
│   ├── 02_transformer.md
│   ├── 03_nlp_basics.md
│   ├── 04_pytorch.md
│   ├── 05_ml_algorithms.md
│   ├── 06_python_coding.md
│   └── 07_interview_prep.md
├── models/             # 索引文件 (gitignored)
│   └── index.pkl
└── README.md
```

## 面试考点

| 问题 | 答案要点 |
|------|----------|
| 什么是 RAG？ | 检索增强生成，先检索相关文档，再基于文档生成答案 |
| TF-IDF 原理？ | TF = 词频, IDF = log(总文档数/含词文档数), TF-IDF = TF × IDF |
| 混合检索怎么融合？ | α × TF-IDF_score + (1-α) × Dense_score |
| 增量更新怎么做？ | 检测文档 hash 变化，只重建新增部分的索引 |
| 语义缓存怎么实现？ | 问题向量化 + 余弦相似度 + 阈值判定 |
| 如果知识库没有答案？ | 返回"未找到相关信息"，或靠 LLM 知识补充 |
