# 智能问答机器人：基于 RAG 的检索增强生成系统

> **Python · TF-IDF · Sentence-Transformer · Gradio · Claude API**

独立设计和实现的端到端 RAG 智能问答系统，支持**知识库问答**和**联网搜索**两种模式。包含完整的索引构建、混合检索、答案生成、语义缓存链路，配有 Gradio Web 界面。

---

## ✨ 项目亮点（面向简历）

### 核心技术

| 模块 | 实现 |
|------|------|
| **混合检索** | 从零手写 TF-IDF 稀疏检索 + Dense Embedding 语义检索（Sentence-Transformer），权重 α 融合，兼顾精确匹配与语义相似度 |
| **RAG 全流程** | 文档按标题分块 → 双路索引构建 → 混合检索 Top-K → 子节级答案提取（关键词重叠评分 + 置信度评估）→ 生成答案 |
| **语义缓存** | 基于嵌入向量 + 余弦相似度的缓存机制，阈值 0.85 下相似问题直接命中，响应时间从 ~2s 降至毫秒级 |
| **联网搜索** | 集成 DuckDuckGo Search API，支持知识库 / 联网双模式自由切换 |
| **LLM 增强** | 接入 Claude API 增强生成质量，API 不可用时自动回退到本地提取模式 |
| **Gradio Web UI** | 响应式聊天界面，支持模式切换、示例问题、LLM 开关、公网链接分享 |
| **增量索引** | 基于文档 ID 检测的增量构建 + 文件系统监听模式（--watch 每 30s），无需全量重建 |

### 知识库

构建了 **7 篇 AI/NLP 面试知识库文档**（深度学习、Transformer、NLP 基础、PyTorch、ML 算法、Python 编程、面试准备），覆盖 ~70+ 知识点。

### 架构设计

```
用户提问 → [语义缓存?] → 混合检索 → LLM/本地生成 → 返回 + 缓存
            ↑               ↑
       命中直接返回    TF-IDF + Dense
```

代码五模块解耦：`qa_bot.py` / `retriever.py` / `build_index.py` / `app.py` / `web_agent.py`

---

## 🚀 快速开始

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
python3 qa_bot.py                  # 知识库模式（AI/NLP 面试问答）
python3 qa_bot.py --web            # 联网搜索模式（回答任意问题）
```

启动后输入 `/web` 可在两种模式间切换。

内置命令:
| 命令 | 功能 |
|------|------|
| `/web` | 切换知识库/联网搜索模式 |
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
python3 app.py                     # 知识库模式
python3 app.py --web               # 启动时默认开启联网搜索
python3 app.py --port 8080         # 自定义端口
python3 app.py --share             # 生成公网链接
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
├── web_agent.py        # 联网搜索模块
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
