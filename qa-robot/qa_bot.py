"""
智能问答机器人 — 主程序
========================
基于 RAG (检索增强生成) 架构。

流程:
  1. 用户提问 → 2. 检索相关文档 → 3. 提取答案 → 4. 返回结果

用法:
    python3 qa_bot.py                         # 交互模式
    python3 qa_bot.py --query "什么是Transformer?"  # 单次问答
    python3 qa_bot.py --api                   # API 服务模式

简历亮点:
  - RAG 全流程实现（检索→增强→生成）
  - 混合检索策略（稀疏 + 稠密）
  - 答案置信度评估
  - 可扩展的插件接口
"""

import argparse
import json
import os
import re
import sys
import time
from typing import List, Dict, Optional

import numpy as np


# ============================================================
# 知识库管理器
# ============================================================
class KnowledgeBase:
    """管理知识库文档的加载、分块"""

    def __init__(self, kb_dir: str = "./knowledge_base"):
        self.kb_dir = kb_dir

    def load_documents(self) -> List[Dict]:
        """加载并分块知识库文件"""
        if not os.path.exists(self.kb_dir):
            raise FileNotFoundError(f"知识库目录不存在: {self.kb_dir}")

        documents = []
        files = sorted([f for f in os.listdir(self.kb_dir) if f.endswith(".md")])

        if not files:
            raise FileNotFoundError(f"知识库中没有 .md 文件: {self.kb_dir}")

        for fname in files:
            filepath = os.path.join(self.kb_dir, fname)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # 按 ## 标题分块
            blocks = self._chunk_document(fname, content)
            # 过滤掉无内容的标题块
            blocks = [b for b in blocks if b and not b["id"].endswith("block_0")]
            documents.extend(blocks)

        print(f"📖 加载知识库: {len(files)} 个文件, {len(documents)} 个文档块")
        return documents

    def _chunk_document(self, fname: str, content: str) -> List[Dict]:
        """将文档按标题分块，每块作为一个检索单元"""
        lines = content.split("\n")
        blocks = []
        current_title = ""
        current_content = []
        block_id = 0

        for line in lines:
            if line.startswith("## "):
                # 保存上一个块
                if current_content:
                    blocks.append(self._make_block(
                        fname, block_id, current_title, current_content
                    ))
                    block_id += 1

                current_title = line[3:].strip()
                current_content = [line]
            elif line.startswith("# "):
                current_content.append(line)
            else:
                current_content.append(line)

        # 最后一个块
        if current_content:
            blocks.append(self._make_block(
                fname, block_id, current_title, current_content
            ))

        # 合并相邻小块，让每个块有足够内容
        merged = []
        pending = None
        for block in blocks:
            if block is None:
                continue
            is_tiny = self._block_size(block) < 300
            is_header_block = block["id"].endswith("block_0")

            if is_header_block:
                if pending:
                    merged.append(pending)
                    pending = None
                merged.append(block)
            elif is_tiny and merged:
                prev = merged[-1]
                if not prev["id"].endswith("block_0"):
                    prev["content"] += "\n\n" + block["content"]
                    prev["text"] = prev["content"]
                else:
                    merged.append(block)
            else:
                if pending:
                    merged.append(pending)
                pending = block

        if pending:
            merged.append(pending)

        return merged

    def _block_size(self, block: Dict) -> int:
        """块中非空行文本长度"""
        content = block.get("content", "")
        # 不算标题行的长度
        body = "\n".join(l for l in content.split("\n")
                         if l.strip() and not l.startswith("#"))
        return len(body)

    def _make_block(self, fname: str, block_id: int,
                    title: str, content_lines: List[str]) -> Dict:
        """创建一个文档块"""
        content = "\n".join(content_lines).strip()
        # 跳过纯分隔线块
        if not content or all(c in "-=" for c in content[:20]):
            return None

        doc_id = f"{fname}::block_{block_id}"
        return {
            "id": doc_id,
            "source": fname,
            "title": title or fname,
            "content": content,
            "text": content,  # 检索用文本
        }

    def get_statistics(self) -> Dict:
        """获取知识库统计信息"""
        docs = self.load_documents()
        total_chars = sum(len(d["content"]) for d in docs)
        return {
            "num_docs": len(docs),
            "total_chars": total_chars,
            "avg_doc_length": total_chars // len(docs) if docs else 0,
        }


# ============================================================
# 答案提取器 (Generator)
# ============================================================
class AnswerExtractor:
    """
    从检索到的文档中提取答案

    策略: 文档已按 ## 合并成主题块。从最佳匹配文档中，
    找出包含查询关键词最多的 ## 子节作为答案。
    """

    def __init__(self):
        pass

    def extract(self, query: str, documents: List[Dict],
                scores: List[float]) -> Dict:
        """从文档中提取答案"""
        if not documents:
            return {
                "answer": "抱歉，知识库中没有找到相关信息。请换一种问法试试。",
                "confidence": 0.0,
                "sources": [],
                "context": "",
            }

        best_doc = documents[0]
        best_score = scores[0] if scores else 0
        content = best_doc["content"]
        title = best_doc.get("title", "")

        # 按 ## 子节拆分，找到最佳匹配子节
        answer = self._extract_section(content, query, title)

        # 置信度评估
        confidence = self._estimate_confidence(query, answer, best_score)

        sources = [
            {"title": doc["title"], "source": doc["source"], "score": score}
            for doc, score in zip(documents, scores)
        ]

        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "context": content[:500] + ("..." if len(content) > 500 else ""),
        }

    def _extract_section(self, content: str, query: str, doc_title: str) -> str:
        """
        在文档中找最匹配查询的子节。

        按 ## 标题拆分，评分每个子节与查询的关键词重叠。
        """
        query_terms = set(tokenize(query))
        if not query_terms:
            # 返回文档第一段
            lines = [l for l in content.split("\n")
                     if l.strip() and not l.startswith("#")]
            return lines[0] if lines else content[:200]

        # 按 ## 拆分
        sections = re.split(r'\n(?=## )', content)
        scored_sections = []

        for i, sec in enumerate(sections):
            sec_title = ""
            sec_lines = sec.split("\n")
            if sec_lines[0].startswith("## "):
                sec_title = sec_lines[0][3:].strip()
            # 包含 query_terms 中多少比例的词
            sec_text = " ".join(l for l in sec_lines
                                if not l.startswith("#"))
            sec_terms = set(tokenize(sec_text))
            if sec_terms:
                overlap = len(query_terms & sec_terms)
                # 也考虑标题匹配
                title_terms = set(tokenize(sec_title))
                title_bonus = len(query_terms & title_terms) * 2
                score = overlap + title_bonus
                scored_sections.append((score, sec))

        if scored_sections:
            scored_sections.sort(key=lambda x: x[0], reverse=True)
            best_section = scored_sections[0][1]

            # 清理：移除语言切换行、分隔线，保留关键内容
            lines = best_section.split("\n")
            clean = [l for l in lines if l.strip() and
                     not l.startswith("---") and
                     not l.startswith("===") and
                     not all(c in "-=#* \t" for c in l)]

            # 移除 ## 标题前缀，转为粗体显示
            render = []
            for l in clean:
                if l.startswith("## "):
                    render.append(f"**{l[3:].strip()}**")
                elif l.startswith("# "):
                    render.append(f"**{l[2:].strip()}**")
                else:
                    render.append(l)

            # 限制输出长度
            answer = "\n".join(render[:30])
            if len(answer) > 1500:
                answer = answer[:1500] + "..."
            return answer

        # 兜底
        body = [l for l in content.split("\n")
                if l.strip() and not l.startswith("#")]
        return body[0] if body else content[:300]

    def _extract_relevant_section(self, query: str, content: str) -> str:
        """从文档内容中提取最相关的片段"""
        lines = content.split("\n")
        query_terms = set(tokenize(query))

        # 评分每行
        scored_lines = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line_terms = set(tokenize(line))
            overlap = len(query_terms & line_terms)
            if overlap > 0:
                # 带上下文（前后行）
                context_start = max(0, i - 1)
                context_end = min(len(lines), i + 3)
                context = "\n".join(lines[context_start:context_end])
                scored_lines.append((overlap, context))

        if not scored_lines:
            # 没有直接匹配，返回文档开头
            paragraphs = [l for l in lines if l.strip() and not l.startswith("#")]
            return paragraphs[0] if paragraphs else content[:200]

        # 返回最匹配的上下文
        scored_lines.sort(key=lambda x: x[0], reverse=True)
        return scored_lines[0][1]

    def _estimate_confidence(self, query: str, answer: str,
                             retrieval_score: float) -> float:
        """评估答案置信度 (0~1)"""
        if not answer:
            return 0.0

        query_terms = set(tokenize(query))
        answer_terms = set(tokenize(answer))

        if not query_terms:
            return 0.0

        # 关键词覆盖率
        overlap = len(query_terms & answer_terms)
        keyword_coverage = overlap / len(query_terms)

        # 答案长度适当（太短说明没找到好内容）
        length_score = min(1.0, len(answer) / 200)

        # 综合: 检索分数 + 关键词覆盖 + 长度
        confidence = (
            0.3 * min(1.0, retrieval_score / 2) +
            0.5 * keyword_coverage +
            0.2 * length_score
        )
        return float(min(1.0, max(0.0, confidence)))


def tokenize(text: str) -> List[str]:
    """简单分词"""
    text = text.lower()
    import re
    text = re.sub(r'([一-鿿])([a-zA-Z0-9])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z0-9])([一-鿿])', r'\1 \2', text)
    tokens = []
    for word in text.split():
        if re.match(r'^[一-鿿]+$', word):
            tokens.extend(list(word))
        else:
            tokens.append(word)
    return tokens


# ============================================================
# QA Bot 主类
# ============================================================
class QABot:
    """
    智能问答机器人

    完整 RAG 流程:
      1. Index: 知识库分块 → 构建混合检索索引
      2. Retrieve: 用户问题 → 检索 Top-K 相关文档
      3. Generate: 相关文档 → 提取/生成答案
      4. Present: 答案 + 来源 + 置信度
    """

    def __init__(self, kb_dir: str = "./knowledge_base",
                 index_path: str = "./models/index.pkl"):
        self.kb_dir = kb_dir
        self.index_path = index_path
        self.retriever = None
        self.extractor = AnswerExtractor()
        self.history = []
        self.ready = False

    def initialize(self, force_rebuild: bool = False):
        """初始化：加载或构建索引"""
        from retriever import HybridRetriever

        self.retriever = HybridRetriever(alpha=0.4)

        if not force_rebuild and os.path.exists(self.index_path):
            try:
                self.retriever.load(self.index_path)
                self.ready = True
                print("✅ QA Bot 初始化完成（加载已有索引）")
                return
            except Exception as e:
                print(f"  ⚠️ 索引加载失败: {e}，重新构建")

        print("🔄 构建索引...")
        kb = KnowledgeBase(self.kb_dir)
        documents = kb.load_documents()
        # 过滤 None
        documents = [d for d in documents if d is not None]
        self.retriever.build_index(documents)

        # 保存索引
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        self.retriever.save(self.index_path)
        self.ready = True

        print("✅ QA Bot 初始化完成（新建索引）")

    def ask(self, query: str, top_k: int = 3, verbose: bool = False) -> Dict:
        """
        提问主入口

        Args:
            query: 用户问题
            top_k: 检索文档数量
            verbose: 是否打印详细信息

        Returns:
            {
                "answer": str,
                "confidence": float,
                "sources": [...],
                "context": str,
                "time_taken": float,
            }
        """
        if not self.ready:
            raise RuntimeError("Bot 未初始化，请先调用 initialize()")

        start_time = time.time()

        # Step 1: 检索
        if verbose:
            print("\n🔍 检索中...")
        results = self.retriever.retrieve(query, top_k=top_k)

        if not results:
            return {
                "answer": "抱歉，知识库中没有找到相关信息。",
                "confidence": 0.0,
                "sources": [],
                "context": "",
                "time_taken": time.time() - start_time,
            }

        # Step 2: 提取答案
        docs = [r[0] for r in results]
        scores = [r[1] for r in results]

        extracted = self.extractor.extract(query, docs, scores)

        elapsed = time.time() - start_time

        result = {
            **extracted,
            "time_taken": elapsed,
        }

        # 记录历史
        self.history.append({
            "query": query,
            "answer": result["answer"],
            "confidence": result["confidence"],
        })

        return result

    def ask_stream(self, query: str, top_k: int = 3):
        """
        流式输出（模拟生成效果）
        """
        result = self.ask(query, top_k=top_k)
        answer = result["answer"]
        # 逐字输出
        for char in answer:
            print(char, end="", flush=True)
            time.sleep(0.02)
        print()
        return result


# ============================================================
# CLI 界面
# ============================================================
class CLI:
    """命令行交互界面"""

    def __init__(self, bot: QABot):
        self.bot = bot

    def run(self):
        """启动交互式问答"""
        print("=" * 60)
        print("🤖 智能问答机器人 (RAG)")
        print("    基于: PyTorch + 混合检索 (TF-IDF + Dense)")
        print("    知识库: AI / NLP / 深度学习面试知识点")
        print("-" * 60)
        print("  输入问题开始问答，输入以下命令:")
        print("    /history  — 查看历史记录")
        print("    /stats    — 查看知识库统计")
        print("    /mode     — 切换输出模式")
        print("    q         — 退出")
        print("=" * 60)

        stream_mode = False

        while True:
            try:
                query = input("\n💡 你的问题: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 再见！")
                break

            if not query:
                continue

            if query.lower() == "q":
                print("👋 再见！")
                break

            if query == "/history":
                self._show_history()
                continue

            if query == "/stats":
                self._show_stats()
                continue

            if query == "/mode":
                stream_mode = not stream_mode
                print(f"📢 输出模式: {'流式' if stream_mode else '标准'}")
                continue

            # 问答
            print("\n🤔 思考中...", end="\r")

            if stream_mode:
                result = self.bot.ask_stream(query)
            else:
                result = self.bot.ask(query, verbose=False)

            self._show_result(result)

    def _show_result(self, result: Dict):
        """显示答案"""
        answer = result["answer"]
        confidence = result["confidence"]
        sources = result.get("sources", [])

        # 置信度指示器
        if confidence >= 0.7:
            indicator = "🟢 高置信度"
        elif confidence >= 0.4:
            indicator = "🟡 中等置信度"
        else:
            indicator = "🔴 低置信度"

        print(f"\n🤖 回答: {answer}")
        print(f"📊 置信度: {confidence:.1%} {indicator}")
        print(f"⚡ 耗时: {result.get('time_taken', 0):.2f}s")

        if sources:
            print(f"📚 参考来源 ({len(sources)} 篇):")
            for src in sources[:3]:
                print(f"   · {src['title']} (score={src['score']:.3f})")

    def _show_history(self):
        """显示问答历史"""
        if not self.bot.history:
            print("📭 暂无历史记录")
            return
        print(f"\n📋 问答历史 ({len(self.bot.history)} 条):")
        for i, h in enumerate(self.bot.history[-10:], 1):
            print(f"  {i}. Q: {h['query'][:50]}")
            print(f"     A: {h['answer'][:60]}...")

    def _show_stats(self):
        """显示知识库统计"""
        try:
            kb = KnowledgeBase(self.bot.kb_dir)
            stats = kb.get_statistics()
            print(f"\n📊 知识库统计:")
            print(f"   文档数: {stats['num_docs']}")
            print(f"   总字数: {stats['total_chars']:,}")
            print(f"   平均长度: {stats['avg_doc_length']} 字/篇")
            print(f"   检索模式: 混合检索 (TF-IDF + Dense)")
        except Exception as e:
            print(f"❌ 获取统计失败: {e}")


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="🤖 智能问答机器人")
    parser.add_argument("--query", "-q", type=str, help="单次提问模式")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引")
    parser.add_argument("--kb", type=str, default="./knowledge_base",
                        help="知识库路径")
    parser.add_argument("--topk", type=int, default=3,
                        help="检索文档数量")
    parser.add_argument("--no-color", action="store_true",
                        help="禁用颜色输出（暂未使用）")
    args = parser.parse_args()

    # 初始化
    bot = QABot(kb_dir=args.kb)
    try:
        bot.initialize(force_rebuild=args.rebuild)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)

    # 单次提问
    if args.query:
        result = bot.ask(args.query, top_k=args.topk, verbose=True)
        print(f"\n🤖 {result['answer']}")
        print(f"📊 置信度: {result['confidence']:.1%}")
        print(f"⚡ 耗时: {result['time_taken']:.2f}s")
        if result.get("sources"):
            print("📚 来源:")
            for s in result["sources"][:3]:
                print(f"   · {s['title']} (score={s['score']:.3f})")
        return

    # 交互模式
    cli = CLI(bot)
    cli.run()


if __name__ == "__main__":
    main()
