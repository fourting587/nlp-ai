"""
🌐 Web Agent — 联网搜索代理
=============================
通过 DuckDuckGo 搜索互联网，获取实时信息并返回答案。
不受知识库限制，可回答任何领域的问题。

用法:
    from web_agent import WebAgent
    agent = WebAgent()
    result = agent.ask("今天天气怎么样？")
"""

import time
import re
from typing import List, Dict, Optional


class WebAgent:
    """
    联网搜索代理

    当用户的问题超出知识库范围，或需要实时信息时，
    自动搜索互联网获取答案。
    """

    def __init__(self, max_results: int = 5, max_snippets: int = 3):
        self.max_results = max_results
        self.max_snippets = max_snippets
        self._searcher = None
        self.history = []

    @property
    def _ddgs(self):
        """延迟初始化 DDGS"""
        if self._searcher is None:
            from ddgs import DDGS
            self._searcher = DDGS()
        return self._searcher

    def search(self, query: str, max_results: int = None) -> List[Dict]:
        """
        搜索互联网

        Args:
            query: 搜索关键词
            max_results: 最大结果数

        Returns:
            [{"title": ..., "href": ..., "body": ...}, ...]
        """
        k = max_results or self.max_results
        try:
            results = list(self._ddgs.text(query, max_results=k))
            return results
        except Exception as e:
            print(f"  ⚠️ 搜索失败: {e}")
            # 重试一次
            try:
                self._searcher = None  # 重置连接
                results = list(self._ddgs.text(query, max_results=k))
                return results
            except:
                return []

    def fetch_content(self, url: str, timeout: int = 5) -> Optional[str]:
        """获取网页内容"""
        try:
            import httpx
            headers = {
                "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36")
            }
            resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
            if resp.status_code == 200:
                text = resp.text
                # 提取纯文本（简单去标签）
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text[:2000]  # 限制长度
        except:
            return None

    def _clean_text(self, text: str) -> str:
        """清理 HTML 文本，提取可读内容"""
        # 移除脚本和样式
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        # 标签转空格
        text = re.sub(r'<[^>]+>', ' ', text)
        # 清理多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        # 移除纯乱码/噪音（连续无中文的片段）
        lines = text.split('\n')
        clean = []
        for line in lines:
            line = line.strip()
            if len(line) < 15:
                continue
            # 如果包含中文或较多英文字词则保留
            if re.search(r'[一-鿿]', line) or len(line.split()) > 5:
                clean.append(line)
        return '\n'.join(clean[:30])

    def ask(self, query: str, verbose: bool = False) -> Dict:
        """
        联网搜索并返回答案

        流程:
          1. 用 query 搜索互联网
          2. 取前 N 条结果
          3. 合成答案（标题 + 摘要）
          4. 附上来源链接

        Args:
            query: 用户问题
            verbose: 是否打印详细信息

        Returns:
            {
                "answer": str,
                "sources": [{"title": ..., "url": ..., "snippet": ...}],
                "time_taken": float,
                "source_count": int,
            }
        """
        start = time.time()

        # 1. 搜索
        results = self.search(query)

        if not results:
            elapsed = time.time() - start
            return {
                "answer": "抱歉，联网搜索没有找到相关信息。请换一种问法试试。",
                "sources": [],
                "time_taken": elapsed,
                "source_count": 0,
            }

        # 2. 合成答案
        answer_parts = []
        sources = []

        for i, r in enumerate(results[:self.max_snippets]):
            title = r.get("title", "") or ""
            body = r.get("body", "") or ""
            url = r.get("href", "") or ""

            # 过滤噪音结果
            if not title or len(title) < 2:
                continue
            noise_keywords = ["百度地图", "必应", "map.baidu", "javascript"]
            if any(k in title.lower() for k in noise_keywords):
                continue

            if title and body:
                answer_parts.append(f"**{title}**\n{body[:200]}")
            elif body:
                answer_parts.append(body[:200])

            sources.append({
                "title": title,
                "url": url,
                "snippet": body[:150] if body else "",
            })

        # 3. 组合答案
        if answer_parts:
            answer = "\n\n".join(answer_parts)
        else:
            # 没有干净的结果，用原始结果
            for r in results[:2]:
                t = r.get("title", "")
                b = r.get("body", "")
                if t and b:
                    answer_parts.append(f"**{t}**\n{b[:200]}")
            answer = "\n\n".join(answer_parts) if answer_parts else "搜索结果中没有找到有效内容。"

        elapsed = time.time() - start

        result = {
            "answer": answer,
            "sources": sources,
            "time_taken": elapsed,
            "source_count": len(sources),
        }

        # 记录历史
        self.history.append({
            "query": query,
            "result_count": len(sources),
            "time_taken": elapsed,
        })

        return result

    def ask_with_summary(self, query: str, verbose: bool = False) -> Dict:
        """
        搜索并返回更简洁的摘要答案

        会尝试抓取第一条结果的内容来丰富答案
        """
        result = self.ask(query, verbose=verbose)

        if result["source_count"] > 0 and result["sources"][0]["url"]:
            # 尝试获取第一条结果的详情
            url = result["sources"][0]["url"]
            content = self.fetch_content(url)
            if content and len(content) > 200:
                # 提取与查询相关的内容
                query_words = set(query.lower())
                sentences = re.split(r'[。！？\n]', content)
                relevant = [
                    s.strip() for s in sentences
                    if any(w in s.lower() for w in query_words) and len(s.strip()) > 20
                ]
                if relevant:
                    summary = "。".join(relevant[:3]) + "。"
                    # 替换答案前加详细内容
                    result["answer"] = f"{summary}\n\n---\n\n{result['answer']}"

        return result

    @property
    def stats(self) -> Dict:
        """使用统计"""
        if not self.history:
            return {"total_searches": 0}
        return {
            "total_searches": len(self.history),
            "avg_time": sum(h["time_taken"] for h in self.history) / len(self.history),
        }


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    agent = WebAgent()
    q = "2026年世界杯在哪里举办"
    print(f"❓ {q}")
    result = agent.ask_with_summary(q, verbose=True)
    print(f"\n🤖 {result['answer']}")
    print(f"\n📚 来源 ({result['source_count']} 个):")
    for s in result["sources"][:3]:
        print(f"   · {s['title']}")
        print(f"     {s['url']}")
    print(f"\n⚡ 耗时: {result['time_taken']:.2f}s")
