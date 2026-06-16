"""
检索模块 (Retriever)
=====================
实现三种检索方法：
  1. TF-IDF (稀疏检索) — 关键词匹配
  2. Dense Embedding (稠密检索) — 语义匹配
  3. Hybrid (混合检索) — 加权融合

用法:
    from retriever import HybridRetriever
    retriever = HybridRetriever()
    retriever.build_index(documents)
    results = retriever.retrieve("什么是Transformer?", top_k=3)
"""

import math
import json
import os
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np


# ------------------------------------------------------------
# 文本预处理
# ------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """中文分词：按字符 + 英文按空格"""
    # 中文按字分割，英文按空格
    text = text.lower()
    # 在中文和英文/数字之间加空格
    text = re.sub(r'([一-鿿])([a-zA-Z0-9])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z0-9])([一-鿿])', r'\1 \2', text)
    # 分词
    tokens = []
    for word in text.split():
        if re.match(r'^[一-鿿]+$', word):
            # 中文词按字切分
            tokens.extend(list(word))
        else:
            tokens.append(word)
    return tokens


# ------------------------------------------------------------
# TF-IDF 检索 (从零实现)
# ------------------------------------------------------------
class TfidfRetriever:
    """
    TF-IDF 稀疏检索器

    原理:
        TF: 词在文档中出现频率
        IDF: log(总文档数 / 包含该词的文档数)
        TF-IDF = TF * IDF
    """

    def __init__(self):
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.vocab: Dict[str, int] = {}       # 词 -> id
        self.idf: np.ndarray = None            # (vocab_size,)
        self.tfidf_matrix: np.ndarray = None   # (n_docs, vocab_size)
        self.ready = False

    def build_index(self, documents: List[Dict], texts: List[str]):
        """
        构建 TF-IDF 索引

        Args:
            documents: 文档元数据列表 [{"id": ..., "title": ..., "content": ...}]
            texts: 每篇文档的文本（用于检索的文本）
        """
        self.documents = documents
        self.doc_ids = [d.get("id", str(i)) for i, d in enumerate(documents)]

        # 分词
        tokenized = [tokenize(t) for t in texts]

        # 构建词表
        word_doc_count = Counter()
        doc_word_counts = []
        for tokens in tokenized:
            counter = Counter(tokens)
            doc_word_counts.append(counter)
            for word in counter:
                word_doc_count[word] += 1

        # 过滤低频词
        self.vocab = {}
        for word in word_doc_count:
            self.vocab[word] = len(self.vocab)

        vocab_size = len(self.vocab)
        n_docs = len(documents)

        # 计算 IDF
        self.idf = np.zeros(vocab_size)
        for word, idx in self.vocab.items():
            df = word_doc_count[word]
            self.idf[idx] = math.log((n_docs + 1) / (df + 1)) + 1

        # 构建 TF-IDF 矩阵
        self.tfidf_matrix = np.zeros((n_docs, vocab_size))
        for doc_idx, counter in enumerate(doc_word_counts):
            if not counter:
                continue
            max_tf = max(counter.values())
            for word, count in counter.items():
                if word in self.vocab:
                    word_idx = self.vocab[word]
                    tf = count / max_tf
                    self.tfidf_matrix[doc_idx, word_idx] = tf * self.idf[word_idx]

        self.ready = True
        print(f"  [TF-IDF] 索引完成: {n_docs} 篇文档, {vocab_size} 个词")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """检索最相关的文档"""
        if not self.ready:
            raise RuntimeError("索引未构建，请先调用 build_index()")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # 查询向量
        query_counter = Counter(query_tokens)
        query_vec = np.zeros(len(self.vocab))
        max_tf = max(query_counter.values())
        for word, count in query_counter.items():
            if word in self.vocab:
                idx = self.vocab[word]
                tf = count / max_tf
                query_vec[idx] = tf * self.idf[idx]

        # 余弦相似度
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        query_vec = query_vec / query_norm
        doc_norms = np.linalg.norm(self.tfidf_matrix, axis=1)
        doc_norms[doc_norms == 0] = 1
        normalized_matrix = self.tfidf_matrix / doc_norms[:, np.newaxis]

        scores = normalized_matrix @ query_vec

        # 取 Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.documents[idx], float(scores[idx])))

        return results


# ------------------------------------------------------------
# Dense Embedding 检索 (使用简单词嵌入)
# ------------------------------------------------------------
class DenseRetriever:
    """
    稠密检索 — 用简单词嵌入表示语义

    实现: 基于字符共现的语义嵌入（无需外部预训练模型）
    每个词学习一个 embedding，文档向量 = 词向量均值
    """

    def __init__(self, embed_dim: int = 128):
        self.embed_dim = embed_dim
        self.documents: List[Dict] = []
        self.doc_embeddings: np.ndarray = None
        self.ready = False

    def build_index(self, documents: List[Dict], texts: List[str]):
        """
        构建稠密索引

        使用字符级 n-gram 统计生成语义嵌入：
        不需要预训练模型，完全基于文档自身的统计信息
        """
        self.documents = documents
        n_docs = len(documents)

        tokenized = [tokenize(t) for t in texts]

        # 构建字符级词表 + 共现矩阵
        all_chars = set()
        for tokens in tokenized:
            for token in tokens:
                for c in token:
                    all_chars.add(c)

        char_list = sorted(all_chars)
        char_to_idx = {c: i for i, c in enumerate(char_list)}
        n_chars = len(char_list)

        # 用字符 n-gram 作为语义特征
        doc_vectors = []
        for tokens in tokenized:
            vector = np.zeros(n_chars)
            for token in tokens:
                for c in token:
                    if c in char_to_idx:
                        vector[char_to_idx[c]] += 1
            # 归一化
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            doc_vectors.append(vector)

        self.doc_embeddings = np.array(doc_vectors)

        # 降维到 embed_dim (SVD)，保存投影矩阵
        self.svd_Vt = None
        self.svd_S = None
        if n_docs > 1 and n_chars > self.embed_dim:
            from numpy.linalg import svd
            U, S, Vt = svd(self.doc_embeddings, full_matrices=False)
            # 保留 top embed_dim 维度
            self.doc_embeddings = U[:, :self.embed_dim] * S[:self.embed_dim]
            self.svd_Vt = Vt[:self.embed_dim, :]  # 用于投影查询向量
            self.svd_S = S[:self.embed_dim]

        self.char_to_idx = char_to_idx
        self.n_chars = n_chars
        self.ready = True
        print(f"  [Dense] 索引完成: {n_docs} 篇文档, "
              f"embed_dim={self.doc_embeddings.shape[1]}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """检索最相关的文档"""
        if not self.ready:
            raise RuntimeError("索引未构建")

        # 查询向量化（字符级 bag-of-chars）
        query_tokens = tokenize(query)
        query_vec = np.zeros(self.n_chars)
        for token in query_tokens:
            for c in token:
                if c in self.char_to_idx:
                    query_vec[self.char_to_idx[c]] += 1

        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        query_vec = query_vec / query_norm

        # 如果存储了 SVD 投影矩阵，对查询向量做相同降维
        if self.svd_Vt is not None:
            query_vec = self.svd_Vt @ query_vec
            # 缩放（保持与文档向量一致的尺度）
            query_vec = query_vec * self.svd_S

        # 余弦相似度
        doc_norms = np.linalg.norm(self.doc_embeddings, axis=1)
        doc_norms[doc_norms == 0] = 1
        normalized = self.doc_embeddings / doc_norms[:, np.newaxis]
        scores = normalized @ query_vec

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.documents[idx], float(scores[idx])))

        return results


# ------------------------------------------------------------
# 混合检索
# ------------------------------------------------------------
class HybridRetriever:
    """
    混合检索器 — TF-IDF + Dense Embedding 加权融合

    取长补短：
      - TF-IDF: 精确关键词匹配（召回精确结果）
      - Dense: 语义相似度匹配（召回同义词/近义表达）
    """

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: TF-IDF 权重 (0~1)，1=只TF-IDF，0=只Dense
        """
        self.alpha = alpha
        self.tfidf = TfidfRetriever()
        self.dense = DenseRetriever()
        self.documents = []
        self.ready = False

    def build_index(self, documents: List[Dict]):
        """
        构建索引

        Args:
            documents: [{"id": ..., "title": ..., "content": ..., "text": ...}]
        """
        self.documents = documents
        texts = [d.get("text", d.get("content", "")) for d in documents]

        print("📚 构建检索索引...")
        self.tfidf.build_index(documents, texts)
        self.dense.build_index(documents, texts)
        self.ready = True

    def retrieve(self, query: str, top_k: int = 5,
                 alpha: Optional[float] = None) -> List[Tuple[Dict, float]]:
        """
        混合检索

        Returns:
            [(document, score), ...] 按相关性降序
        """
        if not self.ready:
            raise RuntimeError("索引未构建")

        a = alpha if alpha is not None else self.alpha

        # 分别检索取更多候选
        tfidf_results = self.tfidf.retrieve(query, top_k=top_k * 2)
        dense_results = self.dense.retrieve(query, top_k=top_k * 2)

        # 融合分数
        all_scores = {}
        for doc, score in tfidf_results:
            doc_id = doc.get("id", "")
            all_scores[doc_id] = all_scores.get(doc_id, 0) + a * score

        for doc, score in dense_results:
            doc_id = doc.get("id", "")
            all_scores[doc_id] = all_scores.get(doc_id, 0) + (1 - a) * score

        # 排序
        doc_map = {d.get("id", ""): d for d in self.documents}
        ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in ranked[:top_k]:
            if doc_id in doc_map:
                results.append((doc_map[doc_id], score))

        return results

    def add_documents(self, new_documents: List[Dict]):
        """
        增量添加文档并重建索引
        （数据量小简单实现：追加后完全重建）
        """
        existing_ids = {d.get("id") for d in self.documents}
        truly_new = [d for d in new_documents if d.get("id") not in existing_ids]

        if not truly_new:
            print("  [增量] 没有新文档需要添加")
            return

        print(f"  [增量] 添加 {len(truly_new)} 篇新文档")
        self.documents.extend(truly_new)

        # 重建索引
        texts = [d.get("text", d.get("content", "")) for d in self.documents]
        self.tfidf.build_index(self.documents, texts)
        self.dense.build_index(self.documents, texts)
        self.ready = True

    def get_doc_hashes(self) -> Dict[str, str]:
        """获取所有文档的 hash，用于检测变化"""
        import hashlib
        hashes = {}
        for d in self.documents:
            content = d.get("content", "")
            h = hashlib.md5(content.encode()).hexdigest()
            hashes[d["id"]] = h
        return hashes

    def save(self, path: str):
        """保存索引"""
        import pickle
        import hashlib
        # 文档内容 hash 用于增量检测
        content_hash = hashlib.md5(
            "".join(d.get("content", "") for d in self.documents).encode()
        ).hexdigest()
        data = {
            "documents": self.documents,
            "tfidf_vocab": self.tfidf.vocab,
            "tfidf_idf": self.tfidf.idf,
            "tfidf_matrix": self.tfidf.tfidf_matrix,
            "dense_embeddings": self.dense.doc_embeddings,
            "dense_char_idx": self.dense.char_to_idx,
            "dense_n_chars": self.dense.n_chars,
            "dense_svd_Vt": self.dense.svd_Vt,
            "dense_svd_S": self.dense.svd_S,
            "alpha": self.alpha,
            "content_hash": content_hash,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"💾 索引已保存到: {path}")

    def load(self, path: str):
        """加载索引"""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.documents = data["documents"]
        self.alpha = data["alpha"]

        # 恢复 TF-IDF
        self.tfidf.documents = data["documents"]
        self.tfidf.doc_ids = [d.get("id", "") for d in data["documents"]]
        self.tfidf.vocab = data["tfidf_vocab"]
        self.tfidf.idf = data["tfidf_idf"]
        self.tfidf.tfidf_matrix = data["tfidf_matrix"]
        self.tfidf.ready = True

        # 恢复 Dense
        self.dense.documents = data["documents"]
        self.dense.doc_embeddings = data["dense_embeddings"]
        self.dense.char_to_idx = data["dense_char_idx"]
        self.dense.n_chars = data.get("dense_n_chars", len(self.dense.char_to_idx))
        self.dense.svd_Vt = data.get("dense_svd_Vt", None)
        self.dense.svd_S = data.get("dense_svd_S", None)
        self.dense.ready = True

        self.ready = True
        print(f"📖 索引已加载: {len(data['documents'])} 篇文档")
