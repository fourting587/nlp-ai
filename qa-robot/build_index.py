"""
知识库索引构建工具
====================
python3 build_index.py           # 构建/重建索引
python3 build_index.py --stats   # 查看知识库统计
"""

import argparse
import os
import sys
from qa_bot import QABot, KnowledgeBase


def main():
    parser = argparse.ArgumentParser(description="知识库索引构建")
    parser.add_argument("--rebuild", action="store_true",
                        help="强制重建索引")
    parser.add_argument("--stats", action="store_true",
                        help="查看知识库统计")
    parser.add_argument("--kb", type=str, default="./knowledge_base",
                        help="知识库路径")

    args = parser.parse_args()

    if args.stats:
        kb = KnowledgeBase(args.kb)
        try:
            stats = kb.get_statistics()
            print("📊 知识库统计:")
            print(f"  文档数: {stats['num_docs']}")
            print(f"  总字数: {stats['total_chars']:,}")
            print(f"  平均长度: {stats['avg_doc_length']} 字/篇")
        except Exception as e:
            print(f"❌ 错误: {e}")
        return

    # 构建索引
    print("🔄 构建知识库索引...")
    bot = QABot(kb_dir=args.kb)
    bot.initialize(force_rebuild=args.rebuild)
    print("✅ 完成！")


if __name__ == "__main__":
    main()
