"""
知识库索引构建工具
====================
python3 build_index.py              # 全新构建
python3 build_index.py --incremental  # 增量更新（新增文档自动加入）
python3 build_index.py --rebuild    # 强制重建
python3 build_index.py --stats      # 查看知识库统计
python3 build_index.py --watch      # 持续监听知识库变化
"""

import argparse
import os
import sys
import time
from qa_bot import QABot, KnowledgeBase


def check_for_updates(bot, kb):
    """检查知识库是否有新增或修改的文档"""
    docs = kb.load_documents()
    docs = [d for d in docs if d is not None]

    existing_ids = {d["id"] for d in bot.retriever.documents} if bot.retriever else set()
    new_docs = [d for d in docs if d["id"] not in existing_ids]

    return new_docs, len(docs)


def main():
    parser = argparse.ArgumentParser(description="知识库索引构建")
    parser.add_argument("--rebuild", action="store_true", help="强制重建")
    parser.add_argument("--incremental", action="store_true",
                        help="增量更新（仅添加新文档）")
    parser.add_argument("--stats", action="store_true", help="查看统计")
    parser.add_argument("--watch", action="store_true",
                        help="持续监听知识库变化")
    parser.add_argument("--kb", type=str, default="./knowledge_base")
    parser.add_argument("--index", type=str, default="./models/index.pkl")

    args = parser.parse_args()

    # 统计模式
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

    # 增量更新模式
    if args.incremental:
        if not os.path.exists(args.index):
            print("⚠️  索引不存在，执行全量构建...")
            bot = QABot(kb_dir=args.kb, index_path=args.index)
            bot.initialize(force_rebuild=True)
            print("✅ 索引构建完成")
            return

        print("🔍 检测知识库变更...")
        bot = QABot(kb_dir=args.kb, index_path=args.index)
        bot.initialize(force_rebuild=False)
        kb = KnowledgeBase(args.kb)
        new_docs, total = check_for_updates(bot, kb)
        n_new = len(new_docs)

        if n_new == 0:
            print(f"✅ 已是最新（共 {total} 篇文档，无新增）")
            return

        print(f"📝 发现 {n_new} 篇新文档（共 {total} 篇）")
        bot.retriever.add_documents(new_docs)
        bot.retriever.save(args.index)
        print(f"✅ 增量更新完成！新增 {n_new} 篇文档")
        return

    # 监听模式
    if args.watch:
        print("👀 监听知识库变化 (每 30 秒检查一次, Ctrl+C 退出)")
        bot = QABot(kb_dir=args.kb, index_path=args.index)
        bot.initialize(force_rebuild=False)
        last_count = len(bot.retriever.documents) if bot.retriever else 0

        try:
            while True:
                kb = KnowledgeBase(args.kb)
                new_docs, total = check_for_updates(bot, kb)
                if new_docs:
                    print(f"📝 检测到 {len(new_docs)} 篇新文档，正在更新...")
                    bot.retriever.add_documents(new_docs)
                    bot.retriever.save(args.index)
                    last_count = total
                    print(f"✅ 更新完成（共 {total} 篇文档）")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n👋 监听已停止")
        return

    # 全量构建（默认）
    print("🔄 构建知识库索引...")
    if args.rebuild:
        print("  (强制重建模式)")
    bot = QABot(kb_dir=args.kb, index_path=args.index)
    bot.initialize(force_rebuild=args.rebuild)
    print("✅ 完成！")


if __name__ == "__main__":
    main()
