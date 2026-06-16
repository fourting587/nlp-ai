"""
🌐 QA Bot Web 界面 (Gradio)
=============================
python3 app.py              # 启动 Web UI (默认 7860)
python3 app.py --port 8080  # 指定端口
python3 app.py --llm        # 启用 LLM 生成
python3 app.py --share      # 生成公网链接（Gradio）
"""

import argparse
import os
import sys
import time

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import gradio as gr
except ImportError:
    print("❌ 请先安装 Gradio: pip install gradio")
    sys.exit(1)

from qa_bot import QABot


# CSS 样式
CSS = """
.gradio-container { max-width: 900px !important; margin: auto; }
.chat-message { font-size: 15px; line-height: 1.6; }
footer { display: none !important; }
"""


# ============================================================
# 创建 Bot 实例
# ============================================================
def create_bot(llm=False, api_key=None, kb_dir="./knowledge_base"):
    """创建并初始化 QA Bot"""
    bot = QABot(
        kb_dir=kb_dir,
        use_llm=llm,
        api_key=api_key,
        cache_threshold=0.85,
    )
    bot.initialize(force_rebuild=False)
    return bot


# ============================================================
# Gradio 界面
# ============================================================
def create_ui(bot):
    """创建 Gradio 界面"""

    # 问答函数
    def respond(message, history, use_llm_flag, use_web_flag, top_k):
        """处理用户消息"""
        if not message.strip():
            return "请输入问题。"

        # 更新设置
        bot._use_llm_setting = use_llm_flag
        bot._use_web_setting = use_web_flag

        # 调用 bot
        start = time.time()
        result = bot.ask(message, top_k=int(top_k))

        answer = result["answer"]
        elapsed = time.time() - start

        # 构建元信息
        meta = []
        if result.get("cached"):
            meta.append("⚡ 缓存命中")
        if result.get("web_sourced"):
            meta.append("🌐 联网搜索")
        if result.get("llm_generated"):
            meta.append("🤖 LLM 生成")

        confidence = result["confidence"]
        if confidence >= 0.7:
            conf_icon = "🟢"
        elif confidence >= 0.4:
            conf_icon = "🟡"
        else:
            conf_icon = "🔴"

        meta.append(f"{conf_icon} 置信度: {confidence:.0%}")
        meta.append(f"⏱️ {elapsed:.1f}s")

        # 来源
        sources = result.get("sources", [])
        if sources:
            if result.get("web_sourced"):
                src_text = "🌐 来源: " + " | ".join(
                    s["title"][:30] for s in sources[:3]
                )
            else:
                src_text = "📚 参考: " + " | ".join(
                    f"{s['title']} ({s.get('score', 0):.2f})"
                    for s in sources[:3]
                )
            meta.append(src_text)

        # 组装输出
        output = f"{answer}\n\n---\n{' · '.join(meta)}"
        return output

    with gr.Blocks(
        title="🤖 智能问答机器人",
    ) as demo:
        gr.Markdown(
            """
            # 🤖 智能问答机器人
            ### 基于 RAG 架构的中文 AI 知识库问答系统
            """,
        )

        with gr.Row():
            with gr.Column(scale=2):
                top_k = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1,
                    label="检索文档数",
                    info="知识库模式: 越多答案越丰富",
                )
            with gr.Column(scale=2):
                use_llm_flag = gr.Checkbox(
                    label="LLM 生成",
                    value=bot.use_llm,
                    info="需要 ANTHROPIC_API_KEY",
                    interactive=True,
                )
            with gr.Column(scale=2):
                use_web_flag = gr.Checkbox(
                    label="🌐 联网搜索",
                    value=False,
                    info="开启后可回答任意问题（不限知识库）",
                    interactive=True,
                )

        # 聊天组件
        chatbot = gr.ChatInterface(
            fn=respond,
            additional_inputs=[use_llm_flag, use_web_flag, top_k],
            title="",
            description="输入关于 AI/深度学习/NLP 的技术问题...",
            examples=[
                ["什么是Transformer？", False, False, 3],
                ["过拟合怎么解决？", False, False, 3],
                ["BERT 和 GPT 有什么区别？", False, False, 3],
                ["2026年世界杯在哪里举办？", False, True, 3],
                ["今天比特币价格多少？", False, True, 3],
                ["Python 列表和元组的区别", False, False, 3],
            ],
        )

        # 底部信息
        gr.Markdown(
            """
            ---
            **知识库**: 7 篇 AI/NLP 面试知识点文档 | **检索**: TF-IDF + Dense 混合检索
            """,
        )

    return demo


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="🌐 QA Bot Web 界面")
    parser.add_argument("--port", type=int, default=7860, help="端口号")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="绑定地址")
    parser.add_argument("--llm", action="store_true", help="启用 LLM 生成")
    parser.add_argument("--web", action="store_true", help="启用联网搜索")
    parser.add_argument("--api-key", type=str, default=None, help="Anthropic API Key")
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    parser.add_argument("--kb", type=str, default="./knowledge_base", help="知识库路径")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引")

    args = parser.parse_args()

    # 初始化 Bot
    print(f"🔧 初始化 QA Bot (LLM={'启用' if args.llm else '未启用'}, "
          f"Web={'启用' if args.web else '未启用'})...")
    bot = create_bot(
        llm=args.llm,
        api_key=args.api_key,
        kb_dir=args.kb,
    )
    bot._use_web_setting = args.web
    print(f"✅ Bot 就绪！启动 Web 界面: http://{args.host}:{args.port}")

    # 启动 Gradio
    demo = create_ui(bot)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=CSS,
    )


if __name__ == "__main__":
    main()
