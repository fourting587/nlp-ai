"""
中文情感分析 — 推理预测脚本
=============================
用法:
    python3 predict.py --text "这个酒店很好"
    python3 predict.py --file reviews.txt     # 批量
    python3 predict.py --interactive           # 交互模式
"""

import argparse
import os
import sys

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="中文情感分析 — 预测")
    parser.add_argument("--text", type=str, help="待分析文本")
    parser.add_argument("--file", type=str, help="批量分析文件（每行一条）")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--model", type=str, default="./models/best.pt",
                        help="模型路径")
    parser.add_argument("--max_len", type=int, default=64)
    return parser.parse_args()


def load_model(model_path, device):
    """加载模型和词表"""
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        print("请先运行 python3 train.py 训练模型")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    word2idx = checkpoint["word2idx"]

    # 重建模型结构
    from train import TextCNN
    model = TextCNN(
        vocab_size=len(word2idx),
        embed_dim=args.embed_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, word2idx, args


def encode_text(text, word2idx, max_len=64):
    """将文本编码为 token ids"""
    ids = [word2idx.get(c, word2idx["<UNK>"]) for c in text]
    if len(ids) < max_len:
        ids = ids + [word2idx["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


def predict(text, model, word2idx, device, max_len=64):
    """单条预测"""
    tokens = encode_text(text, word2idx, max_len)
    inputs = torch.tensor([tokens]).to(device)

    with torch.no_grad():
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=-1)
        pred = outputs.argmax(dim=-1).item()

    confidence = probs[0][pred].item()

    return {
        "text": text,
        "label": pred,
        "sentiment": "正面 😊" if pred == 1 else "负面 😞",
        "confidence": confidence,
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    print(f"🧠 加载模型 ({device})...")
    model, word2idx, train_args = load_model(args.model, device)
    max_len = train_args.max_len
    print(f"✅ 模型加载完成！词表大小: {len(word2idx)}")

    # 交互模式
    if args.interactive:
        print("\n🔤 交互模式（输入 q 退出）")
        print("-" * 40)
        while True:
            text = input("请输入评论: ").strip()
            if text.lower() in ("q", "quit", "exit"):
                break
            if not text:
                continue
            result = predict(text, model, word2idx, device, max_len)
            print(f"   → {result['sentiment']}  (置信度: {result['confidence']:.2%})")
        return

    # 批量
    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ 文件不存在: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"📄 批量分析 {len(texts)} 条...")
        print("-" * 60)
        positives = 0
        for text in texts:
            result = predict(text, model, word2idx, device, max_len)
            icon = "😊" if result["label"] == 1 else "😞"
            print(f"{icon} | {result['confidence']:>5.1%} | {text}")
            if result["label"] == 1:
                positives += 1
        print("-" * 60)
        total = len(texts)
        neg = total - positives
        print(f"总计: {total} 条 | 正面: {positives}({positives/total*100:.1f}%) | "
              f"负面: {neg}({neg/total*100:.1f}%)")
        return

    # 单条
    if args.text:
        result = predict(args.text, model, word2idx, device, max_len)
        print(f"📝 评论: {result['text']}")
        print(f"🏷️  情感: {result['sentiment']}")
        print(f"📊 置信度: {result['confidence']:.2%}")
    else:
        print("请指定 --text、--file 或使用 --interactive 模式")


if __name__ == "__main__":
    main()
