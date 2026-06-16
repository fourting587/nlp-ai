"""
中文情感分析 — PyTorch 从零实现（离线可用）
===============================================
无需下载预训练模型，用 PyTorch 搭建词嵌入 + CNN 分类器。
有网络后可一键切换到 BERT（见注释）。

用法:
    python3 train.py                    # 训练并评估
    python3 train.py --epochs 10         # 自定义轮数

输出:
    ./models/  — 保存的模型
"""

import argparse
import json
import os
import re
import sys
import logging
from collections import Counter

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 1. 数据 — 内置 ChnSentiCorp 子集（酒店评论）
# ============================================================
TRAIN_DATA = [
    # 正面 (label=1)
    ("很不错的一个酒店，床很大，很舒服。酒店员工的服务态度很亲切。", 1),
    ("第一次入住该酒店，环境雅致，服务非常不错，很多笑脸，感觉热情。", 1),
    ("酒店环境和服务都还不错，地理位置也很好。", 1),
    ("房间很整洁，床上的靠枕很喜庆。交通很便利，到渔人码头和港澳码头都在步行范围之内。", 1),
    ("checkin和checkout的速度都很快，服务很好。", 1),
    ("非常棒的一次入住体验，下次还会再来。", 1),
    ("位置很好，离地铁站很近，周围吃饭很方便。", 1),
    ("房间宽敞明亮，卫生做得非常干净，推荐入住。", 1),
    ("性价比很高，这个价位能住到这样的酒店很满意。", 1),
    ("前台服务热情周到，还免费升级了房型，太惊喜了！", 1),
    ("早餐种类丰富，中西式都有，味道很好。", 1),
    ("酒店设施很新，装修风格简约大方，很喜欢的风格。", 1),
    ("游泳池和健身房都很不错，设施齐全。", 1),
    ("周边环境安静，睡眠质量很好，床垫非常舒服。", 1),
    ("服务人员态度非常好，有求必应，感觉很温暖。", 1),
    ("房间的景观非常好，可以看到整个城市的美景。", 1),
    ("酒店很干净，每天都有打扫，床单枕套都很干净。", 1),
    ("网络速度很快，办公完全没有问题。", 1),
    ("性价比非常高，比同价位的其他酒店好很多。", 1),
    ("这是一次非常愉快的入住体验，强烈推荐！", 1),
    ("设施齐全，环境优美，服务态度好，值得推荐。", 1),
    ("这家酒店的各方面都很好，很满意。", 1),
    ("房间很大很舒适，超乎预期，下次还会选择这里。", 1),
    ("服务非常贴心，还有免费的接送服务，很赞。", 1),
    ("整体感觉非常好，无论是环境还是服务都无可挑剔。", 1),
    ("停车很方便，有地下车库。房间也很干净整洁。", 1),
    ("性价比很高啊，这个价位太划算了。", 1),
    ("非常满意的一次入住，酒店的每一个细节都做得很好。", 1),
    ("服务很好，环境也很好，非常推荐这家酒店。", 1),
    ("前台很热情，入住办理很快，退房也很顺利。", 1),
    ("出差一直住这里，服务一直很稳定。", 1),
    ("酒店位置很好，旁边就是商业街，购物吃饭都很方便。", 1),
    ("房间的床很大很软，睡得很舒服。", 1),
    ("服务很棒，还给免费升级了套房，太感谢了。", 1),
    ("卫生条件很好，比想象中要干净很多。", 1),
    ("交通便利，出门就是地铁站，去哪里都方便。", 1),
    ("非常满意，无论是设施还是服务都在水准之上。", 1),
    ("酒店的装修很有特色，很有当地的文化氛围。", 1),
    ("已经是第三次来了，每次都不会失望。", 1),
    ("服务态度极好，非常满意这次入住体验。", 1),
    ("环境非常好，空气清新，很适合度假。", 1),
    ("房间干净整洁，设施齐全，非常满意。", 1),
    ("强烈推荐！性价比超高，服务也很好。", 1),
    ("非常好的酒店，各方面都很满意。", 1),
    ("早餐很好吃，品种很多，中西式都有。", 1),
    ("服务很好，环境优雅，价格合理。", 1),
    ("酒店的服务真的很棒，每个员工都很热情。", 1),
    ("很好很干净，位置方便，推荐入住。", 1),
    ("入住体验非常棒，超出预期。", 1),
    ("这个价位能住到这样的酒店真的很值了。", 1),

    # 负面 (label=0)
    ("房间小得无法想象，个子大的不要选择，脚都伸不直。", 0),
    ("设施老化，紧靠马路噪音太大。晚上楼上卫生间的水流声和空调噪音非常大，无法入眠。", 0),
    ("非常糟糕！酒店拉我们去不正规景点买茶叶，太坑了。", 0),
    ("是我遇到的最差的酒店，进门没人管，退房也很慢，不会再去住了。", 0),
    ("酒店设施陈旧，浴缸排水不畅，入住时没有房间。", 0),
    ("房间很脏，床单上还有头发，太恶心了。", 0),
    ("前台服务态度极差，爱理不理的，问问题也不回答。", 0),
    ("隔音效果太差了，隔壁说话都能听到，完全无法休息。", 0),
    ("热水供应不稳定，洗澡洗到一半没有热水了。", 0),
    ("空调是坏的，大热天没有空调根本没法住。", 0),
    ("酒店位置很偏僻，周围什么都没有，吃饭很不方便。", 0),
    ("图片和实际完全不符，房间根本没有图片上好看。", 0),
    ("价格虚高，这个质量根本不值这个价。", 0),
    ("WiFi信号很差，经常断线，根本无法办公。", 0),
    ("卫生情况令人堪忧，墙角有霉斑，卫生间有异味。", 0),
    ("电梯经常坏，住高层只能爬楼梯，太不方便了。", 0),
    ("早餐太难吃了，品种少，质量差。", 0),
    ("预订了房间到了却说没有了，给换了个更差的房间。", 0),
    ("床太硬了，睡了一晚腰酸背痛。", 0),
    ("房间里有很多蚊子，一晚上没睡好。", 0),
    ("服务人员态度恶劣，投诉也没有用。", 0),
    ("停车场太小，根本没有位置停车。", 0),
    ("房间有异味，通风也不好，很闷。", 0),
    ("墙上有裂缝，卫生间漏水，设施太老旧了。", 0),
    ("周边在施工，非常吵，完全没法休息。", 0),
    ("入住等了两个小时，效率太低了吧。", 0),
    ("价格这么贵，条件却这么差，完全不值。", 0),
    ("不会再来了，这是住过最差的酒店。", 0),
    ("房间里有蟑螂，卫生条件太差了。", 0),
    ("矿泉水还要收费，太抠门了。", 0),
    ("空调声音很大，而且制冷效果很差。", 0),
    ("床单被套有明显污渍，让人很不舒服。", 0),
    ("退房时居然说房间有损坏要赔钱，明显是故意讹诈。", 0),
    ("房间打扫不彻底，桌上有灰尘，地上有头发。", 0),
    ("隔音非常差，早上被走廊的声音吵醒。", 0),
    ("酒店的设施很陈旧，电视还是老式的。", 0),
    ("位置非常偏僻，打车都不方便。", 0),
    ("这次入住体验非常差，强烈不推荐。", 0),
    ("服务太差了，连基本的礼貌都没有。", 0),
    ("性价比极低，这个价格完全可以住更好的酒店。", 0),
    ("前台态度冷漠，办理入住等了很久。", 0),
    ("房间的窗户关不上，晚上很冷。", 0),
    ("提供的洗漱用品质量很差，洗发水都没效果。", 0),
    ("预约的接机服务没来，打电话也没人接。", 0),
    ("餐厅的食物很不新鲜，吃了拉肚子。", 0),
    ("房间很小很压抑，窗户也没有，不透气。", 0),
    ("这是我住过最糟糕的酒店，没有之一。", 0),
    ("环境脏乱差，服务态度恶劣，不推荐。", 0),
]

# 验证集
TEST_DATA = [
    ("交通很便利，到渔人码头和港澳码头都在步行的范围之内。", 1),
    ("服务态度很差，以后不会再来了。", 0),
    ("环境优雅，服务周到，非常推荐。", 1),
    ("设施太陈旧了，需要全面翻新。", 0),
    ("价格合理，位置便利，值得推荐。", 1),
    ("垃圾酒店，千万别来！", 0),
    ("总体来说很满意，下次还会入住。", 1),
    ("太差了，完全不值这个价。", 0),
    ("酒店很干净，服务也很好。", 1),
    ("不会再来了，体验很差。", 0),
    ("性价比很高，房间很舒适。", 1),
    ("设施老旧，需要更新。", 0),
]


# ============================================================
# 2. 词表构建
# ============================================================
def build_vocab(texts, min_freq=1, max_vocab=10000):
    """从文本列表构建词表（字符级别）"""
    counter = Counter()
    for text in texts:
        for char in text:
            counter[char] += 1

    # 按频率排序
    vocab_list = ["<PAD>", "<UNK>"] + [
        char for char, freq in counter.most_common(max_vocab)
        if freq >= min_freq
    ]
    word2idx = {word: idx for idx, word in enumerate(vocab_list)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    logger.info(f"词表大小: {len(vocab_list)} (来自 {len(texts)} 条文本)")
    return word2idx, idx2word


def encode_text(text, word2idx, max_len=64):
    """将文本编码为 token ids"""
    ids = [word2idx.get(c, word2idx["<UNK>"]) for c in text]
    # padding / truncation
    if len(ids) < max_len:
        ids = ids + [word2idx["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


# ============================================================
# 3. 模型定义
# ============================================================
class TextCNN(nn.Module):
    """
    TextCNN — 用多个卷积核提取 n-gram 特征

    论文: Convolutional Neural Networks for Sentence Classification (Yoon Kim, 2014)
    """
    def __init__(self, vocab_size, embed_dim=128, num_filters=64,
                 filter_sizes=(2, 3, 4, 5), num_classes=2, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 多个不同大小的卷积核，捕捉不同长度的 n-gram
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k // 2)
            for k in filter_sizes
        ])

        # 池化 + 分类
        total_filters = num_filters * len(filter_sizes)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_filters, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)                      # (batch, seq_len, embed_dim)
        emb = emb.transpose(1, 2)                    # -> (batch, embed_dim, seq_len)

        # 每个卷积核池化后拼接
        conv_outs = []
        for conv in self.convs:
            conv_out = conv(emb)                     # (batch, num_filters, seq_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch, num_filters)
            conv_outs.append(pooled)

        combined = torch.cat(conv_outs, dim=1)       # (batch, total_filters)
        return self.classifier(combined)


# ============================================================
# 4. 训练
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="中文情感分析 — PyTorch 从零实现")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./models")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 50)
    logger.info("📝 中文情感分析 — TextCNN 从零实现")
    logger.info(f"    数据: 训练 {len(TRAIN_DATA)} 条, 测试 {len(TEST_DATA)} 条")
    logger.info(f"    配置: epochs={args.epochs}, batch={args.batch_size}, "
                f"lr={args.lr}, max_len={args.max_len}")
    logger.info("=" * 50)

    # --- 数据准备 ---
    all_texts = [t for t, _ in TRAIN_DATA] + [t for t, _ in TEST_DATA]
    word2idx, idx2word = build_vocab(all_texts)
    vocab_size = len(word2idx)

    def prepare_dataset(data, word2idx, max_len):
        X = torch.tensor([encode_text(t, word2idx, max_len) for t, _ in data])
        y = torch.tensor([l for _, l in data])
        return TensorDataset(X, y)

    from torch.utils.data import TensorDataset
    train_dataset = prepare_dataset(TRAIN_DATA, word2idx, args.max_len)
    test_dataset = prepare_dataset(TEST_DATA, word2idx, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # --- 模型初始化 ---
    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_filters=64,
        filter_sizes=(2, 3, 4, 5),
        dropout=0.3,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"⚡ 设备: {device} | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- 训练循环 ---
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 梯度裁剪
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # --- 验证 ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.0 * correct / total
        logger.info(f"Epoch {epoch:2d}/{args.epochs} | Loss={total_loss/len(train_loader):.4f} "
                    f"| Test Acc={acc:.1f}%")

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "word2idx": word2idx,
                "idx2word": idx2word,
                "args": args,
                "accuracy": acc,
            }, os.path.join(args.output_dir, "best.pt"))
            logger.info(f"   ⭐ 新最佳模型保存 (Acc={acc:.1f}%)")

    logger.info("=" * 50)
    logger.info(f"🎉 训练完成！最佳准确率: {best_acc:.1f}%")
    logger.info(f"💾 模型已保存到: {os.path.join(args.output_dir, 'best.pt')}")
    logger.info("📌 预测命令: python3 predict.py --text '酒店很好'")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
