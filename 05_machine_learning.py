"""
Step 5 - 机器学习入门 (scikit-learn)
======================================
运行: python3 05_machine_learning.py

学习目标:
  1. 理解 ML 核心概念（监督/无监督学习、过拟合、评估）
  2. 掌握 scikit-learn 统一 API（fit / predict / score）
  3. 能独立完成分类、回归、聚类任务
  4. 学会交叉验证、混淆矩阵、ROC 等评估方法
  5. 完成手写数字识别和文本分类两个实战项目
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# 中文字体（复用可视化教程的设置）
import matplotlib.font_manager as fm
chinese_fonts = ["Noto Serif CJK SC", "SimHei", "Microsoft YaHei",
                 "AR PL UKai CN", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei"]
available = {f.name for f in fm.fontManager.ttflist}
used_font = next((f for f in chinese_fonts if f in available), "DejaVu Sans")
matplotlib.rcParams["font.family"] = used_font
matplotlib.rcParams["axes.unicode_minus"] = False

os.makedirs("plots", exist_ok=True)

# =====================================================================
# PART 0: ML 核心概念（先读一遍再往下跑代码）
# =====================================================================
print("=" * 60)
print("PART 0: 机器学习核心概念")
print("=" * 60)

print("""
🧠 什么是机器学习？
   从数据中自动学习规律，然后用学到的规律做预测。

📂 三大学习范式:

  监督学习 ── 数据有标签（答案）
    ├── 分类（Classification）→ 预测类别: 垃圾/非垃圾、猫/狗
    └── 回归（Regression）   → 预测数值: 房价、温度

  无监督学习 ── 数据没有标签
    └── 聚类（Clustering）→ 自动分组: 用户分群、异常检测

  强化学习 ── 通过奖励信号学习（暂不涉及）

📐 训练/测试集拆分
   train = 训练模型, test = 评估泛化能力（模型没见过的数据）

⚖️ 过拟合 vs 欠拟合
   过拟合 = 死记硬背训练集 → 测试集表现差
   欠拟合 = 连训练集都没学好

📊 评估指标
   分类: accuracy, precision, recall, F1, AUC
   回归: R², MSE, MAE
""")

print("（概念部分读完，继续往下跑代码 ↓）\n")

# =====================================================================
# PART 1: scikit-learn 统一 API
# =====================================================================
print("\n" + "=" * 60)
print("PART 1: scikit-learn 统一 API 风格")
print("=" * 60)

print("""
所有 scikit-learn 模型都遵循同样的三步曲:

  1. model = 算法类(超参数)  ← 创建模型
  2. model.fit(X_train, y_train)  ← 训练
  3. model.predict(X_test)        ← 预测

  model.score(X_test, y_test)     ← 评估（分类=准确率, 回归=R²）
""")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载经典的鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target
print(f"数据集: {iris.DESCR[:200]}...\n")
print(f"特征矩阵 X: {X.shape}  (150 个样本, 4 个特征)")
print(f"标签 y: {y.shape}  (3 种鸢尾花)")
print(f"特征名: {iris.feature_names}")
print(f"类别名: {iris.target_names}")

# 拆分训练集 / 测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n训练集: {X_train.shape}  测试集: {X_test.shape}")

# 训练 KNN 分类器
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 预测 + 评估
y_pred = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)
print(f"\nKNN 分类器准确率: {accuracy:.2%}")

# 预测一个新样本
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # 特征值
pred = knn.predict(new_flower)
print(f"新样本预测结果: {iris.target_names[pred][0]}")


# =====================================================================
# PART 2: 分类算法
# =====================================================================
print("\n" + "=" * 60)
print("PART 2: 分类算法对比")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 用酒数据集（葡萄酒分类）
from sklearn.datasets import load_wine
wine = load_wine()
X_w, y_w = wine.data, wine.target
print(f"葡萄酒数据集: {X_w.shape}, 类别数: {len(wine.target_names)}")

X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(
    X_w, y_w, test_size=0.3, random_state=42, stratify=y_w
)

# 标准化（很多算法需要）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_w_train_scaled = scaler.fit_transform(X_w_train)
X_w_test_scaled = scaler.transform(X_w_test)

# 对比多种分类器
classifiers = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
    "决策树": DecisionTreeClassifier(max_depth=5, random_state=42),
    "SVM (RBF核)": SVC(kernel="rbf", random_state=42),
}

results = {}
for name, clf in classifiers.items():
    clf.fit(X_w_train_scaled, y_w_train)
    train_acc = clf.score(X_w_train_scaled, y_w_train)
    test_acc = clf.score(X_w_test_scaled, y_w_test)
    results[name] = {"训练准确率": train_acc, "测试准确率": test_acc}
    print(f"  {name:15s}  训练: {train_acc:.2%}  测试: {test_acc:.2%}")

# 结果可视化
results_df = pd.DataFrame(results).T
fig, ax = plt.subplots(figsize=(8, 5))
results_df.plot(kind="bar", ax=ax, color=["#4ECDC4", "#FF6B6B"])
ax.set_title("不同分类器性能对比", fontsize=14)
ax.set_ylabel("准确率")
ax.set_ylim(0, 1.05)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
ax.legend(loc="lower right")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("plots/ml01_classifier_comparison.png", dpi=150)
plt.close()
print("  ✅ plots/ml01_classifier_comparison.png")

# 详细评估：混淆矩阵 + 分类报告
print("\n--- 以 SVM 为例，看详细评估 ---")
best_clf = SVC(kernel="rbf", random_state=42)
best_clf.fit(X_w_train_scaled, y_w_train)
y_w_pred = best_clf.predict(X_w_test_scaled)

print("分类报告:")
print(classification_report(y_w_test, y_w_pred, target_names=wine.target_names))

# 混淆矩阵
cm = confusion_matrix(y_w_test, y_w_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=wine.target_names)
disp.plot(cmap="Blues")
plt.title("SVM 混淆矩阵 (葡萄酒数据集)", fontsize=14)
plt.tight_layout()
plt.savefig("plots/ml02_confusion_matrix.png", dpi=150)
plt.close()
print("  ✅ plots/ml02_confusion_matrix.png")


# =====================================================================
# PART 3: 回归算法
# =====================================================================
print("\n" + "=" * 60)
print("PART 3: 回归算法")
print("=" * 60)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

# 用 make_regression 生成回归数据集（不需要联网下载）
# 模拟房价场景：样本数=500, 特征=8, 有噪声
X_h, y_h, true_coef = make_regression(
    n_samples=500, n_features=8, noise=10,
    random_state=42, coef=True
)
# 让 y 为正数的"房价"（shift 到 50-100 范围）
y_h = y_h - y_h.min() + 50
feature_names = ["面积", "卧室数", "房龄", "距地铁站",
                  "楼层", "绿化率", "学校评分", "车位"]

print(f"模拟房价数据集: {X_h.shape}")
print(f"特征: {feature_names}")

X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(
    X_h, y_h, test_size=0.2, random_state=42
)

# 标准化
scaler_h = StandardScaler()
X_h_train = scaler_h.fit_transform(X_h_train)
X_h_test = scaler_h.transform(X_h_test)

# 对比回归模型
regressors = {
    "线性回归": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.01),
}

print("\n回归结果:")
for name, reg in regressors.items():
    reg.fit(X_h_train, y_h_train)
    y_pred = reg.predict(X_h_test)
    r2 = r2_score(y_h_test, y_pred)
    mse = mean_squared_error(y_h_test, y_pred)
    print(f"  {name:15s}  R²={r2:.4f}  MSE={mse:.4f}")

# 画预测 vs 真实值
y_min, y_max = y_h_test.min(), y_h_test.max()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, reg) in zip(axes, regressors.items()):
    reg.fit(X_h_train, y_h_train)
    y_pred = reg.predict(X_h_test)
    ax.scatter(y_h_test, y_pred, alpha=0.3, s=10, color="#4ECDC4")
    ax.plot([y_min, y_max], [y_min, y_max], "--", color="red", linewidth=1)
    ax.set_xlabel("真实值")
    ax.set_ylabel("预测值")
    ax.set_title(f"{name} (R²={r2_score(y_h_test, y_pred):.3f})")
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig("plots/ml03_regression.png", dpi=150)
plt.close()
print("  ✅ plots/ml03_regression.png")

# 特征重要性（线性回归的系数）
lr = LinearRegression()
lr.fit(X_h_train, y_h_train)
coef_df = pd.DataFrame({
    "特征": feature_names,
    "系数": lr.coef_
}).sort_values("系数", key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#FF6B6B" if c < 0 else "#4ECDC4" for c in coef_df["系数"]]
ax.barh(range(len(coef_df)), coef_df["系数"], color=colors)
ax.set_yticks(range(len(coef_df)))
ax.set_yticklabels(coef_df["特征"])
ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
ax.set_title("特征对房价的影响方向与大小", fontsize=14)
ax.set_xlabel("回归系数 (正=推高房价, 负=压低房价)")
plt.tight_layout()
plt.savefig("plots/ml04_feature_importance.png", dpi=150)
plt.close()
print("  ✅ plots/ml04_feature_importance.png")


# =====================================================================
# PART 4: 聚类（无监督学习）
# =====================================================================
print("\n" + "=" * 60)
print("PART 4: 聚类 — 无监督学习")
print("=" * 60)

from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons

# 生成一个人造数据集（3 个簇）
X_blob, y_blob = make_blobs(n_samples=300, centers=3,
                             cluster_std=1.0, random_state=42)

print("人造聚类数据: 300 个样本, 3 个簇")

# K-Means 聚类
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
y_kmeans = kmeans.fit_predict(X_blob)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左：真实标签
axes[0].scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap="viridis",
                s=40, edgecolors="white", linewidth=0.5)
axes[0].set_title("真实聚类结果", fontsize=14)

# 右：K-Means 结果
axes[1].scatter(X_blob[:, 0], X_blob[:, 1], c=y_kmeans, cmap="viridis",
                s=40, edgecolors="white", linewidth=0.5)
axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c="red", marker="X", s=200, label="质心")
axes[1].set_title("K-Means 聚类结果", fontsize=14)
axes[1].legend()

plt.tight_layout()
plt.savefig("plots/ml05_kmeans.png", dpi=150)
plt.close()
print("  ✅ plots/ml05_kmeans.png")

# 手肘法找最佳 K
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X_blob)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, "o-", color="#FF6B6B", linewidth=2)
plt.axvline(x=3, color="gray", linestyle="--", alpha=0.5, label="最佳 K=3")
plt.xlabel("K (簇数量)")
plt.ylabel("惯性 (Inertia)")
plt.title("手肘法确定最佳 K 值", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/ml06_elbow.png", dpi=150)
plt.close()
print("  ✅ plots/ml06_elbow.png")


# =====================================================================
# PART 5: 交叉验证
# =====================================================================
print("\n" + "=" * 60)
print("PART 5: 交叉验证 (Cross Validation)")
print("=" * 60)

"""
只用一次 train/test split 可能不够稳定（取决于怎么分的）。
交叉验证把数据分 K 份，轮流用 K-1 份训练、1 份验证，
最后取 K 次结果的平均值，更可靠。
"""

from sklearn.model_selection import cross_val_score, KFold

iris = load_iris()
X, y = iris.data, iris.target

# 5 折交叉验证
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, X, y, cv=5, scoring="accuracy")

print(f"5 折交叉验证结果: {scores}")
print(f"平均准确率: {scores.mean():.2%} (±{scores.std():.2%})")

# 对比只用一次 train/test split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
knn.fit(X_tr, y_tr)
single_score = knn.score(X_te, y_te)
print(f"单次 train/test split: {single_score:.2%}")
print(f"结论: 交叉验证更可靠，单次可能偏高或偏低")

# 用交叉验证调参
print("\n用交叉验证找 KNN 的最佳 K 值:")
best_k, best_score = 1, 0
for k in range(1, 30):
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k),
                             X, y, cv=5, scoring="accuracy")
    avg_score = scores.mean()
    if avg_score > best_score:
        best_score = avg_score
        best_k = k
print(f"  最佳 K 值: {best_k}, 交叉验证准确率: {best_score:.2%}")


# =====================================================================
# PART 6: 过拟合与欠拟合实战演示
# =====================================================================
print("\n" + "=" * 60)
print("PART 6: 过拟合 vs 欠拟合 (可视化)")
print("=" * 60)

np.random.seed(42)
X_deg = np.linspace(0, 1, 20)
y_deg = np.sin(2 * np.pi * X_deg) + np.random.randn(20) * 0.15
X_deg = X_deg.reshape(-1, 1)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
degrees = [1, 4, 15]
titles = ["欠拟合 (degree=1, 太简单)", "刚刚好 (degree=4)", "过拟合 (degree=15)"]

for ax, deg, title in zip(axes, degrees, titles):
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    model.fit(X_deg, y_deg)
    X_smooth = np.linspace(0, 1, 200).reshape(-1, 1)
    y_smooth = model.predict(X_smooth)
    ax.scatter(X_deg, y_deg, s=30, color="#FF6B6B", alpha=0.7, label="训练数据")
    ax.plot(X_smooth, y_smooth, color="#4ECDC4", linewidth=2, label="模型")
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(-2, 2)
    ax.legend()

plt.tight_layout()
plt.savefig("plots/ml07_overfitting.png", dpi=150)
plt.close()
print("  ✅ plots/ml07_overfitting.png")
print("""
肉眼可见:
  degree=1  → 欠拟合（一条直线根本学不会波浪）
  degree=4  → 刚刚好（抓住了整体趋势，没背噪声）
  degree=15 → 过拟合（把每个噪声点都记住了，震荡剧烈）
""")


# =====================================================================
# PART 7: 实战项目 1 — 手写数字识别
# =====================================================================
print("\n" + "=" * 60)
print("PART 7: 实战 — 手写数字识别")
print("=" * 60)

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

digits = load_digits()
X_d, y_d = digits.data, digits.target
print(f"数据集: {X_d.shape}  (1797 张 8×8 手写数字图片)")
print(f"类别: 0-9 共 10 个数字")

# 展示前 10 张数字图片
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"标签: {digits.target[i]}")
    ax.axis("off")
fig.suptitle("手写数字样本", fontsize=14)
plt.tight_layout()
plt.savefig("plots/ml08_digits_samples.png", dpi=150)
plt.close()
print("  ✅ plots/ml08_digits_samples.png")

# 拆分 + 标准化
X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(
    X_d, y_d, test_size=0.3, random_state=42, stratify=y_d
)
scaler_d = StandardScaler()
X_d_train = scaler_d.fit_transform(X_d_train)
X_d_test = scaler_d.transform(X_d_test)

# 训练 SVM
svm_d = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm_d.fit(X_d_train, y_d_train)
y_d_pred = svm_d.predict(X_d_test)
acc = accuracy_score(y_d_test, y_d_pred)

print(f"\nSVM 准确率: {acc:.2%}")

# 混淆矩阵
cm_d = confusion_matrix(y_d_test, y_d_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm_d, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title(f"SVM 手写数字识别 (准确率: {acc:.2%})", fontsize=14)
plt.xlabel("预测")
plt.ylabel("真实")
plt.tight_layout()
plt.savefig("plots/ml09_digits_confusion.png", dpi=150)
plt.close()
print("  ✅ plots/ml09_digits_confusion.png")

# 展示一些预测错误的样本
mis_idx = np.where(y_d_pred != y_d_test)[0]
if len(mis_idx) > 0:
    n_show = min(len(mis_idx), 10)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat[:n_show]):
        idx = mis_idx[i]
        ax.imshow(X_d_test[idx].reshape(8, 8), cmap="gray")
        ax.set_title(f"真实:{y_d_test[idx]} 预测:{y_d_pred[idx]}", color="red")
        ax.axis("off")
    fig.suptitle(f"预测错误的样本 ({len(mis_idx)} / {len(y_d_test)})", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/ml10_digits_errors.png", dpi=150)
    plt.close()
    print("  ✅ plots/ml10_digits_errors.png")

print(f"\n共 {len(mis_idx)} 个错误，准确率 {acc:.2%} — 用深度学习可以做到 99%+")


# =====================================================================
# PART 8: 实战项目 2 — 文本分类（垃圾邮件检测）
# =====================================================================
print("\n" + "=" * 60)
print("PART 8: 实战 — 文本分类（垃圾邮件检测）")
print("=" * 60)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 模拟邮件数据（实际项目中你会从 CSV 读取）
emails = [
    # 正常邮件
    "Hi, let's meet for lunch tomorrow at 12pm",
    "Can you send me the meeting notes from yesterday",
    "The project deadline has been extended to next Friday",
    "Please review the attached document and provide feedback",
    "Happy birthday! Hope you have a great day",
    "Your order has been shipped and will arrive in 3 days",
    "Thank you for your application. We will review it shortly",
    "The quarterly report is due next week",
    "Reminder: team standup at 9:30am tomorrow",
    "Let's schedule a call to discuss the new features",
    "Congratulations on your promotion! Well deserved",
    "Please update the status of your current tasks",
    "The server maintenance is scheduled for this weekend",
    "Your invoice for last month is attached",
    "Thank you for being a valued customer",
    "Lunch today? I found a great new place near the office",
    "The API documentation has been updated",
    "Don't forget to submit your timesheet by Friday",
    "Can you review my pull request when you get a chance",
    "Welcome to the team! Let me know if you need anything",
    # 垃圾邮件
    "Congratulations! You have won a free iPhone! Click here to claim",
    "URGENT: Your account has been compromised, verify now",
    "Make $5000 per week working from home! No experience needed",
    "You have been selected for a free vacation to the Bahamas",
    "Limited offer: Buy one get one free today only",
    "Your Netflix account is suspended. Update payment immediately",
    "Earn Bitcoin fast! Join our investment group now",
    "FREE VIAGRA AND OTHER MEDICATIONS AT DISCOUNT PRICES",
    "You are the lucky winner of our lottery! Send $100 to claim",
    "Get rich quick with this simple system. Guaranteed returns",
    "Act now! Special discount for our loyal customers",
    "Low interest rate personal loans available. Bad credit OK",
    "Your package could not be delivered. Click to reschedule",
    "You have an unclaimed tax refund of $5000",
    "Exclusive membership offer. 90% off for the first month",
    "Work from home and earn $10000 monthly. Start today",
    "Your computer has been infected with a virus. Download fix",
    "Congratulations! You have been chosen for a free gift card",
    "Urgent: Your bank account will be closed. Verify identity",
    "Cheap medications online no prescription needed",
]

labels = [0] * 20 + [1] * 20  # 0=正常, 1=垃圾

print(f"邮件总数: {len(emails)}")
print(f"正常邮件: {labels.count(0)}, 垃圾邮件: {labels.count(1)}")

# 展示几条样本
print("\n正常邮件示例:", emails[0])
print("垃圾邮件示例:", emails[21])

# 拆分
X_e_train, X_e_test, y_e_train, y_e_test = train_test_split(
    emails, labels, test_size=0.3, random_state=42, stratify=labels
)

# 文本特征提取：TF-IDF
# TF-IDF = 词频(TF) × 逆文档频率(IDF)
# 它给"常见但信息量低"的词降权（如"the", "a"），
# 给"少见但有区分度"的词加权
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X_e_train_vec = vectorizer.fit_transform(X_e_train)
X_e_test_vec = vectorizer.transform(X_e_test)

print(f"\n词表大小: {len(vectorizer.get_feature_names_out())}")
print("前 20 个最有区分度的词:")
print(vectorizer.get_feature_names_out()[:20])

# 朴素贝叶斯分类器（文本分类的经典选择）
nb = MultinomialNB()
nb.fit(X_e_train_vec, y_e_train)
y_e_pred = nb.predict(X_e_test_vec)
acc_e = accuracy_score(y_e_test, y_e_pred)

print(f"\n朴素贝叶斯准确率: {acc_e:.2%}")
print("\n分类报告:")
print(classification_report(y_e_test, y_e_pred, target_names=["正常", "垃圾"]))

# 测试新邮件
test_emails = [
    "Hey, are we still on for lunch today?",
    "CONGRATULATIONS YOU WON A FREE CAR!!! CLICK HERE!!!",
    "Can you help me with the project report?",
    "Urgent: Your PayPal account has been limited"
]
test_vec = vectorizer.transform(test_emails)
predictions = nb.predict(test_vec)

print("\n新邮件测试:")
for email, pred in zip(test_emails, predictions):
    label = "📧 正常" if pred == 0 else "🚫 垃圾"
    print(f"  [{label}] {email[:50]}...")


# =====================================================================
# PART 9: 总结 — 什么时候用什么算法
# =====================================================================
print("\n" + "=" * 60)
print("PART 9: 算法选择指南")
print("=" * 60)

print("""
📋 分类问题（目标：离散类别）
  ├── 数据量小、特征少       → SVM / 逻辑回归
  ├── 数据量大               → 随机森林 / XGBoost / 神经网络
  ├── 文本分类               → 朴素贝叶斯 + TF-IDF
  ├── 需要解释性             → 决策树 / 逻辑回归
  └── 图像分类               → 深度学习 (CNN)

📋 回归问题（目标：连续数值）
  ├── 特征少、关系线性       → 线性回归
  ├── 特征多、防过拟合       → Ridge / Lasso
  └── 非线性关系             → 随机森林回归 / SVR

📋 聚类问题（无监督、无标签）
  ├── 簇是球形的             → K-Means
  ├── 形状不规则             → DBSCAN
  └── 层次关系               → 层次聚类

🎯 通用工作流:
  1. 理解问题（分类/回归/聚类？）
  2. 数据探索（可视化 + 统计描述）
  3. 数据预处理（缺失值、标准化、特征工程）
  4. 选基线模型（先用简单的，如线性模型/KNN）
  5. 模型评估（交叉验证，不要只看一次准确率）
  6. 超参数调优（GridSearchCV）
  7. 选择最佳模型，在测试集上做最终评估
""")


# =====================================================================
# 课后练习
# =====================================================================
print("\n" + "=" * 60)
print("课后练习")
print("=" * 60)

print("""
题目 1: 癌症诊断分类（Kaggle 入门级）
  - 使用 sklearn.datasets.load_breast_cancer()
  - 对比 3 种分类器，用交叉验证选最好的
  - 画 ROC 曲线，计算 AUC

题目 2: 房价预测改进
  - 在 PART 3 的房价数据集上，试试随机森林回归
    (from sklearn.ensemble import RandomForestRegressor)
  - 用 GridSearchCV 调参
  - 比较和线性回归的差距

题目 3: 客户分群
  - 生成一个人造数据集（make_blobs, 4 个簇）
  - 用 K-Means 聚类
  - 可视化聚类结果，标注质心
  - 手肘法验证最佳 K 值

题目 4: 完整 ML 项目流水线
  从 load_digits() 开始，完成一个完整的 ML 项目:
    1. 数据探索（展示样本、统计描述）
    2. 预处理（标准化、PCA 降维）
    3. 多模型对比（KNN, SVM, 随机森林）
    4. 交叉验证 + 调参
    5. 最终评估 + 混淆矩阵
    6. 分析错误样本

题目 5（进阶）: 自己收集文本数据
  找 50 条中文评论（正面/负面各 25 条），
  用 CountVectorizer + 朴素贝叶斯做情感分析
""")

print("\n✅ 机器学习入门教程结束！下个阶段: Phase 5 — PyTorch 深度学习")
