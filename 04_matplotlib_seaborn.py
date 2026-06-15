"""
Step 4 - Matplotlib & Seaborn 数据可视化教程
=============================================
运行: python3 04_matplotlib_seaborn.py

运行后所有图表保存在 plots/ 文件夹中，用图片查看器打开即可。

学习目标:
  1. 掌握 Matplotlib 基本图表（折线图、柱状图、散点图、直方图）
  2. 掌握子图布局和样式美化
  3. 掌握 Seaborn 统计图表（箱线图、热力图、分布图）
  4. 能根据数据选择合适的图表类型
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================================================================
# 0. 准备工作
# =====================================================================

# （a）创建 plots 目录
os.makedirs("plots", exist_ok=True)

# （b）设置中文字体
# 先检查系统中可用的中文字体
# 这里用排查法找可用的字体
import matplotlib.font_manager as fm

# 尝试常见中文字体列表
chinese_fonts = [
    "Noto Serif CJK SC", "SimHei", "Microsoft YaHei",
    "AR PL UKai CN", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
    "DejaVu Sans"
]
available = {f.name for f in fm.fontManager.ttflist}
used_font = None
for f in chinese_fonts:
    if f in available:
        used_font = f
        break
if not used_font:
    used_font = "DejaVu Sans"  # fallback 无中文

print(f"使用字体: {used_font}")
matplotlib.rcParams["font.family"] = used_font
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示

# （c）设置 Seaborn 主题
sns.set_theme(style="whitegrid", font=used_font)
# 颜色方案
COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]


# =====================================================================
# 1. Matplotlib 基础
# =====================================================================
print("=" * 60)
print("1. Matplotlib 基础")
print("=" * 60)
print("  图表已保存至 plots/ 目录\n")

# 最简单的一张图
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.figure(figsize=(8, 5))
plt.plot(x, y, marker="o", linestyle="-", color=COLORS[0], linewidth=2)
plt.title("最简单的折线图")
plt.xlabel("X 轴")
plt.ylabel("Y 轴")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/01_basic_line.png", dpi=150)
plt.close()
print("  ✅ plots/01_basic_line.png")


# =====================================================================
# 2. 核心图表类型
# =====================================================================
print("\n" + "=" * 60)
print("2. 核心图表类型")
print("=" * 60)

# --- 2a. 折线图（趋势） ---
print("\n--- 2a. 折线图 ---")

# 生成模拟数据：某 APP 日活跃用户趋势
dates = pd.date_range("2026-01-01", periods=30, freq="D")
dau = np.random.randint(800, 1200, 30) + np.arange(30) * 5  # 上升趋势+随机波动

plt.figure(figsize=(10, 5))
plt.plot(dates, dau, color=COLORS[0], linewidth=2, marker="o", markersize=4)
plt.title("2026年1月 APP 日活跃用户趋势", fontsize=14)
plt.xlabel("日期")
plt.ylabel("活跃用户数")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/02a_line_chart.png", dpi=150)
plt.close()
print("  ✅ plots/02a_line_chart.png")

# 多条折线对比
weeks = np.arange(1, 8)
app_a = np.random.randint(500, 800, 7)
app_b = np.random.randint(400, 700, 7)
app_c = np.random.randint(300, 600, 7)

plt.figure(figsize=(10, 5))
plt.plot(weeks, app_a, "o-", color=COLORS[0], label="App A", linewidth=2)
plt.plot(weeks, app_b, "s--", color=COLORS[1], label="App B", linewidth=2)
plt.plot(weeks, app_c, "^-.", color=COLORS[2], label="App C", linewidth=2)
plt.title("三款 APP 周活跃用户对比", fontsize=14)
plt.xlabel("周次")
plt.ylabel("活跃用户数")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/02b_multi_line.png", dpi=150)
plt.close()
print("  ✅ plots/02b_multi_line.png")


# --- 2b. 柱状图（对比） ---
print("\n--- 2b. 柱状图 ---")

categories = ["计算机", "数学", "物理", "化学", "生物"]
values = [85, 72, 68, 90, 78]

plt.figure(figsize=(8, 5))
bars = plt.bar(categories, values, color=COLORS[:5], edgecolor="white", linewidth=1.5)
# 在柱子上标数值
for bar, v in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(v), ha="center", va="bottom", fontsize=11)
plt.title("各专业就业率对比 (%)", fontsize=14)
plt.ylabel("就业率 (%)")
plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/02c_bar_chart.png", dpi=150)
plt.close()
print("  ✅ plots/02c_bar_chart.png")

# 分组柱状图
groups = ["北京", "上海", "广州", "深圳", "杭州"]
scores_2024 = [85, 90, 78, 88, 82]
scores_2025 = [88, 92, 82, 91, 85]

x = np.arange(len(groups))
width = 0.35

plt.figure(figsize=(10, 5))
bars1 = plt.bar(x - width/2, scores_2024, width, label="2024年", color=COLORS[0])
bars2 = plt.bar(x + width/2, scores_2025, width, label="2025年", color=COLORS[1])
plt.xlabel("城市")
plt.ylabel("满意度评分")
plt.title("不同城市用户满意度对比", fontsize=14)
plt.xticks(x, groups)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/02d_grouped_bar.png", dpi=150)
plt.close()
print("  ✅ plots/02d_grouped_bar.png")


# --- 2c. 散点图（相关性） ---
print("\n--- 2c. 散点图 ---")

np.random.seed(42)
n = 80
hours = np.random.uniform(1, 10, n)
scores = 40 + 5 * hours + np.random.randn(n) * 8  # 正相关

plt.figure(figsize=(8, 6))
plt.scatter(hours, scores, c=COLORS[0], alpha=0.6, edgecolors="white", linewidth=0.5, s=60)
plt.title("学习时间 vs 考试成绩", fontsize=14)
plt.xlabel("每天学习时间 (小时)")
plt.ylabel("考试成绩")
plt.grid(True, alpha=0.3)

# 添加趋势线
z = np.polyfit(hours, scores, 1)
p = np.poly1d(z)
x_sorted = np.sort(hours)
plt.plot(x_sorted, p(x_sorted), "--", color=COLORS[1], linewidth=2, label="趋势线")
plt.legend()
plt.tight_layout()
plt.savefig("plots/02e_scatter.png", dpi=150)
plt.close()
print("  ✅ plots/02e_scatter.png")


# --- 2d. 直方图（分布） ---
print("\n--- 2d. 直方图 ---")

np.random.seed(42)
heights = np.random.normal(170, 6, 1000)  # 身高分布

plt.figure(figsize=(8, 5))
n, bins, patches = plt.hist(heights, bins=30, color=COLORS[0], alpha=0.7,
                            edgecolor="white", density=True)
plt.title("成年男性身高分布", fontsize=14)
plt.xlabel("身高 (cm)")
plt.ylabel("概率密度")
plt.grid(alpha=0.3)

# 叠加正态分布曲线（手动计算，无需 scipy）
def norm_pdf(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
x_range = np.linspace(heights.min(), heights.max(), 200)
plt.plot(x_range, norm_pdf(x_range, heights.mean(), heights.std()),
         "--", color=COLORS[2], linewidth=2, label="理论正态分布")
plt.legend()
plt.tight_layout()
plt.savefig("plots/02f_histogram.png", dpi=150)
plt.close()
print("  ✅ plots/02f_histogram.png")


# --- 2e. 饼图（占比） ---
print("\n--- 2e. 饼图 ---")

labels = ["计算机", "数学", "物理", "化学", "生物"]
sizes = [35, 20, 18, 15, 12]
explode = (0.05, 0, 0, 0, 0)  # 突出计算机

plt.figure(figsize=(8, 6))
wedges, texts, autotexts = plt.pie(
    sizes, labels=labels, autopct="%1.1f%%", startangle=90,
    colors=COLORS[:5], explode=explode, shadow=True,
    textprops={"fontsize": 12}
)
plt.title("各专业学生占比", fontsize=14)
plt.tight_layout()
plt.savefig("plots/02g_pie_chart.png", dpi=150)
plt.close()
print("  ✅ plots/02g_pie_chart.png")


# --- 2f. 箱线图（分布+异常值） ---
print("\n--- 2f. 箱线图 ---")

np.random.seed(42)
data_a = np.random.normal(75, 8, 100)
data_b = np.random.normal(80, 12, 100)
data_c = np.random.normal(70, 5, 100)

plt.figure(figsize=(8, 6))
bp = plt.boxplot([data_a, data_b, data_c],
                 patch_artist=True,
                 boxprops=dict(linewidth=1.5),
                 medianprops=dict(color="red", linewidth=2),
                 flierprops=dict(marker="o", markerfacecolor=COLORS[0], markersize=5))
plt.xticks([1, 2, 3], ["A 班", "B 班", "C 班"])
colors = [COLORS[0], COLORS[1], COLORS[2]]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
plt.title("不同班级成绩分布对比", fontsize=14)
plt.ylabel("成绩")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/02h_boxplot_mpl.png", dpi=150)
plt.close()
print("  ✅ plots/02h_boxplot_mpl.png")


# =====================================================================
# 3. 多子图布局（Subplots）
# =====================================================================
print("\n" + "=" * 60)
print("3. 多子图布局 — 一张图展示多个视角")
print("=" * 60)

np.random.seed(42)
data = np.random.randn(500)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("同一组数据的四种展示方式", fontsize=14, y=1.02)

# 子图 1：折线图
axes[0, 0].plot(data[:100], color=COLORS[0], linewidth=1)
axes[0, 0].set_title("时序折线图")
axes[0, 0].grid(alpha=0.3)

# 子图 2：直方图
axes[0, 1].hist(data, bins=30, color=COLORS[1], alpha=0.7, edgecolor="white")
axes[0, 1].set_title("分布直方图")
axes[0, 1].grid(alpha=0.3)

# 子图 3：箱线图
axes[1, 0].boxplot(data, patch_artist=True,
                   boxprops=dict(facecolor=COLORS[2], alpha=0.6))
axes[1, 0].set_title("箱线图")
axes[1, 0].grid(axis="y", alpha=0.3)

# 子图 4：累计分布
sorted_data = np.sort(data)
cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
axes[1, 1].plot(sorted_data, cumulative, color=COLORS[3], linewidth=2)
axes[1, 1].set_title("累计分布")
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots/03_subplots.png", dpi=150)
plt.close()
print("  ✅ plots/03_subplots.png")


# =====================================================================
# 4. 样式美化
# =====================================================================
print("\n" + "=" * 60)
print("4. 如何让图表更好看")
print("=" * 60)

x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)

# 美化的版本（用完后恢复字体）
plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams["font.family"] = used_font
matplotlib.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, y1, label="sin(x)", color=COLORS[0], linewidth=2.5, alpha=0.9)
ax.plot(x, y2, label="cos(x)", color=COLORS[1], linewidth=2.5, alpha=0.9)
ax.plot(x, y3, label="sin(x)cos(x)", color=COLORS[2], linewidth=2, alpha=0.7, linestyle="--")

ax.set_title("三角函数可视化", fontsize=16, pad=15)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.legend(fontsize=11, frameon=True, shadow=True, loc="upper right")
ax.set_xlim(0, 2 * np.pi)
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
ax.grid(True, alpha=0.3)

# 突出 y=0 轴
ax.axhline(y=0, color="gray", linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig("plots/04_styled_chart.png", dpi=150)
plt.close()

# 恢复默认样式
plt.style.use("default")
print("  ✅ plots/04_styled_chart.png")

# 可用的 style 列表
print("  可用样式:", plt.style.available[:10], "...")  # 只列前 10 个


# =====================================================================
# 5. Seaborn 入门
# =====================================================================
print("\n" + "=" * 60)
print("5. Seaborn — 统计专用可视化库")
print("=" * 60)
print("  Seaborn 基于 Matplotlib，封装了统计图表，一行代码出图\n")

sns.set_theme(style="whitegrid", font=used_font, palette="muted")

# 准备数据
np.random.seed(42)
df = pd.DataFrame({
    "科目": np.random.choice(["数学", "物理", "化学", "英语"], 200),
    "成绩": np.random.normal(75, 12, 200),
    "性别": np.random.choice(["男", "女"], 200),
    "班级": np.random.choice(["A", "B", "C"], 200),
    "学习时间": np.random.uniform(1, 10, 200)
})
df["成绩"] = df["成绩"].clip(0, 100).round(1)
# 让成绩和学习时间有正相关
df["成绩"] += df["学习时间"] * 2
df["成绩"] = df["成绩"].clip(0, 100).round(1)

print("  模拟数据:\n", df.head(), "\n")

# --- 5a. 分布图（histplot / kdeplot）---
print("\n--- 5a. 分布图 ---")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(df["成绩"], bins=25, kde=True, color=COLORS[0], ax=axes[0])
axes[0].set_title("成绩分布直方图 (含KDE)", fontsize=13)

sns.kdeplot(df["成绩"], fill=True, color=COLORS[0], ax=axes[1])
axes[1].set_title("成绩密度曲线", fontsize=13)

plt.tight_layout()
plt.savefig("plots/05a_distribution.png", dpi=150)
plt.close()
print("  ✅ plots/05a_distribution.png")


# --- 5b. 箱线图 + 蜂群图（类别vs数值）---
print("\n--- 5b. 类别对比图 ---")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(data=df, x="科目", y="成绩", palette="Set2", ax=axes[0])
axes[0].set_title("各科目成绩箱线图", fontsize=13)

sns.violinplot(data=df, x="科目", y="成绩", palette="Set2", ax=axes[1])
axes[1].set_title("各科目成绩小提琴图", fontsize=13)

plt.tight_layout()
plt.savefig("plots/05b_category_compare.png", dpi=150)
plt.close()
print("  ✅ plots/05b_category_compare.png")


# --- 5c. 热力图（相关性矩阵）---
print("\n--- 5c. 热力图 ---")

# 构造多列数据用于计算相关性
df_corr = pd.DataFrame({
    "身高": np.random.normal(170, 6, 100),
    "体重": np.random.normal(65, 10, 100),
    "年龄": np.random.randint(18, 30, 100),
    "成绩": np.random.normal(75, 15, 100),
    "学习时间": np.random.uniform(1, 10, 100),
})
# 让某些变量间有相关性
df_corr["体重"] = df_corr["身高"] * 0.5 + np.random.randn(100) * 5
df_corr["成绩"] = df_corr["学习时间"] * 5 + np.random.randn(100) * 10

corr_matrix = df_corr.corr()

plt.figure(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlBu_r", center=0, square=True,
            linewidths=1, cbar_kws={"shrink": 0.8})
plt.title("变量相关性热力图", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("plots/05c_heatmap.png", dpi=150)
plt.close()
print("  ✅ plots/05c_heatmap.png")


# --- 5d. 散点图 + 回归线 ---
print("\n--- 5d. 回归散点图 ---")

plt.figure(figsize=(8, 6))
sns.regplot(data=df, x="学习时间", y="成绩", scatter_kws={"alpha": 0.5, "color": COLORS[0]},
            line_kws={"color": COLORS[1], "linewidth": 2})
plt.title("学习时间 vs 成绩 (含回归线)", fontsize=14)
plt.tight_layout()
plt.savefig("plots/05d_regplot.png", dpi=150)
plt.close()
print("  ✅ plots/05d_regplot.png")


# --- 5e. 分类散点图（stripplot / swarmplot）---
print("\n--- 5e. 分类散点图 ---")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.stripplot(data=df, x="班级", y="成绩", hue="性别", dodge=True, palette="Set2", ax=axes[0])
axes[0].set_title("Stripplot 分类散点", fontsize=13)

sns.swarmplot(data=df, x="班级", y="成绩", hue="性别", dodge=True, palette="Set2", ax=axes[1])
axes[1].set_title("Swarmplot 分类散点（不重叠）", fontsize=13)

plt.tight_layout()
plt.savefig("plots/05e_categorical_scatter.png", dpi=150)
plt.close()
print("  ✅ plots/05e_categorical_scatter.png")


# --- 5f. 面网格（FacetGrid / lmplot）---
print("\n--- 5f. 分面（按类别分开展示）---")

# 如果性别列是中文，FacetGrid 的 hue 可能冲突，先复制一份英文
g = sns.lmplot(data=df, x="学习时间", y="成绩", col="班级", hue="性别",
               palette="Set2", height=4, aspect=1.2, scatter_kws={"alpha": 0.5})
g.fig.suptitle("各班级学习时间 vs 成绩关系", y=1.02, fontsize=14)
plt.savefig("plots/05f_facet.png", dpi=150)
plt.close()
print("  ✅ plots/05f_facet.png")


# --- 5g. pairplot ---
print("\n--- 5g. PairPlot（快速浏览变量关系）---")

# 用少量数据避免图太大
df_sample = df[["成绩", "学习时间", "性别"]].sample(50, random_state=42)
g = sns.pairplot(df_sample, hue="性别", palette="Set2",
                 diag_kind="kde", plot_kws={"alpha": 0.6, "s": 30})
g.fig.suptitle("PairPlot — 快速了解变量间关系", y=1.02, fontsize=14)
plt.savefig("plots/05g_pairplot.png", dpi=150)
plt.close()
print("  ✅ plots/05g_pairplot.png")


# =====================================================================
# 6. 实战：电影评分数据可视化
# =====================================================================
print("\n" + "=" * 60)
print("6. 实战：电影评分数据可视化")
print("=" * 60)

# 模拟电影评分数据
np.random.seed(42)
n = 500
movies_df = pd.DataFrame({
    "电影": np.random.choice(["肖申克的救赎", "霸王别姬", "星际穿越", "千与千寻",
                              "盗梦空间", "阿甘正传", "教父", "泰坦尼克号"], n),
    "评分": np.random.normal(8.5, 1.2, n).clip(0, 10).round(1),
    "观众年龄": np.random.randint(10, 60, n),
    "性别": np.random.choice(["男", "女"], n),
    "评分时间": pd.date_range("2026-01-01", periods=n, freq="h")
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("电影评分多维分析", fontsize=16, y=1.02)

# 图1：各电影评分分布（箱线图）
sns.boxplot(data=movies_df, x="电影", y="评分", palette="Set2", ax=axes[0, 0])
axes[0, 0].set_title("各电影评分分布")
axes[0, 0].tick_params(axis="x", rotation=30)

# 图2：年龄 vs 评分（散点图）
sns.scatterplot(data=movies_df, x="观众年龄", y="评分", hue="性别",
                alpha=0.5, palette="Set2", ax=axes[0, 1])
axes[0, 1].set_title("年龄与评分关系")
axes[0, 1].legend()

# 图3：评分直方图
sns.histplot(data=movies_df, x="评分", bins=25, kde=True,
             color=COLORS[0], ax=axes[1, 0])
axes[1, 0].set_title("评分分布")
axes[1, 0].axvline(movies_df["评分"].mean(), color="red",
                   linestyle="--", label=f"均值={movies_df['评分'].mean():.2f}")
axes[1, 0].legend()

# 图4：性别评分对比
sns.violinplot(data=movies_df, x="性别", y="评分", palette="Set2",
               inner="quartile", ax=axes[1, 1])
axes[1, 1].set_title("男女评分差异")
axes[1, 1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plots/06_movie_analysis.png", dpi=150)
plt.close()
print("  ✅ plots/06_movie_analysis.png")


# =====================================================================
# 7. 电影受欢迎程度排名
# =====================================================================
print("\n--- 额外图表: 电影评分排名 ---")

movie_stats = movies_df.groupby("电影").agg(
    平均评分=("评分", "mean"),
    评分人数=("评分", "count")
).reset_index()

# 柱状图 — 按评分人数排序
movie_stats = movie_stats.sort_values("评分人数", ascending=True)

plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(movie_stats)), movie_stats["评分人数"],
                color=COLORS[:len(movie_stats)])
plt.yticks(range(len(movie_stats)), movie_stats["电影"])
plt.xlabel("评分人数")
plt.title("电影受欢迎程度排名（按评分人数）", fontsize=14)
for bar, row in zip(bars, movie_stats.itertuples()):
    plt.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
             f" {row.平均评分:.2f}分", va="center", fontsize=10)
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/07_movie_ranking.png", dpi=150)
plt.close()
print("  ✅ plots/07_movie_ranking.png")


# =====================================================================
# 8. 动画演示（基础）
# =====================================================================
print("\n" + "=" * 60)
print("8. 动画演示（基础）")
print("=" * 60)

from matplotlib.animation import FuncAnimation

# 生成一个简单的正弦波动画
fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(0, 2 * np.pi, 200)
line, = ax.plot(x, np.sin(x), color=COLORS[0], linewidth=2)
ax.set_ylim(-1.5, 1.5)
ax.set_title("动态正弦波")
ax.grid(alpha=0.3)

def update(frame):
    line.set_ydata(np.sin(x + frame * 0.1))
    return line,

ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
ani.save("plots/08_animation.gif", writer="pillow", fps=20)
plt.close()
print("  ✅ plots/08_animation.gif")


# =====================================================================
# 9. 按需出图技巧
# =====================================================================
print("\n" + "=" * 60)
print("9. 按需出图技巧 — 什么场景用什么图")
print("=" * 60)

print("""
📊 图表选择指南

趋势变化 ──→ 折线图 (plot)
  例：每日用户量、股价走势、温度变化

类别对比 ──→ 柱状图 (bar)
  例：各产品销量、各国 GDP、各科成绩

数据分布 ──→ 直方图 / KDE 图 / 箱线图
  例：身高分布、收入分布、成绩分布
  箱线图还能看出异常值

相关性 ──→ 散点图 + 回归线
  例：学习时间 vs 成绩、广告费 vs 销量
  推荐用 sns.regplot()

占比 ──→ 饼图 / 环形图
  例：市场份额、预算分配
  注意：类别不要超过 5 个

多变量关系 ──→ 热力图 (关联) / pairplot (总览)
  例：房价影响因素分析
  sns.heatmap(corr) 看关联
  sns.pairplot(df) 总览全貌

分组统计 ──→ 箱线图 / 小提琴图 / 柱状图+误差线
  例：各班级成绩、各站点降雨量
  小提琴图比箱线图更"精细"

时间序列 ──→ 折线图 + 滚动平均
  例：股价 MA7、MA30
""")

print("plots/ 目录下共保存以下图表：")
for f in sorted(os.listdir("plots")):
    size = os.path.getsize(f"plots/{f}")
    print(f"  📊 {f}  ({size//1024} KB)")


# =====================================================================
# 10. 课后练习
# =====================================================================
print("\n" + "=" * 60)
print("10. 课后练习")
print("=" * 60)

print("""
题目 1: 天气数据可视化
  生成 2026 年全年的日均温度（模拟数据）：
    - 夏高冬低（用正弦波 + 随机噪声）
    - 画出全年的温度变化折线图
    - 加一个月均温的柱状图子图

题目 2: 多组对比
  用之前学的 NumPy 生成三个班的成绩数据（每个班 50 人），
  用箱线图对比三个班的成绩分布，并标注均值。

题目 3: 相关性分析
  创建包含 5 个变量（如房价、面积、房龄、卧室数、距地铁站距离）
  的模拟数据，画热力图展示它们的相关性。

题目 4: 综合项目
  用 Seaborn 的 FacetGrid 或 pairplot 分析以下场景：
  "不同年龄段、不同性别的人群，在不同时间段的 APP 使用时长"
  数据自己模拟（200 条以上）

题目 5: 样式挑战
  选一个你喜欢的配色方案，画一个带中文标题、网格线、
  图例、标注的自定义图表（自由发挥）
""")

print("\n✅ 数据可视化教程结束！接下来可以继续 Phase 4: 机器学习")
