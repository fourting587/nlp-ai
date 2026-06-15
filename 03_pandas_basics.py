"""
Step 3 - Pandas 系统教程
========================
运行: python3 03_pandas_basics.py

Pandas 是 Python 数据分析的核心库，提供了 DataFrame（类似 Excel 表格）
和 Series（类似一列数据）两种数据结构。做 NLP 项目时，几乎 80% 的数据
清洗和预处理工作都会用到 Pandas。

学习目标:
  1. 掌握 Series 和 DataFrame 的创建与基本操作
  2. 学会读写 CSV / Excel 等常见文件格式
  3. 掌握数据清洗（缺失值、重复值、类型转换）
  4. 掌握分组聚合（split-apply-combine）
  5. 掌握多表合并（merge / join）
"""

# =====================================================================
# 0. 安装（如果还没装的话）
# =====================================================================
# pip install pandas numpy

import pandas as pd
import numpy as np

# 显示设置：展示更多行列（方便学习）
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 100)
pd.set_option('display.max_colwidth', 30)


# =====================================================================
# 1. Series — 一列数据
# =====================================================================
print("=" * 60)
print("1. Series — 一列带索引的数据")
print("=" * 60)

# Series ≈ 带标签的数组（索引 + 数据）
s1 = pd.Series([85, 92, 78, 96, 88])
print("基本 Series:")
print(s1)
print("值:", s1.values, "| 索引:", s1.index)

# 自定义索引
s2 = pd.Series([85, 92, 78, 96, 88],
               index=["张三", "李四", "王五", "赵六", "刘七"],
               name="期末成绩")
print("\n带索引的 Series:")
print(s2)

# 从字典创建（key → index, value → data）
scores_dict = {"张三": 85, "李四": 92, "王五": 78}
s3 = pd.Series(scores_dict)
print("\n从字典创建:")
print(s3)

# 索引和切片
print("\ns2[\"李四\"]:", s2["李四"])           # 标签索引
print("s2.iloc[1]:", s2.iloc[1])             # 位置索引
print("s2[[\"张三\", \"赵六\"]]:\n", s2[["张三", "赵六"]])
print("s2 > 85:\n", s2[s2 > 85])             # 布尔索引

# Series 运算（向量化，保留索引）
print("\n运算（保留索引）:")
print("s2 + 5:\n", s2 + 5)                   # 所有成绩 + 5
print("s2.mean():", s2.mean())
print("s2.max():", s2.max())


# =====================================================================
# 2. DataFrame — 二维表格
# =====================================================================
print("\n" + "=" * 60)
print("2. DataFrame — 二维表格")
print("=" * 60)

# (a) 从字典创建（key → 列名, value → 数据）
df = pd.DataFrame({
    "姓名": ["张三", "李四", "王五", "赵六", "刘七"],
    "年龄": [21, 22, 20, 23, 21],
    "成绩": [85, 92, 78, 96, 88],
    "班级": ["A", "B", "A", "B", "A"]
})
print("从字典创建:")
print(df)

# (b) 从 NumPy 数组创建
df2 = pd.DataFrame(
    np.random.randint(0, 100, (5, 3)),
    columns=["语文", "数学", "英语"]
)
print("\n从 NumPy 创建:")
print(df2)

# (c) 查看数据
print("\n--- 常用查看方法 ---")
print("head(3):\n", df.head(3))
print("tail(2):\n", df.tail(2))
print("info():")
df.info()
print("\ndescribe():\n", df.describe())  # 只统计数值列

print("\nshape:", df.shape)       # (5, 4)
print("columns:", df.columns)    # Index(['姓名', '年龄', '成绩', '班级'])
print("dtypes:\n", df.dtypes)
print("values:\n", df.values)    # 底层 NumPy 数组


# =====================================================================
# 3. 索引与选择 — loc, iloc, []
# =====================================================================
print("\n" + "=" * 60)
print("3. 索引与选择 — loc, iloc, []")
print("=" * 60)

# 三大黄金法则：
#   df[列名]       → 选列
#   df.loc[行标签]  → 按标签选行和列
#   df.iloc[行位置] → 按位置选行和列

df = pd.DataFrame({
    "姓名": ["张三", "李四", "王五", "赵六", "刘七"],
    "年龄": [21, 22, 20, 23, 21],
    "成绩": [85, 92, 78, 96, 88],
    "班级": ["A", "B", "A", "B", "A"]
})
print("原始数据:")
print(df)

# 选列
print("\n--- 选列 ---")
print("df['姓名']:\n", df["姓名"])              # 返回 Series
print("df[['姓名', '成绩']]:\n", df[["姓名", "成绩"]])  # 返回 DataFrame

# 选行（iloc — 按位置）
print("\n--- 选行 (iloc, 按位置) ---")
print("df.iloc[0]:\n", df.iloc[0])                 # 第1行
print("df.iloc[1:3]:\n", df.iloc[1:3])              # 第2-3行
print("df.iloc[[0, 4]]:\n", df.iloc[[0, 4]])        # 第1、5行
print("df.iloc[1:3, 1:3]:\n", df.iloc[1:3, 1:3])   # 行+列切片

# 选行（loc — 按标签）
print("\n--- 选行 (loc, 按标签) ---")
# 这里行索引是默认 0,1,2,...，所以和 iloc 看起来一样
print("df.loc[0]:\n", df.loc[0])
print("df.loc[0:2]:\n", df.loc[0:2])   # 注意：loc 切片包含结尾！
print("df.loc[0:2, ['姓名', '成绩']]:\n", df.loc[0:2, ["姓名", "成绩"]])

# 布尔索引（最常用！）
print("\n--- 布尔索引 ---")
print("成绩 >= 90:\n", df[df["成绩"] >= 90])
print("年龄 < 22 且 成绩 > 80:\n",
      df[(df["年龄"] < 22) & (df["成绩"] > 80)])

# query() — 更简洁的写法
print("\nquery 写法:")
print(df.query("年龄 < 22 and 成绩 > 80"))

# 添加/修改列
df["性别"] = ["男", "女", "男", "女", "男"]
df["成绩加权"] = df["成绩"] * 1.1              # 向量化运算！
print("\n添加新列:\n", df)

# 删除列
df_dropped = df.drop("成绩加权", axis=1)       # axis=1 表示列
print("\n删除列:\n", df_dropped.head(3))


# =====================================================================
# 4. 处理缺失数据（数据清洗的核心）
# =====================================================================
print("\n" + "=" * 60)
print("4. 处理缺失数据")
print("=" * 60)

# 制造含缺失值的 DataFrame
df_missing = pd.DataFrame({
    "姓名": ["张三", "李四", "王五", "赵六", "刘七"],
    "年龄": [21, np.nan, 20, np.nan, 21],
    "成绩": [85, 92, np.nan, 96, 88],
    "班级": ["A", "B", np.nan, "B", "A"]
})
print("含缺失值的数据:")
print(df_missing)

# 检查缺失
print("\n--- 检查缺失 ---")
print("isnull():\n", df_missing.isnull())
print("每列缺失计数:\n", df_missing.isnull().sum())
print("总缺失数:", df_missing.isnull().sum().sum())

# 方法 1：删除缺失行/列
print("\n--- 删除缺失 ---")
print("dropna() 删除含缺失的行:\n", df_missing.dropna())
print("dropna(thresh=3) 至少3个非空才保留:\n", df_missing.dropna(thresh=3))

# 方法 2：填充缺失值
print("\n--- 填充缺失 ---")
print("fillna(0):\n", df_missing.fillna(0))                    # 填充固定值
print("年龄 fillna(mean):")
df_copy = df_missing.copy()
mean_age = df_copy["年龄"].mean()
print(f"  年龄列均值: {mean_age:.1f}")
print(df_copy["年龄"].fillna(mean_age))                         # 填充均值
print("ffill (前向填充):\n", df_missing.ffill())  # 前向填充

# 方法 3：插值
print("\n插值 interpolate:")
df_interp = df_missing.copy()
df_interp["成绩"] = df_interp["成绩"].interpolate()
print(df_interp)


# =====================================================================
# 5. 分组聚合（GroupBy）— split-apply-combine
# =====================================================================
print("\n" + "=" * 60)
print("5. 分组聚合（GroupBy）")
print("=" * 60)

# 制造更多数据
df_group = pd.DataFrame({
    "姓名": ["张三", "李四", "王五", "赵六", "刘七", "小红", "小明", "小刚"],
    "班级": ["A", "B", "A", "B", "A", "B", "A", "B"],
    "科目": ["数学", "数学", "英语", "英语", "数学", "英语", "数学", "英语"],
    "成绩": [85, 92, 78, 96, 88, 74, 91, 85]
})
print("数据:")
print(df_group)

# 按班级分组，求平均成绩
print("\n--- 基础分组 ---")
print("按班级 GroupBy 取 mean:")
print(df_group.groupby("班级")["成绩"].mean())

# 多列分组
print("\n按班级+科目分组:")
print(df_group.groupby(["班级", "科目"])["成绩"].mean())

# 多种聚合函数
print("\n多种聚合函数:")
print(df_group.groupby("班级")["成绩"].agg(["mean", "std", "min", "max"]))

# agg 可以对不同列用不同函数
print("\n不同列用不同聚合:")
print(df_group.groupby("班级").agg({
    "成绩": ["mean", "max"],
    "姓名": "count"
}))

# transform：保留原行数，但用分组值填充
print("\ntransform — 每行附加该组的均值:")
df_group["班级平均"] = df_group.groupby("班级")["成绩"].transform("mean")
print(df_group)


# =====================================================================
# 6. 数据合并 — merge, concat, join
# =====================================================================
print("\n" + "=" * 60)
print("6. 数据合并")
print("=" * 60)

# (a) concat — 简单拼接
df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
print("concat (按行拼接):")
print(pd.concat([df1, df2], axis=0))

# (b) merge — SQL 风格的 JOIN（重点！）
students = pd.DataFrame({
    "学号": ["S001", "S002", "S003", "S004"],
    "姓名": ["张三", "李四", "王五", "赵六"],
    "班级": ["A", "B", "A", "B"]
})
scores = pd.DataFrame({
    "学号": ["S001", "S002", "S001", "S003", "S005"],
    "科目": ["数学", "数学", "英语", "英语", "英语"],
    "成绩": [85, 92, 78, 96, 88]
})
print("\nstudents表:")
print(students)
print("\nscores表:")
print(scores)

# inner join（默认）
inner = pd.merge(students, scores, on="学号", how="inner")
print("\ninner join (只保留两表都有的学号):")
print(inner)

# left join
left = pd.merge(students, scores, on="学号", how="left")
print("\nleft join (保留学生表所有行):")
print(left)
# 注意：S004 没有成绩 → NaN

# right join
right = pd.merge(students, scores, on="学号", how="right")
print("\nright join (保留成绩表所有行):")
print(right)
# 注意：S005 在 students 表无对应 → 姓名 NaN

# outer join
outer = pd.merge(students, scores, on="学号", how="outer")
print("\nouter join (保留两表所有行):")
print(outer)


# =====================================================================
# 7. 数据重塑 — pivot, melt
# =====================================================================
print("\n" + "=" * 60)
print("7. 数据重塑")
print("=" * 60)

df_wide = pd.DataFrame({
    "姓名": ["张三", "李四", "王五"],
    "数学": [85, 92, 78],
    "英语": [88, 96, 82],
    "物理": [90, 85, 80]
})
print("宽表（宽格式 — 每个科目是一列）:")
print(df_wide)

# melt：宽表 → 长表
df_long = df_wide.melt(id_vars=["姓名"],
                       var_name="科目",
                       value_name="成绩")
print("\nmelt 后（长格式 — 每行一个观测）:")
print(df_long)

# pivot：长表 → 宽表
df_back = df_long.pivot(index="姓名", columns="科目", values="成绩")
print("\npivot 回去:")
print(df_back)


# =====================================================================
# 8. 文件读写
# =====================================================================
print("\n" + "=" * 60)
print("8. 文件读写")
print("=" * 60)

# 制造示例数据
df_export = pd.DataFrame({
    "姓名": ["张三", "李四", "王五", "赵六", "刘七"],
    "年龄": [21, 22, 20, 23, 21],
    "城市": ["北京", "上海", "广州", "深圳", "杭州"],
    "月薪": [8000, 12000, 9500, 15000, 11000]
})

# CSV
df_export.to_csv("temp_pandas_demo.csv", index=False, encoding="utf-8")
df_csv = pd.read_csv("temp_pandas_demo.csv")
print("从 CSV 读取:\n", df_csv)

# Excel（需要 pip install openpyxl）
# df_export.to_excel("temp_pandas_demo.xlsx", index=False)
# df_excel = pd.read_excel("temp_pandas_demo.xlsx")

# JSON
df_export.to_json("temp_pandas_demo.json", orient="records", force_ascii=False)
df_json = pd.read_json("temp_pandas_demo.json")
print("\n从 JSON 读取:\n", df_json)

# 清理临时文件
import os
os.remove("temp_pandas_demo.csv")
os.remove("temp_pandas_demo.json")


# =====================================================================
# 9. 常用数据清洗操作
# =====================================================================
print("\n" + "=" * 60)
print("9. 常用数据清洗操作")
print("=" * 60)

# 制造"脏数据"
df_dirty = pd.DataFrame({
    "姓名": ["张三", "李四", "王五", "赵六", "张三", "刘七"],
    "年龄": [21, 22, -1, 23, 21, None],
    "成绩": [85, 92, 78, 96, 85, 88],
    "城市": [" 北京 ", "上海", "gz", "深圳", " 北京 ", "杭州"]
})
print("原始脏数据:")
print(df_dirty)

# (1) 检查重复
print("\n--- 去重 ---")
print("duplicated:", df_dirty.duplicated())
print("duplicated(keep='first'):")
print(df_dirty[df_dirty.duplicated(keep="first")])
print("drop_duplicates:\n", df_dirty.drop_duplicates())

# (2) 类型转换
print("\n--- 类型转换 ---")
print("原始 dtypes:\n", df_dirty.dtypes)
df_dirty["年龄"] = pd.to_numeric(df_dirty["年龄"], errors="coerce")
print("to_numeric 后的年龄:", df_dirty["年龄"].tolist())

# (3) 替换异常值
print("\n--- 替换异常 ---")
df_dirty.loc[df_dirty["年龄"] < 0, "年龄"] = np.nan
print("负年龄→NaN:\n", df_dirty)

# (4) 字符串清理
print("\n--- 字符串清理 ---")
df_dirty["城市"] = df_dirty["城市"].str.strip()        # 去空格
df_dirty["城市"] = df_dirty["城市"].replace("gz", "广州")  # 替换简写
print(df_dirty)


# =====================================================================
# 10. 实战：电影评分数据分析
# =====================================================================
print("\n" + "=" * 60)
print("10. 实战：电影评分数据分析")
print("=" * 60)

# 模拟 MovieLens 风格数据
n = 200
np.random.seed(42)

ratings = pd.DataFrame({
    "user_id": np.random.randint(1, 51, n),      # 50 个用户
    "movie_id": np.random.randint(1, 21, n),     # 20 部电影
    "rating": np.random.randint(1, 6, n),         # 1-5 分
    "timestamp": pd.date_range("2024-01-01", periods=n, freq="h")
})

movies = pd.DataFrame({
    "movie_id": range(1, 21),
    "title": [
        "肖申克的救赎", "霸王别姬", "阿甘正传", "泰坦尼克号",
        "千与千寻", "盗梦空间", "星际穿越", "楚门的世界",
        "这个杀手不太冷", "美丽人生", "辛德勒的名单", "机器人总动员",
        "疯狂动物城", "寻梦环游记", "控方证人", "指环王",
        "教父", "蝙蝠侠：黑暗骑士", "十二怒汉", "飞越疯人院"
    ],
    "genre": ["剧情", "剧情", "剧情", "爱情",
              "动画", "科幻", "科幻", "剧情",
              "剧情", "剧情", "历史", "动画",
              "动画", "动画", "悬疑", "奇幻",
              "犯罪", "动作", "剧情", "剧情"]
})

print("评分数据 (head):")
print(ratings.head())
print("\n电影数据:")
print(movies)

# 分析 1：每部电影平均评分
print("\n--- 分析1: 每部电影平均评分 ---")
movie_avg = ratings.groupby("movie_id")["rating"].agg(["mean", "count"])
movie_avg = movie_avg.merge(movies[["movie_id", "title"]], on="movie_id")
print(movie_avg.sort_values("mean", ascending=False).head(10))

# 分析 2：评分最多（最热门）的电影
print("\n--- 分析2: 评分最多的电影（最热门） ---")
popular = ratings.groupby("movie_id")["rating"].count().reset_index()
popular = popular.rename(columns={"rating": "评分次数"})
popular = popular.merge(movies[["movie_id", "title"]], on="movie_id")
print(popular.sort_values("评分次数", ascending=False).head(5))

# 分析 3：各类型电影的评分分布
print("\n--- 分析3: 各类型电影评分 ---")
with_movies = ratings.merge(movies, on="movie_id", how="left")
genre_stats = with_movies.groupby("genre")["rating"].agg(["mean", "count", "std"])
print(genre_stats.sort_values("mean", ascending=False))

# 分析 4：每个用户的评分习惯
print("\n--- 分析4: 用户评分习惯 ---")
user_stats = ratings.groupby("user_id").agg(
    评分次数=("rating", "count"),
    平均分=("rating", "mean"),
    最高分=("rating", "max"),
    最低分=("rating", "min")
)
print(user_stats.describe())
print("\n最活跃的10个用户:")
print(user_stats.sort_values("评分次数", ascending=False).head(10))


# =====================================================================
# 11. apply 函数（管道操作）
# =====================================================================
print("\n" + "=" * 60)
print("11. apply 函数 — 灵活逐行/逐列处理")
print("=" * 60)

df_apply = pd.DataFrame({
    "姓名": ["张三", "李四", "王五", "赵六"],
    "语文": [85, 92, 78, 96],
    "数学": [90, 85, 80, 88],
    "英语": [88, 96, 82, 90]
})
print("数据:\n", df_apply)

# apply 到行（axis=1）
df_apply["平均分"] = df_apply[["语文", "数学", "英语"]].apply(
    lambda row: round(row.mean(), 1), axis=1
)
print("\n加平均分:\n", df_apply)

# apply 到列（axis=0）
print("\n各科描述统计:")
print(df_apply[["语文", "数学", "英语"]].apply(["mean", "std", "min", "max"]))

# applymap / map：逐元素
print("\n等级转换（map）:")
def score_level(s):
    if s >= 90: return "优秀"
    if s >= 80: return "良好"
    if s >= 70: return "中等"
    return "待提高"

df_level = df_apply.copy()
df_level["语文等级"] = df_level["语文"].map(score_level)
df_level["数学等级"] = df_level["数学"].map(score_level)
df_level["英语等级"] = df_level["英语"].map(score_level)
print(df_level[["姓名", "语文", "语文等级", "数学", "数学等级", "英语", "英语等级"]])


# =====================================================================
# 12. 课后练习
# =====================================================================
print("\n" + "=" * 60)
print("12. 课后练习")
print("=" * 60)

print("""
题目 1: 数据处理
  创建一个包含 100 条学生记录的 DataFrame：
    - 列: name, age, score, city, major
    - age 在 18-25 之间随机
    - score 在 40-100 之间随机（但要让约 10% 为 NaN）
    - city 从 ["北京","上海","广州","深圳","杭州"] 中随机
    - major 从 ["计算机","数学","物理","化学"] 中随机
  然后：
    a) 检查缺失值并处理（用均值填充缺失的 score）
    b) 统计每个城市的平均分
    c) 找出 score > 80 且 major 为计算机的学生
    d) 按专业统计人数和平均分

题目 2: 多表合并
  - orders.csv: order_id, customer_id, amount, date
  - customers.csv: customer_id, name, city, age
  模拟这两个表，用 merge 找出：
    a) 每个客户的消费总额
    b) 哪个城市的客户消费最多
    c) 每个月的销售总额

题目 3: 数据重塑
  用 pandas 实现「交叉表」— 统计每个班级中不同成绩段的人数：
    班级 \\ 成绩段  优秀 良好 及格 不及格
    A              12    8    3    1
    B              10   10    5    2
  提示：用 cut + groupby + unstack 或 pivot_table

题目 4: 时间序列（进阶）
  生成 90 天的销售数据，用 resample 按周和按月汇总：
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    sales = pd.Series(np.random.randint(100, 500, 90), index=dates)
    a) 按周汇总（resample('W')）
    b) 按月汇总（resample('M')）
    c) 计算 7 天滚动平均（rolling(7).mean()）
""")

print("\n✅ Pandas 教程结束！建议接下来学习 04_matplotlib_seaborn.py（数据可视化）")

# =====================================================================
# 附录：高频速查表
# =====================================================================
print("\n" + "=" * 60)
print("附录：高频速查表（考试/面试/写代码时快捷参考）")
print("=" * 60)

print("""
📋 Pandas 高频操作速查

读取文件:
  pd.read_csv('file.csv')
  pd.read_excel('file.xlsx')
  pd.read_json('file.json')

查看数据:
  df.head(), df.tail(), df.sample(5)
  df.info(), df.describe()
  df.shape, df.columns, df.dtypes
  df.nunique(), df['col'].value_counts()

选择数据:
  df['col']              → 选一列 (Series)
  df[['col1', 'col2']]   → 选多列 (DataFrame)
  df.loc[行标签, 列标签]  → 标签索引
  df.iloc[行位置, 列位置]  → 位置索引
  df[条件]                → 布尔索引
  df.query('条件式')       → 字符串查询

清洗数据:
  df.isnull().sum()      → 检查缺失
  df.dropna()            → 删除缺失
  df.fillna(value)       → 填充缺失
  df.drop_duplicates()   → 去重
  df['col'].str.strip()  → 字符串去空格
  pd.to_numeric()        → 转为数值

分组聚合:
  df.groupby('col')['val'].mean()
  df.groupby('col').agg({'val1': 'mean', 'val2': 'sum'})
  df.pivot_table(index='行', columns='列', values='值')

合并:
  pd.merge(df1, df2, on='key', how='left')
  pd.concat([df1, df2], axis=0/1)

排序 & 排名:
  df.sort_values('col', ascending=False)
  df.rank()

时间序列:
  pd.to_datetime(df['date'])
  df.set_index('date').resample('W').mean()
  df['val'].rolling(7).mean()
""")
