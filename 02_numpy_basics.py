"""
Step 2 - NumPy 系统教程
=======================
运行: python3 02_numpy_basics.py

NumPy 是 Python 数据科学生态的基石。几乎所有数据科学库（Pandas、
scikit-learn、PyTorch 等）底层都依赖 NumPy 的 ndarray 数据结构。

学习目标:
  1. 理解 ndarray 和 Python list 的区别
  2. 掌握数组创建、索引、切片
  3. 理解广播机制和向量化运算
  4. 掌握常用统计和线性代数操作
"""

import numpy as np

# =====================================================================
# 1. 为什么用 NumPy？—— ndarray vs Python list
# =====================================================================
print("=" * 60)
print("1. ndarray vs Python list")
print("=" * 60)

# Python list 可以做数值运算，但很慢（底层是 Python 对象指针）
py_list = [1, 2, 3, 4, 5]
print("Python list:", py_list)
print("  类型:", type(py_list))
print("  占用:", py_list.__sizeof__(), "bytes（仅含指针）")

# ndarray：同质、连续内存、支持向量化运算
arr = np.array([1, 2, 3, 4, 5])
print("ndarray:", arr)
print("  类型:", type(arr))
print("  占用:", arr.nbytes, "bytes（连续存储）")

# 核心区别：ndarray 直接对每个元素操作（向量化），无需 for 循环
print("\n两倍运算（无需写循环）:")
print("  py_list * 2:", py_list * 2)   # list 是重复
print("  arr * 2:    ", arr * 2)       # 数组是逐个元素乘


# =====================================================================
# 2. 创建数组
# =====================================================================
print("\n" + "=" * 60)
print("2. 创建数组")
print("=" * 60)

# (a) 从 list / tuple 转换
a1 = np.array([10, 20, 30])
a2 = np.array([[1, 2, 3], [4, 5, 6]])  # 二维
print("一维:", a1)
print("二维:\n", a2)

# 指定数据类型
a3 = np.array([1, 2, 3], dtype=np.float32)
print("float32:", a3, a3.dtype)

# (b) 常用创建函数
print("\n--- 常用创建函数 ---")
print("np.zeros((2,3)):\n", np.zeros((2, 3), dtype=int))
print("np.ones((2,3)):\n", np.ones((2, 3)))
print("np.eye(4):\n", np.eye(4))      # 单位矩阵
print("np.arange(0, 10, 2):", np.arange(0, 10, 2))  # 类似 range
print("np.linspace(0, 1, 5):", np.linspace(0, 1, 5))  # 等间隔取数

# (c) 随机数
print("\n--- 随机数 ---")
np.random.seed(42)  # 固定随机种子，结果可复现
print("均匀分布 [0,1):", np.random.rand(3))        # 均匀分布
print("标准正态:", np.random.randn(3))              # 标准正态分布
print("随机整数 [0,10):", np.random.randint(0, 10, (2, 3)))
print("从 [10,20,30] 抽一个:", np.random.choice([10, 20, 30]))


# =====================================================================
# 3. 数组属性
# =====================================================================
print("\n" + "=" * 60)
print("3. 数组属性")
print("=" * 60)

arr_2d = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])
print("数组:\n", arr_2d)
print("ndim (维数):", arr_2d.ndim)          # 2
print("shape (形状):", arr_2d.shape)        # (3, 4)
print("size (元素总数):", arr_2d.size)      # 12
print("dtype (数据类型):", arr_2d.dtype)     # int64
print("nbytes (占用字节):", arr_2d.nbytes)   # 96 = 12 * 8


# =====================================================================
# 4. 索引和切片
# =====================================================================
print("\n" + "=" * 60)
print("4. 索引和切片")
print("=" * 60)

arr = np.arange(10)  # [0 1 2 3 4 5 6 7 8 9]
print("原始:", arr)

# 和 list 一样
print("arr[3]:", arr[3])
print("arr[-1]:", arr[-1])
print("arr[2:8:2]:", arr[2:8:2])   # [2 4 6]

# 二维索引
mat = np.arange(1, 13).reshape(3, 4)
print("\n3x4 矩阵:\n", mat)
print("mat[1, 2]:", mat[1, 2])     # 第2行第3列 = 7
print("mat[0]:", mat[0])           # 第1行
print("mat[:, 1]:", mat[:, 1])     # 第2列
print("mat[1:, 2:]:\n", mat[1:, 2:])  # 子矩阵

# 花式索引（用整数列表或数组）
print("\n花式索引:")
indices = [0, 2, 3]
print("arr[[0,2,3]]:", arr[indices])
print("mat[[0,2]]:\n", mat[[0, 2]])  # 取第1、3行

# 布尔索引（重点！数据分析中最常用）
print("\n布尔索引:")
ages = np.array([12, 25, 18, 33, 8, 41, 19])
print("ages:", ages)
print("ages > 18:", ages > 18)
print("ages[ages > 18]:", ages[ages > 18])  # 筛选成年人

# 多条件筛选（用 & | ~ 代替 and or not）
print("age >= 18 & age <= 30:",
      ages[(ages >= 18) & (ages <= 30)])  # [25 18 19]


# =====================================================================
# 5. 向量化运算（告别 for 循环）
# =====================================================================
print("\n" + "=" * 60)
print("5. 向量化运算")
print("=" * 60)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("a:", a)
print("b:", b)
print("a + b:", a + b)        # [5 7 9]
print("a - b:", a - b)        # [-3 -3 -3]
print("a * b:", a * b)        # [4 10 18]  元素乘（不是矩阵乘）
print("a / b:", a / b)        # [0.25 0.4  0.5]
print("a ** 2:", a ** 2)      # [1 4 9]
print("np.sqrt(a):", np.sqrt(a))  # [1. 1.414 1.732]
print("np.exp(a):", np.exp(a))    # e 的次方

# Python 标量和数组运算
print("\na + 10:", a + 10)      # 广播（后面讲）
print("np.sin(a):", np.sin(a))

# 比较运算返回布尔数组
print("a > 2:", a > 2)          # [False False  True]


# =====================================================================
# 6. 广播机制（Broadcasting） — 核心概念
# =====================================================================
print("\n" + "=" * 60)
print("6. 广播机制（Broadcasting）")
print("=" * 60)

# 核心规则：当两个数组 shape 不同时，NumPy 会自动"广播"小数组
# 条件：从后往前比较维度，要么相同，要么有一个是 1，要么缺失

print("标量 + 数组: [1,2,3] + 10 =", np.array([1, 2, 3]) + 10)
# 相当于 10 → [10, 10, 10]

print("\n每行都加一个向量:")
matrix = np.arange(12).reshape(3, 4)
row_vec = np.array([1, 0, -1, 2])
print("matrix:\n", matrix)
print("row_vec:", row_vec)
print("matrix + row_vec:\n", matrix + row_vec)  # 每行都加 row_vec

print("\n列向量广播:")
col_vec = np.array([[1], [2], [3]])  # shape (3,1)
print("shape matrix:", matrix.shape, ", col_vec:", col_vec.shape)
print("matrix + col_vec:\n", matrix + col_vec)

# 实用举例：数据标准化 (z-score)
print("\n实用：数据标准化 (z-score):")
data = np.array([[80, 70, 90],
                  [75, 80, 85],
                  [90, 85, 95]])
mean = data.mean(axis=0)    # 每列均值
std = data.std(axis=0)      # 每列标准差
print("原始数据:\n", data)
print("每列均值:", mean)
print("每列标准差:", std)
normalized = (data - mean) / std  # 广播！
print("标准化后:\n", normalized)


# =====================================================================
# 7. 统计函数
# =====================================================================
print("\n" + "=" * 60)
print("7. 统计函数")
print("=" * 60)

data = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
print("数据:\n", data)
print("sum (全部和):", np.sum(data))          # 45
print("mean (平均值):", np.mean(data))         # 5.0
print("std (标准差):", np.std(data))           # 2.58...
print("min:", np.min(data))
print("max:", np.max(data))

# axis 参数：0=按列（跨行），1=按行（跨列）
print("\naxis 参数:")
print("按列求和 axis=0:", np.sum(data, axis=0))          # [12 15 18]
print("按行求和 axis=1:", np.sum(data, axis=1))          # [6 15 24]
print("每列均值 axis=0:", np.mean(data, axis=0))          # [4. 5. 6.]

# 其他实用函数
print("argmax (最大值索引):", np.argmax(data))            # 8
print("argmax axis=1 (每行最大索引):", np.argmax(data, axis=1))
print("cumsum (累积和):", np.cumsum(data))                # [1 3 6 10 15 ...]
print("clip (截断):", np.clip(data, 3, 7))                # 把<3变3，>7变7


# =====================================================================
# 8. 线性代数基础
# =====================================================================
print("\n" + "=" * 60)
print("8. 线性代数基础")
print("=" * 60)

# 矩阵乘法：@ 或 np.dot
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("A:\n", A)
print("B:\n", B)
print("A @ B (矩阵乘):\n", A @ B)           # Python 3.5+
print("np.dot(A, B):\n", np.dot(A, B))        # 旧写法

# 逐元素乘 vs 矩阵乘
print("\n逐元素乘 A * B:\n", A * B)           # 点乘
print("矩阵乘 A @ B:\n", A @ B)               # 叉乘

# 转置
print("\nA 的转置:\n", A.T)

# 逆矩阵
A_inv = np.linalg.inv(A)
print("A 的逆:\n", A_inv)
print("A @ A_inv (应近似单位矩阵):\n", A @ A_inv)

# 行列式
print("det(A):", np.linalg.det(A))

# 特征值、特征向量
eigvals, eigvecs = np.linalg.eig(A)
print("特征值:", eigvals)
print("特征向量:\n", eigvecs)

# 解线性方程组 Ax = b
# 2x + 3y = 8
# 5x + 2y = 9
A_sys = np.array([[2, 3], [5, 2]])
b_sys = np.array([8, 9])
x_sol = np.linalg.solve(A_sys, b_sys)
print("\n解方程组:")
print(f"  2x+3y=8, 5x+2y=9 → x={x_sol[0]:.2f}, y={x_sol[1]:.2f}")


# =====================================================================
# 9. 重塑、拼接、分割
# =====================================================================
print("\n" + "=" * 60)
print("9. 重塑、拼接、分割")
print("=" * 60)

arr = np.arange(12)
print("原始 (12,):", arr)

# reshape：重塑形状（总元素数不变）
reshaped = arr.reshape(3, 4)
print("reshape(3,4):\n", reshaped)

# flatten / ravel：展平
print("flatten:", reshaped.flatten())

# 拼接
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print("\nvstack (垂直拼接):\n", np.vstack([a, b]))   # 加行
print("hstack (水平拼接):\n", np.hstack([a, b]))   # 加列
print("concatenate axis=0:\n", np.concatenate([a, b], axis=0))
print("concatenate axis=1:\n", np.concatenate([a, b], axis=1))

# 分割
big = np.arange(12).reshape(3, 4)
print("\nsplit:\nbig:\n", big)
print("hsplit (竖切两半):", np.hsplit(big, 2))
print("vsplit (横切):", np.vsplit(big, 3))


# =====================================================================
# 10. 实用小技巧
# =====================================================================
print("\n" + "=" * 60)
print("10. 实用小技巧")
print("=" * 60)

# (a) where：条件筛选
scores = np.array([65, 80, 45, 90, 55, 70])
passed = np.where(scores >= 60, "通过", "挂科")
print("scores:", scores)
print("np.where:", passed)

# (b) unique：去重
tags = np.array(['a', 'b', 'a', 'c', 'b', 'a'])
print("unique:", np.unique(tags))
print("带计数:", np.unique(tags, return_counts=True))

# (c) sort 排序
unsorted = np.array([3, 1, 7, 4, 2, 9])
print("sorted:", np.sort(unsorted))
# 原地排序: unsorted.sort()

# (d) save / load
np.savez("temp_demo.npz", data=arr)
loaded = np.load("temp_demo.npz")
print("save/load 测试:", loaded["data"])
import os; os.remove("temp_demo.npz")

# (e) 设置打印选项
np.set_printoptions(precision=2, suppress=True)


# =====================================================================
# 11. 实战练习
# =====================================================================
print("\n" + "=" * 60)
print("11. 实战练习 —— 实现简单线性回归")
print("=" * 60)

"""
用 NumPy 实现一元线性回归（最小二乘法）:
  y = w * x + b

公式:
  w = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
  b = (Σy - w*Σx) / n
"""

print("\n题目: 房价预测")
print("-" * 40)

# 训练数据：房屋面积（m²）和价格（万元）
x = np.array([50, 60, 70, 80, 90, 100, 110, 120])
y = np.array([150, 180, 210, 240, 280, 310, 340, 370])

print("房屋面积 (m²):", x)
print("价格 (万元):", y)

# 用最小二乘法计算 w, b
n = len(x)
w = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
    (n * np.sum(x ** 2) - np.sum(x) ** 2)
b = (np.sum(y) - w * np.sum(x)) / n

print(f"\n回归结果: y = {w:.4f}x + {b:.4f}")
print(f"解释: 每增加 1 m², 房价增加约 {w:.1f} 万元")

# 预测
x_new = np.array([65, 95, 130])
y_pred = w * x_new + b
print(f"\n预测:")
for sq, price in zip(x_new, y_pred):
    print(f"  {sq} m² → {price:.1f} 万元")

# 评估：R² 决定系数
y_pred_all = w * x + b
ss_res = np.sum((y - y_pred_all) ** 2)  # 残差平方和
ss_tot = np.sum((y - np.mean(y)) ** 2)  # 总平方和
r2 = 1 - ss_res / ss_tot
print(f"\nR² = {r2:.4f} (越接近 1 说明拟合越好)")


# =====================================================================
# 12. 更多练习（自己做）
# =====================================================================
print("\n" + "=" * 60)
print("12. 课后练习")
print("=" * 60)

print("""
题目 1: 矩阵运算
  创建两个 5×5 的随机矩阵 A 和 B（整数 0-9），计算：
    a) A + B, A - B, A * B (逐元素乘)
    b) A @ B (矩阵乘)
    c) A 的转置和逆（如果有）
    d) 找出 A 中 > 5 的元素个数

题目 2: 数据筛选
  生成 100 个服从正态分布（均值=170，标准差=6）的身高数据，
  模拟「成年男性身高」。
    a) 计算平均身高、标准差
    b) 找出身高 > 180 的人的索引
    c) 计算身高在 [165, 175] 范围内的人占比

题目 3: 归一化
  生成 shape 为 (10, 4) 的随机数据（10 个样本，4 个特征）。
  用不同方法归一化：
    a) Min-Max 归一化: (x - min) / (max - min)
    b) Z-score 标准化: (x - mean) / std

题目 4: 距离矩阵
  给定 5 个点（二维坐标），计算任意两点间的欧氏距离矩阵。
  提示: 用广播 + (x1-x2)² + (y1-y2)² 向量化实现
""")

print("\n✅ NumPy 教程结束！接下来学习 03_pandas_basics.py")
