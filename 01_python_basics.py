"""
Step 1 - Python 数据科学基础练习
==============================
运行: python3 01_python_basics.py
"""

# ========== 练习 1: 列表和平方数 ==========
print("=" * 40)
print("练习 1: 列表和平方数")
print("=" * 40)

# 题目: 创建一个列表，包含 1 到 10 的平方数
# 期望输出: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# ---- 在这里写你的代码 ----
squares = []
for i in range(1, 11):
    squares.append(i ** 2)
# -------------------------

print("结果:", squares)


# ========== 练习 2: 列表推导式 ==========
print("\n" + "=" * 40)
print("练习 2: 列表推导式")
print("=" * 40)

# 题目: 用一行代码（列表推导式）生成 1 到 20 中的偶数
# 期望输出: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# ---- 在这里写你的代码 ----
evens = [x for x in range(1, 21) if x % 2 == 0]
# -------------------------

print("偶数:", evens)


# ========== 练习 3: 字典操作 ==========
print("\n" + "=" * 40)
print("练习 3: 字典操作")
print("=" * 40)

# 题目: 有一个学生成绩单，请计算平均分
scores = {
    "张三": 85,
    "李四": 92,
    "王五": 78,
    "赵六": 95,
    "刘七": 88
}

# ---- 在这里写你的代码 ----
total = sum(scores.values())
avg = total / len(scores)
# -------------------------

print(f"成绩单: {scores}")
print(f"平均分: {avg:.1f}")


# ========== 练习 4: 函数 ==========
print("\n" + "=" * 40)
print("练习 4: 写一个函数")
print("=" * 40)

# 题目: 写一个函数 is_palindrome，判断字符串是否是回文（正着读反着读一样）
# 比如: "radar" -> True, "hello" -> False

# ---- 在这里写你的代码 ----
def is_palindrome(s):
    return s == s[::-1]
# -------------------------

print(f"is_palindrome('radar'): {is_palindrome('radar')}")
print(f"is_palindrome('hello'): {is_palindrome('hello')}")
print(f"is_palindrome('上海自来水来自海上'): {is_palindrome('上海自来水来自海上')}")


# ========== 练习 5: 文件读写 ==========
print("\n" + "=" * 40)
print("练习 5: 文件读写")
print("=" * 40)

# 题目: 把下面这段文本保存到文件，再读出来统计字数
text = "机器学习是人工智能的核心，自然语言处理是机器学习的重要应用方向。"

# ---- 在这里写你的代码 ----
with open("sample.txt", "w", encoding="utf-8") as f:
    f.write(text)

with open("sample.txt", "r", encoding="utf-8") as f:
    content = f.read()

char_count = len(content)
word_count = len(content.replace("，", "").replace("。", ""))
# -------------------------

print(f"原文: {content}")
print(f"总字符数: {char_count}")
print(f"实际字数（去标点）: {word_count}")
