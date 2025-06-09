# Python与Git基础课程笔记

## 1. 环境准备

```bash
# 检查Python版本
python --version

# 创建并激活虚拟环境
python -m venv myenv

# Windows激活
myenv\Scripts\activate

# Linux/Mac激活
source myenv/bin/activate

# 安装第三方库
pip install requests
```  
## 2. 变量、变量类型、作用域

### 核心知识点
- **基本类型**：`int`、`float`、`str`、`bool`、`list`、`tuple`、`dict`、`set`
- **作用域**：全局变量、局部变量，`global`和`nonlocal`关键字
- **类型转换**：`int()`、`str()`等

```python
# 变量定义
name = "Alice"         # 字符串(str)
age = 20               # 整数(int) 
grades = [90, 85, 88]  # 列表(list)
info = {"name": "Alice", "age": 20}  # 字典(dict)

# 类型转换
age_str = str(age)      # 将整数转换为字符串
number = int("123")     # 将字符串转换为整数

# 作用域示例
x = 10  # 全局变量

def my_function():
    y = 5  # 局部变量
    global x  # 声明使用全局变量x
    x += 1    # 修改全局变量
    print(f"函数内: x={x}, y={y}")

my_function()
print(f"函数外: x={x}")  # x的值已在函数中被修改
```

```markdown
# Python与Git基础课程笔记（完整版）

## 3. 运算符及表达式

### 运算符分类
- **算术**：`+`, `-`, `*`, `/`, `//`, `%`, `**`
- **比较**：`==`, `!=`, `>`, `<`, `>=`, `<=`
- **逻辑**：`and`, `or`, `not`
- **位运算**：`&`, `|`, `^`, `<<`, `>>`

```python
a, b = 10, 3

# 算术运算
print(a + b)   # 13
print(a // b)  # 3 (整除)
print(a ** b)  # 1000 (幂运算)

# 逻辑运算
x, y = True, False
print(x and y)  # False
print(x or y)   # True

# 比较运算
print(a > b)    # True
```

## 4. 语句：条件、循环、异常

### 控制结构
- **条件语句**：`if`/`elif`/`else`
- **循环语句**：`for`/`while` + `break`/`continue`
- **异常处理**：`try`/`except`/`finally`

```python
# 条件语句
score = 85
if score >= 90:
    print("A")
elif score >= 60:
    print("Pass")
else:
    print("Fail")

# 循环语句
for i in range(5):
    if i == 3:
        continue  # 跳过3
    print(i)

# 异常处理
try:
    num = int(input("输入数字: "))
    print(100 / num)
except ZeroDivisionError:
    print("不能除以0!")
except ValueError:
    print("无效输入!")
finally:
    print("执行完成")
```

## 5. 函数基础

### 核心功能
- **参数类型**：位置参数、默认参数、可变参数(`*args`, `**kwargs`)
- **匿名函数**：`lambda`表达式
- **高阶函数**：函数作为参数/返回值

```python
# 基础函数
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"
print(greet("Alice"))          # Hello, Alice!
print(greet("Bob", "Hi"))      # Hi, Bob!

# 可变参数
def sum_numbers(*args):
    return sum(args)
print(sum_numbers(1,2,3,4))    # 10

# lambda表达式
double = lambda x: x * 2
print(double(5))              # 10

# 高阶函数
def apply_func(func, value):
    return func(value)
print(apply_func(lambda x: x**2, 4))  # 16
```

## 6. 包和模块

### 模块管理
- **导入方式**：`import` / `from...import`
- **包结构**：目录 + `__init__.py`
- **第三方库**：`pip`安装使用

```python
# mymodule.py 文件内容
""""
def say_hello():
    return "Hello from module!"
"""

# 主程序中调用

# 第三方库使用
import requests
response = requests.get("https://api.github.com")
print(response.status_code)  # 200


```

## 7. 类和对象

### 面向对象特性
- **三大特性**：继承、封装、多态
- **特殊方法**：`__init__`构造函数
- **继承机制**：`super()`调用父类

```python
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
  
    def introduce(self):
        return f"我是{self.name}, 今年{self.age}岁"

# 继承示例
class GradStudent(Student):
    def __init__(self, name, age, major):
        super().__init__(name, age)
        self.major = major
  
    def introduce(self):  # 方法重写
        return f"我是{self.name}, {self.major}专业研究生"

# 使用
s1 = Student("Alice", 20)
s2 = GradStudent("Bob", 25, "CS")
print(s1.introduce())  # 我是Alice, 今年20岁
print(s2.introduce())  # 我是Bob, CS专业研究生
```

## 8. 装饰器

### 核心概念
- **本质**：高阶函数（接收函数→返回新函数）
- **语法**：`@decorator_name`
- **参数装饰器**：嵌套三层函数

```python
# 基础装饰器
def my_decorator(func):
    def wrapper():
        print("执行前")
        func()
        print("执行后")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()

# 带参数装饰器
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"你好, {name}!")

greet("Alice")
```

## 9. 文件操作

### 核心操作
- **读写模式**：`r`(读)/`w`(写)/`a`(追加)
- **上下文管理**：`with...as`自动关闭
- **文件类型**：文本/csv/json

```python
# 文本文件写入
with open("demo.txt", "w") as f:
    f.write("Python文件操作\n")

# 文本文件读取
with open("demo.txt", "r") as f:
    print(f.read())

# CSV文件操作
import csv
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["姓名", "年龄"])
    writer.writerow(["Alice", 20])
```

## 10. Git命令操作

### 常用命令

```bash
# 仓库初始化
git init

# 添加文件到暂存区
git add .

# 提交变更
git commit -m "提交说明"

# 添加远程仓库
git remote add origin 仓库URL

# 拉取远程更新
git pull --rebase origin main

# 推送到远程
git push origin main

# 配置用户信息
git config --global user.name "用户名"
git config --global user.email "邮箱"
```

> **注意**：Windows系统请使用Git Bash执行命令
```
