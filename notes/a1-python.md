# A1 Python 最小子集

这一单元不教你写 Python 程序。它只列 d2l 代码里会反复出现的语法。
每一条都配一段能跑的代码，看完就自己敲一遍跑一遍。

目标：你翻到书上任何一段 MXNet 代码时，能看懂**代码结构**在干什么，
剩下不认识的只有 MXNet 这个库本身。

---

## 1. 变量没有类型声明

```python
x = 3
x = "现在 x 是字符串了"     # 合法
x = [1, 2, 3]              # 也合法
print(x)
```

变量名只是标签，贴到哪里就是什么。后果：拼错的变量名不会报"未声明"，
而是在运行时抛 `NameError`。所以报错先看拼写。

`None` 表示"没有值"，相当于别的语言的 null：

```python
y = None
if y is None:
    print("y 还没有值")
```

判断 `None` 用 `is None`，不要用 `== None`。

---

## 2. 导入

```python
import torch                      # 最常用
import torch.nn as nn             # 起个别名，后面写 nn.Linear 而不是 torch.nn.Linear
from torch.utils.data import DataLoader
import numpy as np                # 社区惯例，numpy 永远叫 np
import matplotlib.pyplot as plt
```

第二版的 d2l 代码里有一行：

```python
from d2l import torch as d2l
```

意思是"从 `d2l` 这个包里，导入它的 `torch` 子模块，并把它叫做 `d2l`"。
所以后面 `d2l.plt`、`d2l.load_data_fashion_mnist(...)` 都是在用这个包。

你自己写的工具函数可以放 `work/d2llocal.py`，然后 `import d2llocal as dl`。

---

## 3. 函数

```python
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

带默认值、可变参数：

```python
def train(net, lr=0.03, num_epochs=3, **kwargs):
    print("lr =", lr, "epochs =", num_epochs, "其余参数:", kwargs)

train(net=None)                       # lr=0.03 epochs=3 其余参数: {}
train(None, 0.1, device="cuda")       # lr=0.1 epochs=3 其余参数: {'device': 'cuda'}
```

`**kwargs` 把多余的关键字参数收成一个字典。d2l 里大量用这个把参数往下传。

返回多个值其实是返回元组：

```python
def two():
    return 1, 2

a, b = two()        # 解包
print(a, b)
```

### 坑：可变默认参数

```python
def bad(x, acc=[]):        # 错！这个 [] 只在定义时创建一次
    acc.append(x)
    return acc

print(bad(1))   # [1]
print(bad(2))   # [1, 2]  ← 不是 [2]
```

要写 `def good(x, acc=None): if acc is None: acc = []`。

---

## 4. 列表、元组、字典

```python
a = [1, 2, 3]           # 列表，可以改
b = (1, 2, 3)           # 元组，不能改
c = {"lr": 0.03, "epochs": 3}   # 字典，键值对

a[0]          # 1，下标从 0 开始
a[-1]         # 3，负数是倒数
a[0:2]        # [1, 2]，切片，左闭右开
c["lr"]       # 0.03
c.get("x", 0) # 取不到时给默认值，不报错
```

列表推导式（一行生成列表）：

```python
squares = [i * i for i in range(5)]              # [0, 1, 4, 9, 16]
evens   = [i for i in range(10) if i % 2 == 0]   # [0, 2, 4, 6, 8]
names   = [f"layer{i}" for i in range(3)]        # ['layer0', 'layer1', 'layer2']
```

字典也支持推导式：`{k: v * 2 for k, v in c.items()}`。

**注意**：上面 `a[0:2]` 返回的是**新列表**（副本），改它不影响 `a`。
但张量的切片不是这样，见 B02。

---

## 5. 循环

```python
for i in range(3):          # 0 1 2
    print(i)

for i, x in enumerate(["a", "b"]):    # 同时要下标和值
    print(i, x)

for X, y in zip([1, 2], [3, 4]):      # 两个序列并排走
    print(X, y)
```

d2l 里最常见的形态：

```python
for X, y in data_iter:          # data_iter 每次吐一批 (特征, 标签)
    ...
```

`while` 循环：

```python
n = 0
while n < 3:
    n += 1          # Python 没有 n++ 这种写法
```

---

## 6. 类与继承

这是读 d2l 代码的门槛。PyTorch 里所有模型、层、数据集都是类。

```python
class Model:
    def __init__(self, w, b):     # 构造函数，创建对象时自动调用
        self.w = w                # self 是"这个对象自己"，相当于别的语言的 this
        self.b = b

    def forward(self, x):         # 普通方法，第一个参数永远是 self
        return self.w * x + self.b


m = Model(2.0, 1.0)      # 调用 __init__，不需要写 self
print(m.forward(torch.tensor([3.0])))
```

继承与 `super()`：

```python
import torch.nn as nn

class MyLayer(nn.Module):            # 继承 nn.Module
    def __init__(self, in_dim, out_dim):
        super().__init__()           # 必须先调父类构造函数，否则 nn.Module 的内部状态没建好
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, X):
        return X @ self.weight + self.bias
```

要点：

- `super().__init__()` 漏掉，后面 `self.parameters()`、`.to(device)` 全会报奇怪的错。
- PyTorch 约定：**计算写在 `forward` 里，调用时用 `net(X)` 而不是 `net.forward(X)`**。
  `net(X)` 会额外触发 hook，这是 `nn.Module` 的 `__call__` 做的事。
- `nn.Parameter` 包起来的张量才会被 `net.parameters()` 收集，才会被优化器更新。
  普通 `self.w = torch.randn(...)` 不会。

---

## 7. 生成器（`yield`）

d2l 的数据迭代器用它。看到 `yield` 就知道这个函数是"一次产出一批"。

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]      # 注意是 yield 不是 return


for X, y in data_iter(2, features, labels):
    print(X.shape)
    break
```

`yield` 和 `return` 的区别：`return` 一返回函数就结束；
`yield` 交出一个值后**暂停在那里**，下次迭代从暂停处继续。

所以 `data_iter(...)` 本身不执行任何代码，只有开始 `for` 循环才跑。

---

## 8. `with` 语句

```python
with torch.no_grad():
    y = net(X)          # 这个块里不建计算图，省显存
```

`with` 管理"进入"和"退出"两个动作。`torch.no_grad()` 进入时关掉梯度记录，
退出时自动恢复原状，哪怕块里抛异常也恢复。

---

## 9. 异常

```python
try:
    x = int("abc")
except ValueError as e:
    print("转换失败：", e)
finally:
    print("不管成不成功都执行这句")
```

d2l 里常见的是捕获文件不存在、下载失败之类。

---

## 10. 装饰器

看到 `@` 开头的行就是装饰器，作用是"把这个函数包一层"。

```python
class Foo:
    @property
    def shape(self):            # 用的时候写 foo.shape，不写 foo.shape()
        return (2, 3)

    @staticmethod
    def helper(x):              # 不需要 self，当普通函数用
        return x + 1
```

```python
@torch.no_grad()
def evaluate(net, data_iter):
    ...                          # 整个函数都在 no_grad 下运行
```

你不需要会写装饰器，但要认识这几个：

| 装饰器 | 作用 |
|---|---|
| `@property` | 方法当属性读 |
| `@staticmethod` | 不用 `self` 的方法 |
| `@torch.no_grad()` | 整个函数不记梯度 |
| `@d2l.add_to_class(X)` | d2l 专用的，给已有类动态加方法 |

---

## 11. 类型注解

现代 Python 代码里常见，**只是标注，不影响运行**：

```python
def train(net, X: torch.Tensor, y: torch.Tensor, lr: float = 0.03) -> float:
    ...
```

`x: torch.Tensor` 意思是"预期 x 是张量"，写错了不会有任何报错。
读代码时把注解当注释看即可。

---

## 12. 字符串

```python
name = "fashion"
epoch = 3
print(f"第 {epoch} 轮，数据集 {name}")      # f-string，最常用
print("第 {} 轮".format(epoch))             # 老写法
print(f"loss = {loss:.3f}")                 # 保留三位小数
```

---

## 13. 会咬人的几个坑

**整数除法**：`3 / 2` 得 `1.5`，`3 // 2` 得 `1`。

**`is` 与 `==`**：`==` 比值，`is` 比是不是同一个对象。判断 `None` 用 `is`。

**作用域**：函数内赋值就是局部变量，想改外面的要 `global` / `nonlocal`（少用）。

**浅拷贝**：
```python
a = [1, 2, 3]
b = a            # b 和 a 是同一个列表
b.append(4)
print(a)         # [1, 2, 3, 4]  ← a 也变了
c = a.copy()     # 这才是副本
```
张量同理，`x2 = x` 不复制数据，`x.clone()` 才复制。

**链式赋值不会创建新对象**：`a = b = []` 之后 `a` 和 `b` 是同一个列表。

---

## 自测

1. `def f(x, acc=[])` 和 `def f(x, acc=None)` 有什么区别？为什么前者是 bug？
2. 为什么 `nn.Module` 子类里 `super().__init__()` 不能省？
3. `net(X)` 和 `net.forward(X)` 有什么区别？
4. `self.w = torch.randn(3, 4)` 和 `self.w = nn.Parameter(torch.randn(3, 4))` 有什么区别？
5. 下面这段哪里错了：
   ```python
   def data_iter(X, y, batch_size):
       for i in range(0, len(X), batch_size):
           return X[i:i+batch_size], y[i:i+batch_size]
   ```
6. `yield` 是什么？`data_iter(...)` 调用后立刻发生了什么？
7. `with torch.no_grad():` 块里如果抛异常，梯度记录状态会怎样？

<details>
<summary>做完再看：答案</summary>

1. 默认值 `[]` 只在函数**定义时**创建一次，所有调用共享同一个列表，会越积越多。
   用 `None` 当默认值，函数内部再新建。
2. `nn.Module` 的 `__init__` 负责初始化内部的状态（参数注册表、子模块表、hook 表）。
   不调它，后面 `self.w = nn.Parameter(...)` 无处登记，`parameters()` 返回空，
   优化器没有可更新的参数。
3. `net(X)` 走 `nn.Module.__call__`，会依次触发 forward pre-hook、`forward`、forward hook。
   直接 `net.forward(X)` 绕过 hook，有些框架功能（如 `torch.compile`、梯度检查点）会失效。
4. 普通属性不会被 `net.parameters()` 收集，优化器看不到它，训练时永远不更新。
5. 用了 `return` 而不是 `yield`，只返回第一批就结束。要改成 `yield`，
   并且切片索引应该用张量（`X` 是张量时 `X[i:i+batch_size]` 可以，但更常见的写法是先打乱下标）。
6. `yield` 把函数变成生成器。调用 `data_iter(...)` 时函数体一行都不执行，
   只返回一个生成器对象；第一次 `for` 迭代才开始执行到第一个 `yield`。
7. 恢复。`with` 的退出逻辑在异常时同样执行。

</details>
