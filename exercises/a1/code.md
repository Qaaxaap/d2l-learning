# A1 代码题

把实现写在 `work/a1/a1.py`。函数名、类名、签名必须和下面一致。

验收：`just check a1`。断言全过之后我还会读代码，看你有没有真的用上该用的语法。

这几题不为难你，目的是把 Python 的类和 dunder 亲手写一遍。读懂和写出来是两回事。

---

## T1 可调用的计数器

写一个类 `Counter`，它同时维护两个状态：**当前值**和**被调用次数**。

```python
c = Counter()            # 当前值从 0 开始
c()                      # 返回 1
c(5)                     # 返回 6
len(c)                   # 返回 2
repr(c)                  # 形如 "Counter(value=6, calls=2)"
Counter(start=10)()      # 返回 11
```

逐条讲清楚，别猜：

| 调用 | 做什么 | 返回 |
|---|---|---|
| `Counter(start=0)` | 当前值设为 `start`，调用次数设为 0 | 对象本身 |
| `c(step=1)` | 当前值**加上** `step`（不是设为 step），调用次数加 1 | 加完之后的当前值 |
| `len(c)` | 不改任何状态 | 调用次数 |
| `repr(c)` | 不改任何状态 | 形如 `Counter(value=<当前值>, calls=<调用次数>)` 的字符串，`value=` 和 `calls=` 后面跟正确的数字即可 |

要实现的四个 dunder：`__init__`、`__call__`、`__len__`、`__repr__`。

`step` 要有默认值，这样 `c()` 和 `c(1)` 等价。

## T2 一个支持运算符的类

```python
class Vec2:
    """二维向量，需要支持：
        Vec2(1, 2) + Vec2(3, 4)   -> Vec2(4, 6)
        Vec2(1, 2) == Vec2(1, 2)  -> True
        print(Vec2(1, 2))         -> 类似 "Vec2(1, 2)"
        v.x, v.y                  -> 属性可读
    """
```

这一题考的就是讲义第 4 节。实现 `__init__`、`__add__`、`__eq__`、`__repr__`。

`__eq__` 要能和不同类型的对象比较而不炸（比如 `Vec2(1, 2) == "abc"` 应该返回 `False`，
不是抛异常）。

## T3 生成器

```python
def batch_indices(n: int, batch_size: int):
    """按 batch_size 切分 range(n)，逐批产出索引列表。

    n=5, batch_size=2 时依次产出 [0, 1]、[2, 3]、[4]。
    必须用 yield 实现，不许返回一个列表。
    """
```

## T4 继承 nn.Module

```python
class ScaledLinear(nn.Module):
    """带缩放系数的线性层。
    构造：ScaledLinear(in_features, out_features, scale=1.0)
    权重初始化为 torch.randn(in_features, out_features) * scale
    偏置初始化为全零
    forward(X) 返回 X @ weight + bias
    """
```

这一题要把讲义第 3 节的三条都做对：调 `super().__init__()`、参数用 `nn.Parameter` 包起来。
验收断言会检查 `parameters()` 里恰好有两个张量。

## T5 上下文管理器

```python
class Timer:
    """用法：
        t = Timer()
        with t:
            time.sleep(0.05)
        print(t.elapsed)     # 大约 0.05
    """
```

需要实现 `__enter__` 和 `__exit__`。

`__enter__` 的返回值就是 `with ... as x` 里的那个 `x`。测试用的是 `t = Timer(); with t:`
这种写法，所以它返回什么不影响判分，但按惯例应当返回 `self`。

`elapsed` 在退出之后可读，单位秒。

## T6 改错

下面这段代码有四类问题。把代码抄进 `a1.py` 顶部的注释里，每处标出位置、说明为什么错、怎么改。

```python
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        self.weight = torch.randn(in_dim, out_dim)
        self.bias = torch.zeros(out_dim)

    def forward(self, X):
        return X @ self.weight + self.bias


def batches(data=[]):
    for i in range(len(data)):
        return data[i]


net = Net(3, 1)
print(len(list(net.parameters())))
print(net(torch.randn(2, 3)))
```

提示：四处分别关于父类初始化、参数登记、`return` 与 `yield`、以及可变默认参数。
先把四处找齐再看答案。
