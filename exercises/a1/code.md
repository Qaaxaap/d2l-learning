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

`Vec2` 表示二维向量，两个属性 `x`、`y`。

| 表达式 | 做什么 | 返回 |
|---|---|---|
| `Vec2(1, 2)` | 构造 | 对象，`v.x == 1`、`v.y == 2` |
| `Vec2(1, 2) + Vec2(3, 4)` | 逐分量相加，**不修改两个操作数** | 新的 `Vec2`，`x == 4`、`y == 6` |
| `Vec2(1, 2) == Vec2(1, 2)` | 逐分量比较 | `True` |
| `Vec2(1, 2) == Vec2(1, 3)` | 同上 | `False` |
| `Vec2(1, 2) == "abc"` | 与别的类型比较 | `False`，**不能抛异常** |
| `repr(Vec2(1, 2))` | 字符串表示 | 形如 `Vec2(1, 2)` |

要实现的 dunder：`__init__`、`__add__`、`__eq__`、`__repr__`。

`__eq__` 遇到类型不对的东西，讲义第 4 节末尾讲了该返回什么（不是 `False`，也不是抛异常，
是一个特殊的常量，交给 Python 去处理）。这是本题的一个考点。

## T3 生成器

```python
def batch_indices(n: int, batch_size: int):
    """把 0 到 n-1 这 n 个索引按 batch_size 分批，逐批产出。"""
```

| 调用 | 产出 |
|---|---|
| `list(batch_indices(5, 2))` | `[[0, 1], [2, 3], [4]]` |
| `list(batch_indices(4, 2))` | `[[0, 1], [2, 3]]`，整除时不多出空批 |
| `list(batch_indices(0, 2))` | `[]` |
| `batch_indices(5, 2)` | 必须是**生成器对象**，不是 list |

每一批是 Python 的 `list`（不是 `range`，也不是张量）。

必须用 `yield` 实现。验收会检查它返回的是生成器对象。

## T4 继承 nn.Module

写一个 `nn.Module` 子类，练讲义第 3 节的三条规矩。它算什么不重要，按规格实现即可——
矩阵乘法、线性层那些是 B03 与 B06 的内容，现在不用管。

### 你要定义的

| 项 | 要求 |
|---|---|
| 类名 | `ScaledShift`，继承 `nn.Module` |
| 构造签名 | `def __init__(self, size)` |
| 属性 `self.scale` | 一维、长度 `size` 的 `nn.Parameter`，初值全 1 |
| 属性 `self.bias` | 一维、长度 `size` 的 `nn.Parameter`，初值全 0 |
| 方法 `forward(self, X)` | 返回 `X * self.scale + self.bias`。`*` 是逐元素乘法，不是矩阵乘法 |

### 验收会检查的

这些是从外面看这个对象时应该成立的事，不是你额外要写的代码。

| 检查 | 期望 |
|---|---|
| `ScaledShift(3)` 是 `nn.Module` 的实例 | `True` |
| `net.scale`、`net.bias` 的类型 | 都是 `nn.Parameter`，不是普通张量 |
| `len(list(net.parameters()))` | `2` |
| 两个参数的形状 | 都是一维、长度 `size` |
| `net(X)` 的输出形状 | 与输入 `X` 相同 |
| `net(X).sum().backward()` 之后 | 两个参数的 `.grad` 都不为 `None` |

讲义第 3 节的三条规矩都要做对：调 `super().__init__()`、参数用 `nn.Parameter` 包起来、
调用时写 `net(X)` 而不是 `net.forward(X)`。

这个模块的意义是让你亲手把参数交给框架管理。真正会用上它的场景在 B06。

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

| 项 | 要求 |
|---|---|
| `__enter__` | 记下起始时刻。返回值就是 `with ... as x` 里的 `x`，按惯例返回 `self` |
| `__exit__` | 记下结束时刻，算出 `self.elapsed` |
| `elapsed` | 单位秒的 float，退出 `with` 之后可读 |
| 块内抛异常 | `elapsed` 仍然要被赋值，异常照常往外抛 |

计时起点放在 `__enter__` 里，不是 `__init__` 里。`Timer()` 创建到进入 `with` 之间可能有间隔。

## T6 改错

下面这段代码有四类问题。把代码抄进 `work/a1/a1.py` 顶部的注释里，
每处标出三样：**位置**（第几行）、**为什么错**、**怎么改**。

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

四处分别关于父类初始化、参数登记、`return` 与 `yield`、可变默认参数。先把四处找齐再看。

这一题不跑代码，只写分析。
