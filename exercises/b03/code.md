# B03 代码题

把你的实现写在 `work/b03/b03.py` 里，函数名和签名必须和下面一致。

验收：`just check b03`。断言全过，并且我读过代码之后，这一题才算过。

不许用讲义没讲过的库函数绕过要求。题目说"不许用"的就是不许用。

---

## T1 按轴求和的形状

```python
def reduce_shape(shape: tuple, dim: int, keepdim: bool) -> tuple:
    """算出「对形状 shape 的张量沿 dim 求和」之后的结果形状。

    不调用 torch，纯推导。
    """
```

| 调用 | 应当返回 |
|---|---|
| `reduce_shape((3, 4), 0, False)` | `(4,)` |
| `reduce_shape((3, 4), 0, True)` | `(1, 4)` |
| `reduce_shape((3, 4), 1, False)` | `(3,)` |
| `reduce_shape((2, 3, 4), 1, False)` | `(2, 4)` |
| `reduce_shape((2, 3, 4), 1, True)` | `(2, 1, 4)` |
| `reduce_shape((5,), 0, False)` | `()` |

返回值必须是普通 Python `tuple`，形状为空时返回 `()`（长度 0 的元组）。

`dim` 取负数时按 Python 的下标惯例从右往左数：`dim=-1` 是最后一维。
`reduce_shape((2, 3, 4), -1, False)` 应当返回 `(2, 3)`。

规则见讲义第 4 节。

## T2 该用哪个乘法函数

```python
def matmul_kind(a: torch.Tensor, b: torch.Tensor) -> str:
    """判断这两个张量相乘时应当用哪个函数，返回最专用的那个。"""
```

| a 的形状 | b 的形状 | 应当返回 |
|---|---|---|
| `(3,)` | `(3,)` | `"dot"` |
| `(2, 3)` | `(3,)` | `"mv"` |
| `(2, 3)` | `(3, 4)` | `"mm"` |
| `(2, 3, 4)` | `(4, 5)` | `"matmul"` |
| `(2, 3, 4)` | `(2, 4, 5)` | `"matmul"` |

判定顺序：两个都是一维用 `dot`；一个二维一个一维用 `mv`；两个都是二维用 `mm`；
其余能用 `matmul` 的情形一律返回 `"matmul"`。

只处理能相乘的形状组合，形状对不上的情形不用管。四个函数的区别见讲义第 0 节。

## T3 手写 L2 范数

```python
def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """用基本运算算出 x 的 L2 范数。

    不许用 torch.linalg.vector_norm、torch.linalg.norm、torch.norm、
    torch.hypot 等任何现成的范数函数。
    """
```

| 调用 | 应当返回 |
|---|---|
| `l2_norm(torch.tensor([3.0, 4.0]))` | `tensor(5.)` |
| `l2_norm(torch.zeros(5))` | `tensor(0.)` |
| `l2_norm(torch.ones(2, 3))` | `tensor(2.4495)`（$\sqrt{6}$） |

对任意形状都成立，先展平再算。返回 0 维张量，不是 Python 浮点数。

## T4 按行 L2 归一化

```python
def normalize_rows(X: torch.Tensor) -> torch.Tensor:
    """把 X 的每一行除以该行的 L2 范数，返回新张量。

    要求：
    - 不修改 X
    - 某行的元素全是 0 时，结果里那一行保持全 0，不许出现 nan
    - 返回的 dtype 与 X 相同
    """
```

| 调用 | 应当返回 |
|---|---|
| `normalize_rows(torch.tensor([[3.0, 4.0]]))` | `tensor([[0.6, 0.8]])` |
| `normalize_rows(torch.zeros(2, 3))` | 全 0，且不含 nan |
| `normalize_rows(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))` | `tensor([[1., 0.], [0., 0.]])` |

不许用循环逐行处理，用张量运算一次算完。需要的话可以用 `torch.where`，它的用法见讲义第 4 节末尾。

## T5 改错

下面这段代码有三处问题。把代码抄进 `work/b03/b03.py` 顶部的注释里，
每处标出三样：**位置**（第几行）、**为什么错**、**怎么改**。

```python
import torch

A = torch.arange(6.0).reshape(2, 3)
x = torch.ones(3)

# 想算 A 乘以向量 x
y = A * x

# 想每一行除以该行元素之和
s = A.sum(dim=1)
z = A / s

# 想交换 B 的最后两维
B = torch.arange(24.0).reshape(2, 3, 4)
C = B.T
```

三处分别关于：**运算符的选择**、**求和后的形状**、**高维转置的语义**。

三处的共同点是它们都不报错或者报错位置离出错位置很远。写分析时说明各自的症状。
