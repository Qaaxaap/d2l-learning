# B11 代码题

把你的实现写在 `work/b11/b11.py` 里，函数名与签名必须和下面一致。

验收：`just check b11`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章要做什么

B04 里用 `backward()` 一行拿到梯度。这一章先**手写**一遍这个过程——
不调用 autograd，只用链式法则与矩阵运算，把四个参数的梯度推出来；
再写一个**数值梯度**作为独立的验证手段；最后写两种初始化。

手写反向传播是这一章的核心。它没有"标准答案"可抄，
只有"形状对不对、数值对不对"两条判据——而后者正好可以用数值梯度来验。

验收分三类：**精确**（Xavier 的区间、数值梯度与解析梯度的吻合）、
**范围**（初始化后的方差量级）、**趋势**（层数增加时连乘结果的走向）。

---

## T1 手写反向传播

```python
def forward_backward(x: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor,
                     W2: torch.Tensor, b2: torch.Tensor, y: torch.Tensor,
                     lam: float = 0.0) -> tuple:
    """一个单隐藏层的 MLP，手写正向与反向。

    前向：
        z = W1 @ x + b1
        h = relu(z)                # relu 自己写，不许用 torch.relu
        o = W2 @ h + b2
        L = 0.5 * sum((o - y) ** 2)
        s = lam / 2 * (||W1||_F^2 + ||W2||_F^2)
        J = L + s

    形状：x (d, 1)、y (q, 1)、W1 (h, d)、b1 (h, 1)、W2 (q, h)、b2 (q, 1)。

    返回 (J, grads)，J 是标量张量，grads 是字典，键为 "W1"、"b1"、"W2"、"b2"，
    值是各个梯度，形状与对应参数相同。

    **不许调用 backward()、不许用任何 autograd 相关的东西**（包括 torch.autograd、
    requires_grad）。全部梯度用链式法则手推。
    """
```

| 检查 | 期望 |
|---|---|
| 返回值 | `(J, grads)`，`grads` 有四个键，形状与参数一致 |
| `lam=0` 时的 `J` | 等于 `0.5 * sum((o - y) ** 2)` |
| 四个梯度的**数值** | 与数值梯度吻合（相对误差 1e-4 以内）|
| `lam>0` 时 | `W1`、`W2` 的梯度比 `lam=0` 时多 `lam * W` |
| 代码里 | 没有 `backward`、没有 `requires_grad` |

**提示**：写完后用 T2 与 T3 两个工具自查。手推容易在转置上出错，
而形状对不上时数值梯度会直接告诉你第几个元素错了。

## T2 数值梯度

```python
def numeric_grad(f, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """中心差分数值梯度。

    对 x 的每个元素，把它加上 eps 与减去 eps 各算一次 f，差商 (f(+) - f(-)) / (2*eps)
    就是该元素上的近似偏导。返回与 x 同形状的张量。

    f 接受一个张量返回一个标量张量。不会修改传入的 x。
    """
```

| 检查 | 期望 |
|---|---|
| `f = lambda t: (t ** 2).sum()`，`x = [1, 2, 3]` | 梯度约 `[2, 4, 6]` |
| `f = lambda t: (t.sum())`，任意 `x` | 全 1 |
| `x` 本身 | 没被改动 |
| 返回形状 | 与 `x` 相同 |

## T3 梯度检查

```python
def grad_check(analytic: torch.Tensor, numeric: torch.Tensor,
               tol: float = 1e-4) -> bool:
    """比较两组梯度是否吻合。

    相对误差定义为：
        ||analytic - numeric|| / max(||analytic|| + ||numeric||, 1e-8)
    小于 tol 返回 True，否则 False。
    """
```

| 检查 | 期望 |
|---|---|
| 两组完全相同的梯度 | `True` |
| 差一个很小的量（1e-6 量级）| `True` |
| 相差 1% | `False` |
| 两组都是全零 | `True` |

## T4 两种初始化

```python
def xavier_uniform(a: int, b: int, seed: int = 0) -> torch.Tensor:
    """返回形状 (a, b) 的 Xavier 均匀初始化权重。

    每个元素从 U(-r, r) 采样，r = sqrt(6 / (a + b))，a 是输入个数、b 是输出个数。
    同一个 seed 给出相同结果。
    """


def he_uniform(a: int, b: int, seed: int = 0) -> torch.Tensor:
    """返回形状 (a, b) 的 He 均匀初始化权重。

    每个元素从 U(-r, r) 采样，r = sqrt(6 / a)。
    """
```

| 检查 | 期望 |
|---|---|
| `xavier_uniform(100, 50)` 的取值范围 | 在 ±0.2 以内（$r=\sqrt{6/150}=0.2$）|
| `he_uniform(100, 50)` 的取值范围 | 在 ±0.245 以内（$r=\sqrt{6/100}\approx0.245$）|
| 同一 seed | 结果相同 |
| 大样本上的方差 | Xavier 约为 $r^2/3$ |

## T5 改错

下面这段手写反向传播有三处问题。把代码抄进 `work/b11/b11.py` 顶部的注释里，
每处标出**位置**、**为什么错**、**怎么改**。

```python
import torch

def relu(t):
    return torch.maximum(t, torch.zeros_like(t))

def backward(x, W1, b1, W2, b2, y, lam=0.0):
    z = W1 @ x + b1
    h = relu(z)
    o = W2 @ h + b2
    L = 0.5 * ((o - y) ** 2).sum()
    s = lam / 2 * ((W1 ** 2).sum() + (W2 ** 2).sum())
    J = L + s

    d_o = o - y                          # 对 o 的梯度
    d_W2 = d_o @ h.t() + lam * W2        # 对 W2 的梯度
    d_b2 = d_o.sum()                     # 对 b2 的梯度
    d_h = W2 @ d_o                       # 对 h 的梯度
    d_z = d_h * (z >= 0).float()         # 对 z 的梯度
    d_W1 = d_z @ x.t() + lam * W1        # 对 W1 的梯度
    d_b1 = d_z.sum()                     # 对 b1 的梯度
    return J, {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}
```

三处分别指三个位置：**对 `h` 求导的转置**、**偏置梯度的形状**、
**relu 在 0 处的导数**。

三段提示：第一处可能不报错，只有数值梯度会指出问题；
第二处两处偏置各有一份，求和时少了 `keepdim` 会让形状对不上；
第三处只在 z 恰好为 0 的那一点上与要求不符，前两处改完用数值梯度一测就会露出来。
