# B04 代码题

把你的实现写在 `work/b04/b04.py` 里，函数名和签名必须和下面一致。

验收：`just check b04`。断言全过，并且我读过代码之后，这一题才算过。

不许用讲义没讲过的库函数绕过要求。题目说"不许用"的就是不许用。

---

## T1 手算二次型的梯度

```python
def grad_of_quadratic(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """返回 y = xᵀAx 对 x 的梯度。

    A 形状 (n, n)，x 形状 (n,)，y 是标量。
    用解析公式算，不许用 autograd（不许出现 requires_grad / backward / autograd.grad）。
    """
```

| 输入 | 应当返回 |
|---|---|
| `A = I₃`，`x = [1., 1., 1.]` | `[2., 2., 2.]`（形状 `(3,)`） |
| `A = I₃`，`x = [1., 2., 3.]` | `[2., 4., 6.]` |
| 任意非对称 `A` | `(A + A.T) @ x` |

一般形式的推导在讲义第 2.3 节。

## T2 叶子与非叶子

```python
def grad_report(x: torch.Tensor) -> dict:
    """对 y = (x * 2).sum() 反传，返回一份报告，键固定为：

    x_grad     -> x.grad
    y_grad     -> y.grad
    x_is_leaf  -> bool
    y_is_leaf  -> bool
    """
```

`x` 传入时已经开了 `requires_grad`。`y.grad` 会是 `None`，这是正常的，报告里如实返回它。

访问非叶子的 `.grad` 会打印一条很长的 `UserWarning`（大意是"这个张量的 .grad 不会被填充"）。
那是 torch 在提醒你别误用，这一题就要求你访问它，忽略即可。

讲义第 5.1 节讲了为什么。

## T3 连续反传两次

```python
def grad_after_two_backwards(x: torch.Tensor) -> torch.Tensor:
    """对 y = (x * x).sum() 连续反传两次，返回最终的 x.grad。

    x 形状 (4,)，传入时已开 requires_grad。
    不许在两次之间清零梯度。
    """
```

| 输入 | 应当返回 |
|---|---|
| `[0., 1., 2., 3.]` | `[0., 4., 8., 12.]` |

## T4 一次参数更新

```python
def descend_step(w: torch.Tensor, step: float) -> None:
    """对一个标量目标 y = (w * w).sum() 做一次梯度下降：w ← w - step * dy/dw。

    - w 形状 (n,)，是叶子张量，已经开了 requires_grad
    - 就地修改 w：不要返回新张量，也不要把名字 w 重新绑定到别的张量上
    - 更新完之后 w 必须仍然是叶子，且 requires_grad 仍为 True
    - 更新过程中不许产生新的计算图，也不许残留上一轮的梯度
    """
```

| 情形 | 期望 |
|---|---|
| `w = [1., 2.]`，`step = 0.1` | `w` 变成 `[0.8, 1.6]`（`y` 对 `w` 的梯度是 `2w`） |

**不许用 `torch.optim` 里的任何优化器。** 两个原因：优化器是 B06 的内容；
更重要的是这道题练的正是优化器内部替你做的事——清梯度、在 `no_grad` 下原地更新。
调一次 `optimizer.step()`，那三个坑全被封装掉，题目就没意义了。

提示：这个函数会被连续调用两次，第二次开始时不能带着第一次的梯度残留。

## T5 改错

下面这段代码有三处问题。把代码抄进 `work/b04/b04.py` 顶部的注释里，
每处标出三样：**位置**（第几行）、**为什么错**、**怎么改**。

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = (x * 2).sum()
y.backward()
print(x.grad)

w = torch.tensor([1.0, 2.0])
z = w * 3
z.backward()

p = torch.tensor([1.0], requires_grad=True)
y = (p * 2).sum()
y.backward()
p = p - 0.1 * p.grad
```

三处分别关于：**求导的前提**、**非标量输出**、**参数更新的写法**。

它们都会在运行时出问题，只是报错的位置和时机不同。写分析时说明各自的症状。
