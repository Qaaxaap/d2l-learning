"""
import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = (x * 2).sum()
y.backward()
print(x.grad) 未请求计算图，需要requires_grad=True

w = torch.tensor([1.0, 2.0])
z = w * 3
z.backward() 反向传播的起点应为标量，括号里写 torch.ones_like(z)或者给z.sum()反向传播。

p = torch.tensor([1.0], requires_grad=True)
y = (p * 2).sum()
y.backward()
p = p - 0.1 * p.grad 没有清零，并且产出了新的计算图，用no_grad,并且在每次backward之前清零grad。

"""

import torch

def grad_of_quadratic(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """返回 y = xᵀAx 对 x 的梯度。

    A 形状 (n, n)，x 形状 (n,)，y 是标量。
    用解析公式算，不许用 autograd（不许出现 requires_grad / backward / autograd.grad）。
    """
    return (A + A.T) @ x

def grad_report(x: torch.Tensor) -> dict:
    """对 y = (x * 2).sum() 反传，返回一份报告，键固定为：

    x_grad     -> x.grad
    y_grad     -> y.grad
    x_is_leaf  -> bool
    y_is_leaf  -> bool
    """
    y = (x * 2).sum()
    y.backward()
    d = {}
    d["x_grad"] = x.grad
    d["y_grad"] = y.grad
    d["x_is_leaf"] = x.is_leaf
    d["y_is_leaf"] = y.is_leaf 
    return d

def grad_after_two_backwards(x: torch.Tensor) -> torch.Tensor:
    """对 y = (x * x).sum() 连续反传两次，返回最终的 x.grad。

    x 形状 (4,)，传入时已开 requires_grad。
    不许在两次之间清零梯度。
    """
    y = (x * x).sum()
    y.backward(retain_graph=True)
    y.backward()
    return x.grad

def descend_step(w: torch.Tensor, step: float) -> None:
    """对一个标量目标 y = (w * w).sum() 做一次梯度下降：w ← w - step * dy/dw。

    - w 形状 (n,)，是叶子张量，已经开了 requires_grad
    - 就地修改 w：不要返回新张量，也不要把名字 w 重新绑定到别的张量上
    - 更新完之后 w 必须仍然是叶子，且 requires_grad 仍为 True
    - 更新过程中不许产生新的计算图，也不许残留上一轮的梯度
    """
    w.grad = None
    y = (w * w).sum()
    y.backward()
    with torch.no_grad():
        w -= step * w.grad
