"""B04 参考答案。做完再看。

跑一遍：
    python3 solutions/b04/b04.py
或者拿它验断言：
    python3 tools/check_b04.py solutions/b04/b04.py
"""

import torch


def grad_of_quadratic(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """y = xᵀAx 的梯度是 (A + Aᵀ)x。A 对称时退化成 2Ax。"""
    return (A + A.T) @ x


def grad_report(x: torch.Tensor) -> dict:
    """反传一次，把叶子与非叶子的状态如实报出来。"""
    y = (x * 2).sum()
    y.backward()
    return {
        "x_grad": x.grad,
        "y_grad": y.grad,  # 非叶子，默认不保存，是 None
        "x_is_leaf": x.is_leaf,
        "y_is_leaf": y.is_leaf,
    }


def grad_after_two_backwards(x: torch.Tensor) -> torch.Tensor:
    """两次反传，梯度累加。第一次要用 retain_graph 保住图。"""
    y = (x * x).sum()
    y.backward(retain_graph=True)
    y.backward()
    return x.grad


def train_step(w: torch.Tensor, x: torch.Tensor, lr: float) -> None:
    """一次梯度下降。清梯度、反传、在 no_grad 里原地更新。"""
    if w.grad is not None:
        w.grad = None
    loss = (x @ w).sum()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad


"""T5 改错

第 1 处：`x = torch.tensor([1.0, 2.0, 3.0])`，`y.backward()` 会报
    RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    求导的前提是至少有一个叶子张量开了 requires_grad，从它出发才建得起图。
    x 就是那个叶子，要在创建时写 requires_grad=True。

第 2 处：`z = w * 3` 有三个元素，不是标量，`z.backward()` 会报
    RuntimeError: grad can be implicitly created only for scalar outputs
    backward() 默认从标量出发反传。非标量输出要么先归约成标量（z.sum().backward()），
    要么显式传 gradient 参数，说明每个输出分量的权重。

第 3 处：`p = p - 0.1 * p.grad` 把名字 p 重新绑定到了一个新的张量上。
    那个新张量不是叶子、也不带 requires_grad，下一轮反传时 p.grad 会是 None
    （或者直接报 does not require grad）。要让 p 本身被更新，必须原地改：
        with torch.no_grad():
            p -= 0.1 * p.grad
    这样 p 还是原来那个叶子，梯度也留在同一个对象上。
    不用 no_grad 而直接写 p -= ... 会报
    RuntimeError: a leaf Variable that requires grad is being used in an in-place operation
"""


if __name__ == "__main__":
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x = torch.tensor([1.0, -1.0])
    print("(A + A.T) @ x =", grad_of_quadratic(A, x))

    xg = torch.tensor([1.0, 2.0], requires_grad=True)
    print("report =", grad_report(xg))

    xr = torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    print("two backwards =", grad_after_two_backwards(xr))

    w = torch.tensor([0.0, 0.0], requires_grad=True)
    data = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    train_step(w, data, 0.5)
    train_step(w, data, 0.5)
    print("w after two steps =", w, "is_leaf =", w.is_leaf)
