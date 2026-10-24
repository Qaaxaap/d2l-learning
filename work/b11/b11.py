import torch
import torch.nn as nn

def build_model(x: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor,
                W2: torch.Tensor, b2: torch.Tensor, y: torch.Tensor,
                lam: float = 0.0):
    """按下面的前向算出目标函数 J，返回一个标量张量。

        z = W1 @ x + b1
        h = relu(z)
        o = W2 @ h + b2
        L = 0.5 * sum((o - y) ** 2)
        s = lam / 2 * (||W1||_F^2 + ||W2||_F^2)
        J = L + s

    形状：x (d, 1)、y (q, 1)、W1 (h, d)、b1 (h, 1)、W2 (q, h)、b2 (q, 1)。
    四个参数由调用方传进来，**函数里不要重新初始化它们**。
    """
    z = W1 @ x + b1
    h = torch.where(z > 0, z, torch.zeros_like(z))
    o = W2 @ h + b2
    L = 0.5 * ((o - y) ** 2).sum()
    s = lam / 2 * ((W1 ** 2).sum() + (W2 ** 2).sum())
    J = L + s
    return J.sum()
    


def grads_by_autograd(x: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor,
                      W2: torch.Tensor, b2: torch.Tensor, y: torch.Tensor,
                      lam: float = 0.0) -> dict:
    """对四个参数求梯度，返回 {"W1": ..., "b1": ..., "W2": ..., "b2": ...}。

    返回的梯度要**脱离计算图**（调用方会拿它们做数值比较），形状与参数相同。
    不要修改传进来的参数。
    """
    l = build_model(x.requires_grad(), W1, b1, W2, b2, y, lam)
    l.backward()
    return {
        "W1": W1.grad.detach(),
        "b1": b1.grad.detach(),
        "W2": W2.grad.detach(),
        "b2": b2.grad.detach(),
    }