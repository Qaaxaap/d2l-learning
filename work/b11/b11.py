from numpy import zeros_like
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
    rg = lambda x : x.clone().requires_grad_()
    W1, b1, W2, b2 = rg(W1), rg(b1), rg(W2), rg(b2)
    l = build_model(x, W1, b1, W2, b2, y, lam)
    l.backward()
    return {
        "W1": W1.grad.detach(),
        "b1": b1.grad.detach(),
        "W2": W2.grad.detach(),
        "b2": b2.grad.detach(),
    }

def numeric_grad(f, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """中心差分数值梯度。

    对 x 的每个元素，把它加上 eps 与减去 eps 各算一次 f，差商 (f(+) - f(-)) / (2*eps)
    就是该元素上的近似偏导。返回与 x 同形状的张量。

    f 接受一个张量返回一个标量张量。不会修改传入的 x。
    """
    x = x.to(torch.float64)
    grad = []
    for i, xi in enumerate(x.flatten()):
        epi = torch.zeros_like(x, dtype=torch.float64).flatten()
        epi[i] += eps
        epi = epi.reshape(x.shape)
        grad.append((f(x + epi) - f(x - epi)) / (2 * eps))
    return torch.tensor(grad).reshape(x.shape).to(torch.float32)

def grad_check(analytic: torch.Tensor, numeric: torch.Tensor,
               tol: float = 1e-4) -> bool:
    """比较两组梯度是否吻合。

    相对误差定义为：
        ||analytic - numeric|| / max(||analytic|| + ||numeric||, 1e-8)
    小于 tol 返回 True，否则 False。
    """
    if (analytic - numeric).sum() / max(analytic.sum() + numeric.sum(), 1e-8) < tol:
        return True
    return False

def xavier_uniform(a: int, b: int, seed: int = 0) -> torch.Tensor:
    """返回形状 (a, b) 的 Xavier 均匀初始化权重。

    每个元素从 U(-r, r) 采样，r = sqrt(6 / (a + b))，a 是输入个数、b 是输出个数。
    同一个 seed 给出相同结果。
    """
    r = (6 / (a + b)) ** 0.5
    torch.manual_seed(seed)
    return torch.rand(size=(a, b), dtype=torch.float32) * r

def he_uniform(a: int, b: int, seed: int = 0) -> torch.Tensor:
    """返回形状 (a, b) 的 He 均匀初始化权重。

    每个元素从 U(-r, r) 采样，r = sqrt(6 / a)。
    """
    r = (6 / a) ** 0.5
    torch.manual_seed(seed)
    return torch.rand(size=(a, b), dtype=torch.float32) * r