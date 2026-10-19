"""B11 参考答案。做完再看。

跑一遍：
    python3 solutions/b11/b11.py
或者拿它验断言：
    python3 tools/check_b11.py solutions/b11/b11.py
"""

import torch


def relu(t: torch.Tensor) -> torch.Tensor:
    return torch.maximum(t, torch.zeros_like(t))


# ---------- T1 手写反向传播 ----------

def forward_backward(x: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor,
                     W2: torch.Tensor, b2: torch.Tensor, y: torch.Tensor,
                     lam: float = 0.0) -> tuple:
    # 前向
    z = W1 @ x + b1
    h = relu(z)
    o = W2 @ h + b2
    loss = 0.5 * ((o - y) ** 2).sum()
    reg = lam / 2 * ((W1**2).sum() + (W2**2).sum())
    J = loss + reg

    # 反向，按讲义 2.2 节的顺序倒着来
    d_o = o - y                          # dL/do，平方损失对 o 求导
    d_W2 = d_o @ h.t() + lam * W2        # dJ/dW2 = d_o h^T + lam*W2
    d_b2 = d_o.sum(dim=1, keepdim=True)  # dJ/db2 = sum(d_o)，按列求和
    d_h = W2.t() @ d_o                   # dJ/dh = W2^T d_o，注意转置
    d_z = d_h * (z > 0).float()          # 过 relu：逐元素乘，导数是 0/1（0 处取 0）
    d_W1 = d_z @ x.t() + lam * W1        # dJ/dW1 = d_z x^T + lam*W1
    d_b1 = d_z.sum(dim=1, keepdim=True)  # dJ/db1 = sum(d_z)
    return J, {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}


# ---------- T2 数值梯度 ----------

def numeric_grad(f, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """中心差分。逐元素加减 eps 各算一次 f。

    x 在函数内被临时改动又恢复；传进来的是副本，所以调用者看不到改动。
    用 flatten() 拿到视图，避免索引多维下标。
    """
    # 内部用 float64：float32 下 (f(+e) - f(-e)) 只有 4e-6 量级，
    # 而这两个值各自的表示误差就有约 1e-7，差商会失真。
    work = x.clone().double()
    flat = work.flatten()
    grad = torch.zeros_like(flat)
    for i in range(flat.numel()):
        orig = flat[i].item()
        flat[i] = orig + eps
        fp = f(work).item()
        flat[i] = orig - eps
        fm = f(work).item()
        flat[i] = orig
        grad[i] = (fp - fm) / (2 * eps)
    return grad.reshape(x.shape).to(x.dtype)


# ---------- T3 梯度检查 ----------

def grad_check(analytic: torch.Tensor, numeric: torch.Tensor,
               tol: float = 1e-4) -> bool:
    diff = (analytic - numeric).norm()
    scale = max((analytic.norm() + numeric.norm()).item(), 1e-8)
    return (diff / scale).item() < tol


# ---------- T4 两种初始化 ----------

def xavier_uniform(a: int, b: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = (6.0 / (a + b)) ** 0.5
    return (torch.rand(a, b, generator=g) * 2 - 1) * r


def he_uniform(a: int, b: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = (6.0 / a) ** 0.5
    return (torch.rand(a, b, generator=g) * 2 - 1) * r


"""T5 改错

第 1 处：`d_h = W2 @ d_o`
    正向是 `o = W2 @ h`，对 `h` 求导要乘 `W2` 的**转置**：
    `d_h = W2.t() @ d_o`。
    这一处有个隐蔽之处：两个形状在方阵时恰好也对得上，所以不报错，
    只有数值梯度会指出数值不对。

第 2 处：`d_b2 = d_o.sum()`
    偏置 `b2` 的形状是 (q, 1)，而 `d_o.sum()` 是标量。少了 `dim=1, keepdim=True`：
    `d_b2 = d_o.sum(dim=1, keepdim=True)`。
    两处偏置都有这个问题（`d_b1` 同）。短向量时可能不报错，长向量时形状对不上。

第 3 处：`d_z = d_h * (z >= 0).float()`
    ReLU 在 z=0 处的导数按约定取 0，用 `>=` 会把它取成 1。
    本课程统一用 `(z > 0)`。
    这一处只在 z 恰好为 0 的元素上不同，随机输入下很少命中，
    用 `torch.zeros` 造输入时才会露出来。
"""


if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)
    d, h, q = 5, 4, 3
    x = torch.randn(d, 1, generator=g)
    W1 = torch.randn(h, d, generator=g)
    b1 = torch.randn(h, 1, generator=g)
    W2 = torch.randn(q, h, generator=g)
    b2 = torch.randn(q, 1, generator=g)
    y = torch.randn(q, 1, generator=g)

    J, grads = forward_backward(x, W1, b1, W2, b2, y, lam=0.1)
    print("目标函数：", round(J.item(), 6))
    print("W1 梯度形状：", tuple(grads["W1"].shape), " b2 梯度形状：", tuple(grads["b2"].shape))

    # 用数值梯度验一验 W2 的单个元素
    def loss_of_w2(w2):
        return forward_backward(x, W1, b1, w2, b2, y, lam=0.1)[0]

    ng = numeric_grad(loss_of_w2, W2)
    print("W2 的解析梯度与数值梯度是否吻合：", grad_check(grads["W2"], ng))

    print("xavier 界：", (6 / 150) ** 0.5, " 实测最大：", xavier_uniform(100, 50).abs().max().item())
