"""B25 参考答案。做完再看。

跑一遍：
    python3 solutions/b25/b25.py
或者拿它验断言：
    python3 tools/check_b25.py solutions/b25/b25.py
"""

import torch


# ---------- T1 动量法 ----------

def sgd_momentum_step(params: list[torch.Tensor], grads: list[torch.Tensor],
                      velocity: list[torch.Tensor], lr: float,
                      momentum: float) -> None:
    """一歩动量法更新，原地修改 params 与 velocity。

    velocity 与 params 同形状，初始全 0。
    torch 的约定：buf = momentum*buf + g，然后 p -= lr*buf。
    """
    for p, g, v in zip(params, grads, velocity):
        v.mul_(momentum).add_(g)
        p.add_(v, alpha=-lr)


# ---------- T2 AdaGrad ----------

def adagrad_step(params: list[torch.Tensor], grads: list[torch.Tensor],
                 s: list[torch.Tensor], lr: float,
                 eps: float = 1e-10) -> None:
    """一步 AdaGrad 更新。s 是梯度平方的累加，初始全 0。

    用书上的写法：x -= lr * g / sqrt(s + eps)。
    """
    for p, g, si in zip(params, grads, s):
        si.addcmul_(g, g)                      # s += g ⊙ g
        p.addcdiv_(g, si.sqrt().add(eps), value=-lr)


# ---------- T3 RMSProp ----------

def rmsprop_step(params: list[torch.Tensor], grads: list[torch.Tensor],
                 s: list[torch.Tensor], lr: float, gamma: float = 0.9,
                 eps: float = 1e-10) -> None:
    """一步 RMSProp 更新。s 是梯度平方的指数加权移动平均，初始全 0。"""
    for p, g, si in zip(params, grads, s):
        si.mul_(gamma).addcmul_(g, g, value=1 - gamma)
        p.addcdiv_(g, si.sqrt().add(eps), value=-lr)


# ---------- T4 Adam ----------

def adam_step(params: list[torch.Tensor], grads: list[torch.Tensor],
              v: list[torch.Tensor], s: list[torch.Tensor], lr: float,
              t: int, beta1: float = 0.9, beta2: float = 0.999,
              eps: float = 1e-8) -> None:
    """一步 Adam 更新，含偏差修正。

    v、s 与 params 同形状，初始全 0。
    t 是**从 1 开始**的步数，偏差修正用 1 - beta1**t。
    """
    for p, g, vi, si in zip(params, grads, v, s):
        vi.mul_(beta1).add_(g, alpha=1 - beta1)          # v = β1 v + (1-β1) g
        si.mul_(beta2).addcmul_(g, g, value=1 - beta2)   # s = β2 s + (1-β2) g²
        v_hat = vi / (1 - beta1 ** t)                     # 偏差修正
        s_hat = si / (1 - beta2 ** t)
        p.addcdiv_(v_hat, s_hat.sqrt().add(eps), value=-lr)


"""T5 改错

第 1 处：`v` 与 `s` 没有偏差修正
    `v = β1 v + (1-β1)g` 在 t=1 时只有 0.1g。不除 `1-β1^t` 的话前几步更新量只有应有的
    1/10，训练前半段变慢。
    改法：算 `v_hat = v / (1 - beta1**t)`、`s_hat = s / (1 - beta2**t)`。

第 2 处：`t` 从 0 开始算，`beta1 ** t` 在 t=0 时是 1，除出来是 0
    步数要从 1 开始，或者用 `t + 1`。
    改法：每步先 `t += 1` 再算偏差修正。

第 3 处：`p.data -= ...` 里用了 `.data`，但 `v` 的更新没有 `no_grad`
    在 autograd 记录下修改需要梯度的张量会报 "a leaf Variable that requires grad
    is being used in an in-place operation"。
    改法：整个更新放在 `torch.no_grad()` 里。
"""


if __name__ == "__main__":
    torch.manual_seed(0)
    p = [torch.randn(3)]
    g = [torch.randn(3)]
    v = [torch.zeros(3)]
    sgd_momentum_step(p, g, v, 0.1, 0.9)
    print("T1 动量:", p[0].tolist()[:3])

    p = [torch.randn(3)]
    s = [torch.zeros(3)]
    adagrad_step(p, [torch.randn(3)], s, 0.1)
    print("T2 AdaGrad:", p[0].tolist()[:3])

    p = [torch.randn(3)]
    s = [torch.zeros(3)]
    rmsprop_step(p, [torch.randn(3)], s, 0.1)
    print("T3 RMSProp:", p[0].tolist()[:3])

    p = [torch.randn(3)]
    v, s = [torch.zeros(3)], [torch.zeros(3)]
    adam_step(p, [torch.randn(3)], v, s, 0.1, t=1)
    print("T4 Adam:", p[0].tolist()[:3])
