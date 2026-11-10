# B25 代码题

把你的实现写在 `work/b25/b25.py` 里，函数名与签名必须和下面一致。

验收：`just check b25`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章做什么

手写四个优化器的**单步更新**，并与 `torch.optim` 的对应实现逐步对比。

**这一章不训练网络**。每个函数只做一次参数更新，断言用固定梯度序列跑 10 步，
与 torch 的结果逐个元素比对。

**对比是这一章的核心**：损失下降只能说明方向对，逐步对比才能说明公式对。

---

## 公共约定

四个函数都是**原地更新**：直接改传进来的 `params` 与状态列表，不返回值。

状态张量（`velocity`、`s`、`v`）与参数**同形状**，初始全 0，由调用方创建。

---

## T1 动量法

```python
def sgd_momentum_step(params: list[torch.Tensor], grads: list[torch.Tensor],
                      velocity: list[torch.Tensor], lr: float,
                      momentum: float) -> None:
    """一步动量法更新。velocity 与 params 同形状，初始全 0。"""
```

**torch 的写法**（与书上差一个 `lr` 因子，见讲义 3.3）：

```
velocity = momentum * velocity + grad
params   = params - lr * velocity
```

| 检查 | 期望 |
|---|---|
| 与 `torch.optim.SGD(momentum=0.9)` 跑 10 步 | 逐个元素一致 |
| `momentum=0` | 退化为普通 SGD |

## T2 AdaGrad

```python
def adagrad_step(params: list[torch.Tensor], grads: list[torch.Tensor],
                 s: list[torch.Tensor], lr: float,
                 eps: float = 1e-10) -> None:
    """一步 AdaGrad 更新。s 是梯度平方的累加，初始全 0。"""
```

```
s      = s + grad * grad
params = params - lr * grad / (sqrt(s) + eps)
```

**`eps` 的位置**：这里写的是 `sqrt(s) + eps`（torch 的写法）。书上写的是
`sqrt(s + eps)`，两者差别极小（`eps` 是 1e-10 量级），断言用 `eps=1e-10` 时对得上。

| 检查 | 期望 |
|---|---|
| 与 `torch.optim.Adagrad` 跑 10 步 | 逐个元素一致 |
| 同样的梯度下更新量 | **逐步变小**（学习率被 `s` 衰减）|

## T3 RMSProp

```python
def rmsprop_step(params: list[torch.Tensor], grads: list[torch.Tensor],
                 s: list[torch.Tensor], lr: float, gamma: float = 0.9,
                 eps: float = 1e-10) -> None:
    """一步 RMSProp 更新。s 是梯度平方的指数加权移动平均，初始全 0。"""
```

```
s      = gamma * s + (1 - gamma) * grad * grad
params = params - lr * grad / (sqrt(s) + eps)
```

**注意 torch 里这个 `gamma` 叫 `alpha`**。

| 检查 | 期望 |
|---|---|
| 与 `torch.optim.RMSprop(alpha=0.9)` 跑 10 步 | 逐个元素一致 |
| 同样的梯度下更新量 | 从初始值**降到稳态后不再下降**（不像 AdaGrad 一路降到 0）|

## T4 Adam

```python
def adam_step(params: list[torch.Tensor], grads: list[torch.Tensor],
              v: list[torch.Tensor], s: list[torch.Tensor], lr: float,
              t: int, beta1: float = 0.9, beta2: float = 0.999,
              eps: float = 1e-8) -> None:
    """一步 Adam 更新，含偏差修正。

    v、s 与 params 同形状，初始全 0。t 是从 1 开始的步数。
    """
```

```
v      = beta1 * v + (1 - beta1) * grad
s      = beta2 * s + (1 - beta2) * grad * grad
v_hat  = v / (1 - beta1 ** t)          # 偏差修正
s_hat  = s / (1 - beta2 ** t)
params = params - lr * v_hat / (sqrt(s_hat) + eps)
```

| 检查 | 期望 |
|---|---|
| 与 `torch.optim.Adam` 跑 10 步 | 逐个元素一致 |
| 第一步的更新量 | **约为 `lr`**（偏差修正的效果，不修只有 `0.1 * lr`）|
| 状态形状 | 与参数一致，且被更新过 |

**`t` 从 1 开始**。`beta1 ** t` 在 `t=0` 时是 1，偏差修正会除出 0。

## T5 改错

下面这段想实现 Adam。**三处**问题。

```python
def adam_step(params, grads, v, s, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
    for p, g, vi, si in zip(params, grads, v, s):
        vi.mul_(beta1).add_(g, alpha=1 - beta1)
        si.mul_(beta2).addcmul_(g, g, value=1 - beta2)
        g_prime = lr * vi / (si.sqrt() + eps)
        p.data -= g_prime
```

调用方会从 `t=0` 开始逐次加一传入。

三处分别指：**`v` 与 `s` 的用法**、**`t` 的起点**、**反向传播时的保护**。

三段提示：第一处让前几步的更新量只有应有的 1/10；第二处会让 `beta1 ** t` 在第一步
等于 1、除出 0；第三处会让带梯度的参数在 `.backward()` 时抛原地操作错误。
