# B06 代码题

把你的实现写在 `work/b06/b06.py` 里，函数名和签名必须和下面一致。

验收：`just check b06`。断言全过，并且我读过代码之后，这一题才算过。

**这一章有四个解锁点**：`data_iter`、`linreg`、`squared_loss`、`sgd`。纸质书里它们标着
"本函数已保存在 d2lzh 包中"，按仓库规则在书里出现那句话之前要自己写。所以下面这几道题
是自己实现，不许调 `d2l` 包或 `torch.utils.data` 里的现成封装——除了 T6 明确要求的那些。

---

## T1 生成人造数据集

```python
def make_data(n: int, w: torch.Tensor, b: float, noise: float = 0.01,
              seed: int = 0) -> tuple:
    """生成线性回归的人造数据。

    - X 形状 (n, len(w))，每个元素从标准正态分布抽
    - 真实关系是 y = X @ w + b，再加一个噪声项
    - 噪声从均值 0、标准差 noise 的正态分布抽，形状与 y 相同
    - 同一个 seed 必须给出完全相同的结果
    - y 的形状是 (n, 1)

    返回 (X, y)，都是 float32 张量。
    """
```

| 调用 | 期望 |
|---|---|
| `make_data(1000, torch.tensor([2.0, -3.4]), 4.2)` | `X` 是 `(1000, 2)`，`y` 是 `(1000, 1)` |
| 同参数调两次 | 两次结果完全相同 |
| `y` 与 `X @ w + b` 之差 | 标准差接近 0.01 |

## T2 `data_iter`：分批读数据

```python
def data_iter(batch_size: int, features: torch.Tensor, labels: torch.Tensor,
              seed: int = 0):
    """按批量大小随机切分数据，逐批产出 (X, y)。

    要求：
    - 是生成器（用 yield），不是返回一个列表
    - 每轮把样本顺序打乱后再切分
    - 最后一个批量可以不满（样本数不被批量大小整除时）
    - 同一个 seed 给出相同的切分顺序
    """
```

| 调用 | 期望 |
|---|---|
| `list(data_iter(10, X, y))`，X 有 1000 行 | 100 个批量，每个特征形状 `(10, 2)`、标签形状 `(10, 1)` |
| `list(data_iter(3, X, y))` | 334 个批量，最后一个只有 1 个样本 |
| 把所有批量的特征拼起来 | 与 `X` 是同一组行（顺序不同、内容不丢） |
| 同参数调两次 | 切分顺序完全相同 |

`X` 有 1000 行时 `batch_size=3`：334 个批量（333 个满的加 1 个只剩 1 个样本的）。
拼接用 `torch.cat(..., dim=0)`（B03 第 8 节）。

## T3 模型与损失

```python
def linreg(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """线性回归的预测：X @ w + b。"""


def squared_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """平方损失：(y_hat - y)² / 2，逐元素，不取平均。"""
```

| 调用 | 期望 |
|---|---|
| `linreg(torch.ones(3, 2), torch.tensor([[1.], [2.]]), torch.tensor([0.5]))` | `(3, 1)`，每行 `3.5` |
| `squared_loss(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 4.0]))` | `tensor([0.1250, 2.0000])` |
| `squared_loss` 的输入形状不同但可广播 | 结果形状与 `y_hat` 相同 |

`squared_loss` 里要先把 `y` 变形成 `y_hat` 的形状，书上这么写是为了处理 `y` 是 `(n,)`
而 `y_hat` 是 `(n, 1)` 的情况。形状规则见 B02 第 4 节。

## T4 `sgd`：参数更新

```python
def sgd(params: list, lr: float, batch_size: int) -> None:
    """小批量随机梯度下降。对 params 里的每个参数做：

        p ← p - lr * p.grad / batch_size

    要求：
    - 就地更新，不要返回新张量
    - 更新后该参数的 .grad 要清零
    - 更新过程中不许产生新的计算图
    """
```

| 情形 | 期望 |
|---|---|
| 参数的 `.grad` 是 `[2, 4]`，`lr=0.1`，`batch_size=2` | 参数减少 `[0.1, 0.2]` |
| 更新之后 | 该参数的 `.grad` 全是 0 |
| 连续调用两次（中间不重新反传） | 第二次不改变参数（梯度已清零） |

## T5 训练循环：从零实现

```python
def train_scratch(X: torch.Tensor, y: torch.Tensor, lr: float = 0.03,
                  num_epochs: int = 3, batch_size: int = 10,
                  seed: int = 0) -> tuple:
    """用手写的零件训练线性回归。

    用 T2 的 data_iter 取批、T3 的 linreg 与 squared_loss 算损失、T4 的 sgd 更新参数。
    参数自己初始化：w 用均值 0、标准差 0.01 的正态随机数，形状 (d, 1)；b 用全零，形状 (1,)。
    两者都要开 requires_grad。

    返回 (w, b, losses)：
    - w、b 是训练完的参数
    - losses 是每个 epoch 结束时的训练损失，Python 浮点数组成的列表，长度 num_epochs
    """
```

| 检查 | 期望 |
|---|---|
| 用 T1 生成的数据（真实 `w=[2, -3.4]`、`b=4.2`），跑 3 轮 | 学到的 `w`、`b` 与真实值相差在 0.05 以内 |
| `losses` 的长度 | 等于 `num_epochs`，而且逐轮下降 |

**注意**：`loss` 是小批量上的和（逐元素平方损失之和），不是标量。反传前要先求和成标量，
这一点与书上不同（MXNet 会隐式求和）。

## T6 简洁实现

```python
def train_concise(X: torch.Tensor, y: torch.Tensor, lr: float = 0.03,
                  num_epochs: int = 3, batch_size: int = 10,
                  seed: int = 0) -> tuple:
    """用框架的封装重写一遍，返回 (net, losses)。

    - net 是 nn.Sequential(nn.Linear(d, 1)) 这样的模块
    - 数据用 TensorDataset 包起来，交给 DataLoader 分批
    - 损失用 nn.MSELoss()
    - 优化器用 torch.optim.SGD
    - losses 与 T5 一样，是每个 epoch 结束时的损失，Python 浮点数组成的列表
    """
```

| 检查 | 期望 |
|---|---|
| 用同一份数据跑 3 轮 | `net.weight` 与 `net.bias` 接近真实值 |
| 与自己写的那版比 | 同样超参数下，两者的损失在同一量级 |

**注意**：`nn.MSELoss()` 取的是**平均**，书上的 `squared_loss` 不取平均也不除以 2。
所以两版的损失数值不会完全相等（差一个因子），断言只比较参数的收敛结果。

## T7 改错

下面这段训练循环有三处问题。把代码抄进 `work/b06/b06.py` 顶部的注释里，
每处标出三样：**位置**（第几行）、**为什么错**、**怎么改**。

```python
import torch
import torch.nn as nn

w = torch.randn(2, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr, batch_size = 0.5, 10
net = lambda X: X @ w + b
loss_fn = lambda y_hat, y: (y_hat - y) ** 2

for epoch in range(3):
    for X, y in batches:
        l = loss_fn(net(X), y)
        l.backward()
        w = w - lr * w.grad / batch_size
        b = b - lr * b.grad / batch_size
```

三处分别关于：**非标量反传**、**参数更新的写法**、**梯度的生命周期**。
