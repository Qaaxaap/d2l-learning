# B08 代码题

把你的实现写在 `work/b08/b08.py` 里，函数名和签名必须和下面一致。

验收：`just check b08`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：关于数据与迭代器

与 B07 相同的部分：

- `loader`、`train_iter`、`test_iter` 都是**参数**，由验收脚本传进来（`DataLoader`）
- 每次迭代给一个 `(X, y)`：`X` 形状 `(batch, 1, 28, 28)`、像素值 0 到 1；`y` 形状 `(batch,)`，元素是 0 到 9 的整数
- 最后一个批量的样本数可能小于 `batch_size`
- 线性层只接受 `(n, 784)`，**拉平要在你的函数里做**（或交给 `nn.Flatten()`）

**这一章的断言用全量数据**（6 万训练、1 万测试），10 轮训练约 12 秒。
原因是 MLP 的参数量是 20 万，用 1 万样本训练会不稳（讲义 4.5 节有实测数据）。

验收同样是三类：**精确**（形状、参数个数、激活函数的逐元素结果）、
**范围**（精度阈值）、**趋势**（损失逐轮下降）。

---

## T1 激活函数

```python
def relu(x: torch.Tensor) -> torch.Tensor:
    """逐元素取 x 与 0 的较大值。

    不许用 torch.relu、torch.nn.functional.relu 或 nn.ReLU。
    """


def relu_grad(x: torch.Tensor) -> torch.Tensor:
    """ReLU 的导数，逐元素。

    x > 0 处为 1，x < 0 处为 0，x == 0 处取 0。
    """
```

| 调用 | 期望 |
|---|---|
| `relu(torch.tensor([-2., -0.5, 0., 0.5, 2.]))` | `[0., 0., 0., 0.5, 2.]` |
| `relu_grad(torch.tensor([-2., 0., 3.]))` | `[0., 0., 1.]` |
| `relu` 对二维张量 | 逐元素处理，形状不变 |

## T2 参数与模型

```python
def init_params(num_inputs: int = 784, num_hiddens: int = 256,
                num_outputs: int = 10, seed: int = 0) -> tuple:
    """返回 (W1, b1, W2, b2)。

    W1 形状 (784, 256)，b1 形状 (256,)，W2 形状 (256, 10)，b2 形状 (10,)。
    两个 W 用均值 0、标准差 0.01 的正态随机数，两个 b 全零。
    四个都要开 requires_grad。同一个 seed 给出相同结果。
    """


def mlp_forward(X: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor,
                W2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """X 形状 (n, 1, 28, 28)。返回 (n, 10) 的 logits（不做 softmax）。"""
```

| 调用 | 期望 |
|---|---|
| `init_params()` 的四个形状 | `(784, 256)`、`(256,)`、`(256, 10)`、`(10,)` |
| 四个参数的 `requires_grad` | 都是 `True` |
| 同参数调两次 | 结果完全相同 |
| `mlp_forward(torch.randn(4, 1, 28, 28), *params)` | 形状 `(4, 10)` |
| 输出的取值范围 | 有负数（说明没过 ReLU，也没过 softmax） |

## T3 参数量

```python
def count_params(net) -> int:
    """返回 net 里所有可训练参数的个数之和，Python 整数。"""
```

| 调用 | 期望 |
|---|---|
| 上面那个 MLP（`nn.Sequential(nn.Flatten(), nn.Linear(784,256), nn.ReLU(), nn.Linear(256,10))`） | `203530` |
| B07 的单层模型（`nn.Linear(784, 10)`） | `7850` |

## T4 从零实现：训练

这一题需要三个零件：T1 的 `relu`、T2 的参数与 `mlp_forward`、以及下面这个 `sgd`。

```python
def sgd(params: list, lr: float, batch_size: int) -> None:
    """与 B07 的 T4 相同，这一章自己再写一遍。

    对每个参数做 p ← p - lr * p.grad / batch_size，更新后就地清零它的梯度。
    就地更新，过程中不建计算图。
    """


def train_from_scratch(train_iter, test_iter, num_epochs: int = 10,
                       lr: float = 0.1, batch_size: int = 256,
                       seed: int = 0) -> dict:
    """用手写的 relu、mlp_forward、cross_entropy 与 sgd 训练。

    交叉熵与准确率用 B07 里自己写的那两份（`cross_entropy` 与 `evaluate_accuracy`，
    抄进本章或重新实现一遍都行）。
    返回一个字典，键固定为：
        train_loss -> 每个 epoch 的平均训练损失，浮点列表
        train_acc  -> 每个 epoch 结束时的训练集准确率，浮点列表
        test_acc   -> 每个 epoch 结束时的测试集准确率，浮点列表
    三个列表长度都是 num_epochs，**且必须是三个独立的列表**。
    """
```

| 检查 | 期望 |
|---|---|
| 10 轮之后 | `test_acc[-1]` 达到 0.82 以上 |
| `train_loss` | 逐轮下降，第一轮在 0.3 到 2.0 之间（实测约 0.86），最后一轮更低 |

**注意 `train_loss` 是每个 epoch 的**平均**损失**，不是某一批的。未训练时单批损失约 log(10) ≈ 2.3，但一轮有 234 批，第一轮跑完平均值已经降到 0.86 左右。

**算 epoch 平均损失时留意两种约定**：T4 自己写的 `cross_entropy` 返回的是这一批的**和**，
T5 的 `nn.CrossEntropyLoss` 默认返回**平均**。要得到同样的 `train_loss`，
T4 直接累加再除以样本数，T5 得先乘回样本数再除以样本数（或者换个 reduction，
见 B07 讲义 5.1.3）。写错的话数值会差一个批量大小的量级。
| 三个列表的长度 | 都等于 `num_epochs` |

## T5 简洁实现

```python
def train_concise(train_iter, test_iter, num_epochs: int = 10,
                  lr: float = 0.1, seed: int = 0) -> dict:
    """用框架的封装重写一遍。

    模型用 nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))，
    损失用 nn.CrossEntropyLoss，优化器用 torch.optim.SGD。
    返回的字典键与 T4 相同，三个键是三个独立的列表。
    """
```

| 检查 | 期望 |
|---|---|
| 10 轮之后 | `test_acc[-1]` 达到 0.82 以上 |
| 与自己写的那版比 | 精度在同一量级（允许因初始化与打乱顺序有几分的差异） |

## T6 改错

下面这段训练代码有三处问题。把代码抄进 `work/b08/b08.py` 顶部的注释里，
每处标出三样：**位置**（第几行）、**为什么错**、**怎么改**。

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.Linear(256, 10),
    nn.ReLU(),
)
for name, p in net.named_parameters():
    with torch.no_grad():
        p.zero_()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

for epoch in range(10):
    for X, y in train_iter:
        optimizer.zero_grad()
        l = loss_fn(net(X), y)
        l.backward()
        optimizer.step()
```

三处分别关于：**少了激活函数**、**ReLU 的位置**、**初始化**。

三段提示：三处都不报错，只是训练结果不对；前两处都在 `nn.Sequential` 那几行里，
一个该有的没有、一个出现的位置不对；第三处会让所有隐藏单元永远保持相同。
