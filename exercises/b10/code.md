# B10 代码题

把你的实现写在 `work/b10/b10.py` 里，函数名与签名必须和下面一致。

验收：`just check b10`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章的两个实验

**实验一：权重衰减**（讲义 1.3 节）。造一个必然过拟合的场景——
200 维特征、20 个训练样本——然后比较 `lam=0` 与 `lam` 不为 0 的差别。
这个实验全程在小数据上跑，很快。

**实验二：丢弃法**（讲义 2 节）。用 Fashion-MNIST 训练一个带丢弃层的 MLP，
比较有无丢弃的测试精度。这个实验要跑几轮，比 B09 慢。

两个实验共用 B06 的零件：平方损失、交叉熵、`sgd`、小批量。新增的只有三样：
惩罚项、丢弃函数、训练/推理模式切换。

验收分三类：**精确**（惩罚项数值、掩码的统计性质、丢弃函数的三个边界）、
**范围**（测试精度阈值）、**趋势**（加正则化之后训练损失与测试损失的差距变小）。

---

## T1 造高维数据

```python
def make_high_dim(n_train: int = 20, n_test: int = 100,
                  num_inputs: int = 200, seed: int = 0) -> tuple:
    """按讲义 1.3 节的公式生成数据。

    返回 (x_train, y_train, x_test, y_test)：
    x 形状 (n, 200)，取标准正态随机数；y 形状 (n, 1)。
    标签按 y = 0.05 + 0.01 * sum(x 的每一维) + 噪声，噪声标准差 0.01。
    四个张量都是 float32，同一个 seed 给出相同结果。
    """
```

| 检查 | 期望 |
|---|---|
| 四个形状 | `(20, 200)`、`(20, 1)`、`(100, 200)`、`(100, 1)` |
| `dtype` | 都是 `float32` |
| 同一 seed 调两次 | 结果完全相同 |
| `y` 的标准差 | 在 0.1 到 1.0 之间（200 维各 0.01 的贡献，量级约 0.14）|

## T2 惩罚项

```python
def l2_penalty(w: torch.Tensor, lam: float) -> torch.Tensor:
    """返回 (lam / 2) * sum(w ** 2)，一个标量张量。"""
```

| 调用 | 期望 |
|---|---|
| `l2_penalty(torch.tensor([3.0, 4.0]), 1.0)` | `12.5`（即 $(9+16)/2$）|
| `l2_penalty(torch.tensor([3.0, 4.0]), 0.0)` | `0.0` |
| `l2_penalty(torch.zeros(5), 2.0)` | `0.0` |

## T3 带惩罚项的训练

```python
def train_high_dim(train_iter, test_iter, lam: float = 0.0,
                   num_epochs: int = 3, lr: float = 0.003,
                   seed: int = 0) -> dict:
    """在高维数据上训练线性回归，返回 {"train_loss": [...], "test_loss": [...]}。

    - train_iter 与 test_iter 由调用方给（DataLoader），产出 (X, y)，形状分别是
      (batch, 200) 与 (batch, 1)
    - 权重 w 形状 (200, 1)，偏置 b 形状 (1,)，都用正态随机数初始化（标准差 0.01）
    - 训练时的损失 = 这一批的平方损失之和 + lam/2 * ||w||^2 / 批量大小
    - **记录进列表的是不含惩罚项的平均平方损失**（讲义 1.4 节）
    - 两个列表长度都是 num_epochs
    """
```

| 检查 | 期望 |
|---|---|
| `lam=0`、3 轮跑完 | 训练损失很低（低于 0.05），测试损失明显更高 |
| `lam` 不为 0 时 | 测试损失比 `lam=0` 时低 |
| `lam=0` 与 `lam=3` 的对比 | `lam=3` 的训练损失高于 `lam=0`（这是正常的，正则化换来了泛化）|
| 两个列表 | 长度都是 3，且是不同的列表对象 |

## T4 丢弃函数

```python
def dropout_layer(X: torch.Tensor, dropout: float) -> torch.Tensor:
    """按丢弃概率 dropout 随机清零 X 的元素，并把留下的元素除以 (1 - dropout)。

    三个边界：
      dropout == 0 时原样返回
      dropout == 1 时返回全零（与 X 同形状、同 dtype、同设备）
    其余情况随机决定每个元素是否保留。

    返回与 X 同形状同 dtype 的新张量，不修改 X 本身。
    """
```

| 检查 | 期望 |
|---|---|
| `dropout=0` | 返回值与 `X` 完全相等 |
| `dropout=1` | 全零，形状与 `X` 相同 |
| `dropout=0.5`，大张量上跑 | 约一半元素为 0（比例在 0.4 到 0.6 之间）|
| 留下的元素的**均值** | 与 `X` 的均值接近（说明做过 $1/(1-p)$ 缩放）|
| 调用两次 | 结果不同（确实是随机的）|
| `X` 本身 | 没被改动 |

## T5 带丢弃层的 MLP

```python
def make_dropout_net(num_inputs: int = 784, num_hiddens1: int = 256,
                     num_hiddens2: int = 256, num_outputs: int = 10,
                     dropout1: float = 0.2, dropout2: float = 0.5,
                     seed: int = 0):
    """返回一个 nn.Sequential：
    Flatten → Linear(784,256) → ReLU → Dropout(dropout1)
            → Linear(256,256) → ReLU → Dropout(dropout2)
            → Linear(256,10)
    """


def evaluate(net, loader) -> float:
    """net 接受 (batch, 1, 28, 28) 返回 (batch, 10) 的 logits。

    在整个 loader 上算准确率，返回 Python 浮点数。
    **评估时丢弃层不应生效**，而且评估前后要正确恢复模型原来的模式。
    """


def train_dropout(train_iter, test_iter, use_dropout: bool = True,
                  num_epochs: int = 5, lr: float = 0.5,
                  batch_size: int = 256, seed: int = 0) -> dict:
    """训练上面那个 MLP，返回 {"train_loss": [...], "train_acc": [...],
    "test_acc": [...]}，三个列表长度都是 num_epochs，元素是 Python 浮点数。

    use_dropout=False 时两个丢弃层的概率都设成 0。
    损失用 nn.CrossEntropyLoss（它返回的是平均损失，记录的也是平均，不必再除）。
    """
```

| 检查 | 期望 |
|---|---|
| 5 轮、`use_dropout=True` | `test_acc[-1]` 达到 0.75 以上 |
| 三个列表 | 长度都是 5，是三个不同的对象 |
| `evaluate` 调两次 | **结果完全相同**（评估时丢弃层不生效）|
| `evaluate` 之后再训练一轮 | 训练正常（模式切回去了）|

## T6 改错

下面这段代码想训练一个带丢弃层的模型，有三处问题。把代码抄进 `work/b10/b10.py`
顶部的注释里，每处标出**位置**、**为什么错**、**怎么改**。

```python
import torch
import torch.nn as nn

def test_acc(net, loader):
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            correct += (net(X).argmax(dim=1) == y).sum().item()
            total += y.numel()
    return correct / total

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.Linear(256, 10),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5, weight_decay=0.01)

for epoch in range(5):
    total, n = 0, 0
    for X, y in train_iter:
        optimizer.zero_grad()
        loss = loss_fn(net(X), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * y.numel()
        n += y.numel()
    print(f"epoch {epoch}: 训练损失 {total / n:.4f}  测试精度 {test_acc(net, test_iter):.4f}")
```

三处分别指三个位置：**丢弃层与激活函数的顺序**、**评估函数的写法**、
**`weight_decay` 与这个实验的关系**。

三段提示：第一处不会报错，但 dropout 作用错了东西；第二处这个函数少做了一件事，
在当前调用位置会给出偏低的精度；第三处是这个实验本来要比较的东西，加上它之后就比较不成了。

（顺带说一句：`weight_decay` 与丢弃法可以同时用，这里要去掉是因为实验设计要归因，
不是两者冲突。讲义 1.5 节有 `weight_decay` 的有效强度正比于学习率的实测数据。）
