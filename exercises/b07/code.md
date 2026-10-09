# B07 代码题

把你的实现写在 `work/b07/b07.py` 里，函数名和签名必须和下面一致。

验收：`just check b07`。断言全过，并且我读过代码之后，这一题才算过。

**这一章有四个解锁点**：`get_fashion_mnist_labels`、`show_fashion_mnist`（都在纸质书 3.5）、
`evaluate_accuracy`、`train_ch3`（3.6）。按仓库规则，在书上出现那句话之前要自己写。

**T1 到 T5 不许用 `nn.CrossEntropyLoss`、`nn.NLLLoss`、`torch.log_softmax`、`torch.logsumexp`**，
softmax 与交叉熵都要自己算，数值稳定的平移也要自己写（T6 才用框架封装）。

数据已经在仓库的 `data/` 下（Fashion-MNIST，训练 60000 张、测试 10000 张，28×28 灰度）。

---

## T1 softmax 与交叉熵

```python
def softmax(o: torch.Tensor) -> torch.Tensor:
    """按最后一维做 softmax，返回同形状的概率。

    要求数值稳定：logits 里有 1000 这样的数时不能出现 nan。
    input 形状 (n, q) 时按第 1 维归一化，每一行之和为 1。
    """


def cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """交叉熵损失。

    - logits 形状 (n, q)，是未经 softmax 的分数
    - y 形状 (n,)，元素是 0 到 q-1 的整数标签（不是独热）
    - 返回标量：这一批样本的交叉熵**之和**（不取平均）

    不许调用 nn.CrossEntropyLoss、torch.log_softmax 或 torch.logsumexp。
    """
```

| 调用 | 期望 |
|---|---|
| `softmax(torch.tensor([[1000.0, 0.0]]))` | `tensor([[1., 0.]])`，不能是 nan |
| `softmax(torch.zeros(3, 4))` | 每一行都是 0.25 |
| `softmax(torch.randn(5, 10)).sum(dim=1)` | 每个元素都接近 1 |
| `cross_entropy(torch.tensor([[2.0, 1.0, 0.1]]), torch.tensor([0]))` | 约 `0.41703` |
| `cross_entropy(logits, y)` 与 `nn.CrossEntropyLoss(reduction="sum")(logits, y)` | 数值一致（允许 1e-5 误差） |

第二张表最后一行是自查办法：拿框架的结果对自己的。注意对照的是 `reduction="sum"`，
因为这里的约定是返回和，与 B06 的 `squared_loss` 一致——除以批量大小那一步放在 `sgd` 里做，
**两边都除会把学习率压小几百倍**。

## T2 精度

```python
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """返回命中率，Python 浮点数。

    logits 形状 (n, q)，y 形状 (n,) 是整数标签。
    不需要先算 softmax（想想为什么）。
    """
```

| 调用 | 期望 |
|---|---|
| `accuracy(torch.tensor([[2.0, 1.0], [0.1, 3.0]]), torch.tensor([0, 1]))` | `1.0` |
| `accuracy(torch.tensor([[1.0, 2.0], [3.0, 0.1]]), torch.tensor([0, 1]))` | `0.0` |
| `accuracy(torch.zeros(4, 3), torch.tensor([0, 1, 2, 0]))` | `0.5`（全零 logits 时 argmax 全取 0，四个里对两个） |

## T3 在数据集上评估

```python
def evaluate_accuracy(net, data_iter) -> float:
    """net 是可调用对象，接受 (n, 784) 返回 (n, 10) 的 logits。

    在整个 data_iter 上算准确率，返回 Python 浮点数。
    评估过程不应建立计算图。
    """
```

| 调用 | 期望 |
|---|---|
| 未训练的随机模型（10 类） | 三成以下（随机水平附近） |
| 训练好的模型 | 0.8 以上 |

## T4 从零实现：训练

```python
def make_net(num_inputs: int = 784, num_outputs: int = 10, seed: int = 0):
    """返回 (W, b) 两个叶子张量：W 形状 (784, 10) 用均值 0、标准差 0.01 的正态随机数，
    b 形状 (10,) 用全零。都用 seed 保证可复现，都开 requires_grad。"""


def train_from_scratch(train_iter, test_iter, num_epochs: int = 5,
                       lr: float = 0.1, batch_size: int = 256,
                       seed: int = 0) -> dict:
    """用手写的 softmax、交叉熵与 sgd 训练。

    返回一个字典，键固定为：
        train_loss -> 每个 epoch 的平均训练损失，浮点列表
        train_acc  -> 每个 epoch 结束时的训练集准确率，浮点列表
        test_acc   -> 每个 epoch 结束时的测试集准确率，浮点列表
    三个列表长度都是 num_epochs。
    """
```

| 检查 | 期望 |
|---|---|
| 5 轮之后 | `test_acc[-1]` 达到 0.75 以上（10000 样本的子集） |
| `train_loss` | 逐轮下降 |
| 三个列表的长度 | 都等于 `num_epochs` |

超参数用 5 轮、学习率 0.1、批量 256（讲义 5.4 节说明了为什么与 B06 不同）。

## T5 混淆矩阵

```python
def confusion_matrix(net, data_iter, q: int = 10) -> torch.Tensor:
    """返回 (q, q) 的整数张量。

    第 i 行第 j 列 = 真实类别是 i 而被预测成 j 的样本数。
    对角线上是预测正确的数量，整张表所有元素之和等于样本总数。
    """
```

| 检查 | 期望 |
|---|---|
| 矩阵形状 | `(10, 10)` |
| 所有元素之和 | 等于 `data_iter` 里的样本总数 |
| 对角线之和 / 总数 | 与 `evaluate_accuracy` 返回的准确率一致（误差 1e-6 内） |

## T6 简洁实现

```python
def train_concise(train_iter, test_iter, num_epochs: int = 5,
                  lr: float = 0.1, seed: int = 0) -> dict:
    """用 nn.Linear、nn.CrossEntropyLoss、torch.optim.SGD 重写一遍。

    返回的字典键与 T4 相同。
    """
```

| 检查 | 期望 |
|---|---|
| 5 轮之后 | `test_acc[-1]` 达到 0.75 以上，与 T4 同一量级 |

**注意**：`nn.CrossEntropyLoss` 接收的是 logits，**不要再套一层 softmax**（讲义 6.2 节）。
两版的损失数值不会完全相同（自己写的按批量平均，框架版也是），但精度应当接近。

## T7 改错

下面这段训练代码有三处问题。把代码抄进 `work/b07/b07.py` 顶部的注释里，
每处标出三样：**位置**（第几行）、**为什么错**、**怎么改**。

```python
import torch
import torch.nn as nn

net = nn.Sequential(nn.Linear(784, 10))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

for epoch in range(5):
    for X, y in train_iter:
        X = X.reshape(X.shape[0], -1)
        out = torch.softmax(net(X), dim=1)
        l = loss_fn(out, y)
        optimizer.step()
        l.backward()
```

三处分别关于：**损失的输入**、**梯度的生命周期**、**参数更新的时机**。

三段提示：三处都不报错，只是训练结果不对；第一处会让损失降得很慢；
第二处是**少了**一个调用；第三处是两句的**顺序反了**。
