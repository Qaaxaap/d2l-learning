# B16 代码题

把你的实现写在 `work/b16/b16.py` 里，函数名与签名必须和下面一致。

验收：`just check b16`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章做什么

搭第一个卷积网络，在 Fashion-MNIST 上训练到可用精度。

**结构要求**：第一版的结构（无填充、最大池化），激活用 **ReLU**（不用 sigmoid，
理由见讲义 4.5 节）。

**数据已经在本机** `data/FashionMNIST`，`download=True` 不会重新下载。

验收分三类：**精确**（参数量、形状）、**范围**（精度下限）、**趋势**（损失下降）。

---

## T1 搭 LeNet

```python
def make_lenet(in_channels: int = 1, num_classes: int = 10) -> nn.Module:
    """第一版结构的 LeNet。

    输入 (N, in_channels, 28, 28)，输出 (N, num_classes)。

    结构（与讲义 2.1 节一致）：
        卷积 5×5 → ReLU → 最大池化 2×2 步幅 2
        卷积 5×5 → ReLU → 最大池化 2×2 步幅 2
        展平
        全连接 120 → ReLU
        全连接 84  → ReLU
        全连接 num_classes
    """
```

| 检查 | 期望 |
|---|---|
| 前向形状 | `(4, 1, 28, 28)` → `(4, 10)` |
| 参数量 | `in_channels=1, num_classes=10` 时 **44426** |
| 展平层的输入维度 | 256 |
| 激活函数 | 不能含 `nn.Sigmoid` |
| `state_dict` 里的键数 | 8（4 个卷积/线性层各 2 个）|

**参数量怎么算**（自己核对）：

```
卷积1:  6 * 1 * 5 * 5 + 6            = 156
卷积2: 16 * 6 * 5 * 5 + 16           = 2416
线性1: 120 * 256 + 120               = 30840
线性2: 84 * 120 + 84                 = 10164
线性3: 10 * 84 + 10                  = 850
合计                                  = 44426
```

## T2 评估准确率

```python
def evaluate(net: nn.Module, loader, device) -> float:
    """返回模型在 loader 上的分类准确率，范围 [0, 1]。不改变 net 的参数。"""
```

`loader` 是 `DataLoader`，每批产出 `(X, y)`：`X` 形状 `(batch, 1, 28, 28)`、
`y` 形状 `(batch,)`（类别索引 0 到 9）。

| 检查 | 期望 |
|---|---|
| 一个全猜 0 类的模型 | 准确率等于标签里 0 的比例 |
| 一个完美模型（标签就是预测）| 准确率 1.0 |
| 空 loader | 不报除零，返回 0.0 |
| `net` 的模式 | 调用后应当处于 eval 模式 |

## T3 训练

```python
def train_lenet(net: nn.Module, train_iter, test_iter, num_epochs: int,
                lr: float, device) -> dict:
    """训练 net，每个 epoch 记一次指标。

    返回 {"train_loss": [...], "train_acc": [...], "test_acc": [...]}
    三个列表的长度都等于 num_epochs。
    """
```

| 检查 | 期望 |
|---|---|
| 返回结构 | 三个键，各是长度 `num_epochs` 的列表 |
| 训练损失 | 最后一个 epoch 明显低于第一个 |
| 训练准确率 | 落在 [0, 1] |
| 测试准确率 | 用小数据集训练几个 epoch 后超过 0.5 |
| 训练后的参数 | 与训练前不同 |

**注意**：`train_iter` 与 `test_iter` 由调用方给，不自己造。

## T4 改错

下面这段代码想在 Fashion-MNIST 上训练 LeNet。**三处**问题。

```python
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

net = nn.Sequential(
    nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
    nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 10),
)

device = torch.device("cuda")
transform = transforms.Compose([])
train_set = datasets.FashionMNIST("data", True, transform=transform)
train_iter = DataLoader(train_set, batch_size=256, shuffle=True)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        loss = loss_fn(net(X), y)
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch + 1}, loss {loss.item():.4f}")
```

三处分别指：**数据变换**、**模型的设备**、**梯度的处理**。

三段提示：第一处报 `TypeError`，信息里会提到 `PIL.Image`；第二处要等第一处修好才暴露，
报设备不匹配；第三处不报错，但训练效果很差。
