# B17 代码题

把你的实现写在 `work/b17/b17.py` 里，函数名与签名必须和下面一致。

验收：`just check b17`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章做什么

搭四个经典网络，验证结构与参数量。**这一章不训练**——四个网络都在 ImageNet 尺度设计，
本机用它跑 Fashion-MNIST 只是为了确认形状能走通。

输入统一是 `(N, 1, 224, 224)`，输出 `(N, num_classes)`。

验收分两类：**精确**（输出形状、参数量）、**结构**（有哪些层）。

参数量要精确对上，所以形状链算错一个数就会暴露。

---

## T1 AlexNet

```python
def make_alexnet(in_channels: int = 1, num_classes: int = 10) -> nn.Module:
    """输入 (N, in_channels, 224, 224)。

    卷积部分：卷积 11×11 步幅 4（96）→ ReLU → 最大池化 3×3 步幅 2
              卷积 5×5 填充 2（256）→ ReLU → 最大池化 3×3 步幅 2
              卷积 3×3 填充 1（384）→ ReLU
              卷积 3×3 填充 1（384）→ ReLU
              卷积 3×3 填充 1（256）→ ReLU → 最大池化 3×3 步幅 2

    全连接部分：Flatten（展平后 256×5×5 = 6400）
                → Linear(6400, 4096) → ReLU → Dropout(0.5)
                → Linear(4096, 4096) → ReLU → Dropout(0.5)
                → Linear(4096, num_classes)
    """
```

| 检查 | 期望 |
|---|---|
| 前向形状 | `(1, 1, 224, 224)` → `(1, 10)` |
| 参数量 | **46,764,746** |
| Dropout 层数 | 2 |
| 第一个卷积层 | `kernel_size=11, stride=4` |

**参数以 46764746 为准**，自己算一遍形状链核对。

## T2 VGG

```python
def vgg_block(num_convs: int, num_channels: int,
              in_channels: int | None = None) -> nn.Module:
    """num_convs 个 3×3 卷积（填充 1、ReLU）+ 一个 2×2 步幅 2 的最大池化。

    in_channels 为 None 时，块内所有卷积的输入输出都是 num_channels。
    """


def make_vgg(conv_arch, num_classes: int = 10,
             in_channels: int = 1) -> nn.Module:
    """conv_arch 形如 ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))。

    全连接部分：Flatten（展平后 512×7×7 = 25088）
                → Linear(25088, 4096) → ReLU → Dropout(0.5)
                → Linear(4096, 4096) → ReLU → Dropout(0.5)
                → Linear(4096, num_classes)
    """
```

| 检查 | 期望 |
|---|---|
| `vgg_block(2, 128)` 对 `(1, 128, 56, 56)` | `(1, 128, 28, 28)` |
| `vgg_block(1, 64, in_channels=3)` 对 `(1, 3, 224, 224)` | `(1, 64, 112, 112)` |
| VGG-11 前向 | `(1, 1, 224, 224)` → `(1, 10)` |
| VGG-11 参数量 | **128,806,154** |
| 展平后维度 | 25088 |

## T3 NiN

```python
def nin_block(num_channels: int, kernel_size: int, strides: int,
              padding: int, in_channels: int | None = None) -> nn.Module:
    """一个普通卷积（核大小、步幅、填充可配）+ 两个 1×1 卷积，每个后接 ReLU。"""


def make_nin(in_channels: int = 1, num_classes: int = 10) -> nn.Module:
    """结构见讲义 3.4 节。最后用 AdaptiveAvgPool2d(1) + Flatten。"""
```

| 检查 | 期望 |
|---|---|
| `nin_block` 对 `(1, 96, 54, 54)` | `(1, 96, 54, 54)` |
| NiN 前向 | `(1, 1, 224, 224)` → `(1, 10)` |
| NiN 参数量 | **1,992,166** |
| 有没有 `nn.Linear` | **没有** |
| 有没有 `AdaptiveAvgPool2d` | 有 |

## T4 GoogLeNet

```python
class Inception(nn.Module):
    def __init__(self, c1, c2, c3, c4, in_channels: int):
        """c1、c4 是整数；c2、c3 是二元组 (降维通道, 输出通道)。

        四条线路见讲义 4.1 节，最后在通道维拼接。
        """


def make_googlenet(in_channels: int = 1, num_classes: int = 10) -> nn.Module:
    """五个模块，见讲义 4.3 节。最后接 Linear(1024, num_classes)。"""
```

| 检查 | 期望 |
|---|---|
| `Inception(64, (96,128), (16,32), 32, 192)` 对 `(1,192,28,28)` | `(1, 256, 28, 28)` |
| 上面那个块的通道拼接顺序 | 前 64、次 128、再 32、末 32 |
| GoogLeNet 前向 | `(1, 1, 224, 224)` → `(1, 10)` |
| GoogLeNet 参数量 | **5,977,530** |
| 有没有全局平均池化 | 有 |

## T5 改错

下面这段想实现一个 Inception 块。**三处**问题。

```python
class Inception(nn.Module):
    def __init__(self, c1, c2, c3, c4, in_channels):
        super().__init__()
        self.p1 = nn.Sequential(nn.Conv2d(in_channels, c1, 1), nn.ReLU())
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], 1), nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], 3), nn.ReLU())
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], 1), nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], 5), nn.ReLU())
        self.p4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(in_channels, c4, 1), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.p1(x), self.p2(x), self.p3(x), self.p4(x)])
```

三处分别指：**3×3 卷积的填充**、**5×5 卷积的填充**、**`torch.cat` 的维度**。

三段提示：前两处的高宽对不上；第三处默认在**批量维**拼接——通道数不同时会报
`Sizes of tensors must match except in dimension 0`，如果四条线的通道数恰好相同，
则不报错、把批量拼成 4 倍。
