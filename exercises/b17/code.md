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


def make_vgg(conv_arch: list[tuple[int, int]], num_classes: int = 10,
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
    """输入 (N, in_channels, 224, 224)。依次是：

    nin_block(96, 11, 4, 0, in_channels)   224 → 54
    MaxPool 3×3，步幅 2                     54 → 26
    nin_block(256, 5, 1, 2)                 26 → 26
    MaxPool 3×3，步幅 2                     26 → 12
    nin_block(384, 3, 1, 1)                 12 → 12
    MaxPool 3×3，步幅 2                     12 → 5
    Dropout(0.5)
    nin_block(num_classes, 3, 1, 1)         5 → 5
    AdaptiveAvgPool2d(1) → Flatten          5 → 1
    """


# 第一个 nin_block 的输入通道不是它自己的输出通道，要显式传 in_channels。
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
    def __init__(self, c1: int, c2: tuple[int, int], c3: tuple[int, int],
                 c4: int, in_channels: int) -> None:
        """in_channels 是输入通道数，四条并行线路都吃同一个输入：

        线路 1：1×1 卷积，输出 c1 个通道
        线路 2：1×1 卷积（输出 c2[0]）→ ReLU → 3×3 卷积（填充 1，输出 c2[1]）
        线路 3：1×1 卷积（输出 c3[0]）→ ReLU → 5×5 卷积（填充 2，输出 c3[1]）
        线路 4：3×3 最大池化（步幅 1，填充 1）→ 1×1 卷积（输出 c4）

        每条线路的最后一层之后过 ReLU，再拼接。

        先拼接、再对整体过一次 ReLU 与上面等价（ReLU 是逐元素运算，拼接只改变排列），
        两种写法都收。
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x 形状 (batch, in_channels, H, W)。

        返回四条线路在通道维拼接后的结果，形状
        (batch, c1 + c2[1] + c3[1] + c4, H, W)。
        """


def make_googlenet(in_channels: int = 1, num_classes: int = 10) -> nn.Module:
    """输入 (N, in_channels, 224, 224)。五个模块依次是：

    b1：Conv 7×7（64，步幅 2，填充 3）→ ReLU
        → MaxPool 3×3（步幅 2，填充 1）

    b2：Conv 1×1（64）→ ReLU
        → Conv 3×3（192，填充 1）→ ReLU
        → MaxPool 3×3（步幅 2，填充 1）

    b3：Inception(64,  (96, 128),  (16, 32), 32, 192)
        Inception(128, (128, 192), (32, 96), 64, 256)
        → MaxPool 3×3（步幅 2，填充 1）

    b4：Inception(192, (96, 208),  (16, 48), 64, 480)
        Inception(160, (112, 224), (24, 64), 64, 512)
        Inception(128, (128, 256), (24, 64), 64, 512)
        Inception(112, (144, 288), (32, 64), 64, 512)
        Inception(256, (160, 320), (32, 128), 128, 528)
        → MaxPool 3×3（步幅 2，填充 1）

    b5：Inception(256, (160, 320), (32, 128), 128, 832)
        Inception(384, (192, 384), (48, 128), 128, 832)
        → AdaptiveAvgPool2d(1) → Flatten

    注意 ReLU 的位置：b1、b2 里每个卷积后面各接一个；Inception 内部四条线路的卷积后
    各接一个（见 Inception 的说明）。模块之间的池化后不接 ReLU，
    全局平均池化 → Flatten → Linear 之间也没有 ReLU。

    最后接 Linear(1024, num_classes)。
    """
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
