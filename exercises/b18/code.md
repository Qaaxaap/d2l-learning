# B18 代码题

把你的实现写在 `work/b18/b18.py` 里，函数名与签名必须和下面一致。

验收：`just check b18`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章做什么

三件东西：手写批量归一化的前向、残差块、稠密块。

**全部是小张量，不训练。** 断言一秒内跑完。

验收分三类：**精确**（BN 输出的均值方差、形状）、**结构**（有哪些层、哪些 buffer）、
**趋势**（移动统计量在更新、eval 模式行为不同）。

---

## T1 手写 BN 的前向

```python
def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
               moving_mean: torch.Tensor, moving_var: torch.Tensor,
               eps: float, momentum: float, training: bool):
    """手写批量归一化的前向。返回 (Y, moving_mean, moving_var)。

    X: 输入张量，形状 (N, features) 或 (N, C, H, W)。
    gamma: 缩放参数，形状同「要归一的那个维度」，由调用方给定（如全 1）。
    beta: 平移参数，形状同上，由调用方给定（如全 0）。
    moving_mean: 移动平均的均值，形状同上。**原地更新**。
    moving_var: 移动平均的方差，形状同上。**原地更新**。
    eps: 加到方差上的小数，防止除零，如 1e-5。
    momentum: 移动平均的动量，0 到 1 之间，如 0.9。
    training: True 用当前批统计量并更新 moving；
              False 用 moving 统计量、不更新。

    gamma、beta、moving_mean、moving_var 的形状与「要归一的那个维度」一致：
    二维输入时是 (features,)，四维输入时是 (1, C, 1, 1)。
    """
```

| 检查 | 期望 |
|---|---|
| 训练模式、`gamma=1, beta=0` | 输出的每列均值 ≈ 0、方差 ≈ 1 |
| 训练模式、`gamma=2, beta=3` | 均值 ≈ 3、方差 ≈ 4 |
| 四维输入 | 每个通道分别归一化 |
| 训练模式后 | `moving_mean/var` 按 `momentum` 更新 |
| 预测模式 | 用 moving 统计量，且不更新它们 |

**更新公式**：`moving = momentum * moving + (1 - momentum) * 当前批统计量`。

## T2 BatchNorm 层

```python
class BatchNorm(nn.Module):
    def __init__(self, num_features: int, num_dims: int = 4,
                 eps: float = 1e-5, momentum: float = 0.1):
        """num_dims=2 用于全连接（参数形状 (num_features,)），
        num_dims=4 用于卷积（参数形状 (1, num_features, 1, 1)）。"""

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X 形状 (batch, num_features) 或 (batch, num_features, H, W)。

        返回同形状的标准化结果。
        """
```

| 检查 | 期望 |
|---|---|
| 可训练参数 | 只有 `gamma`、`beta` 两个，都是 `nn.Parameter` |
| `state_dict` | 含 `gamma`、`beta`、`moving_mean`、`moving_var` 四个键 |
| 训练模式 | 输出被标准化 |
| `eval()` 之后 | 输出与训练模式不同；单个样本的输出不依赖同批其他样本 |

**提示**：`moving_mean` 与 `moving_var` 要用 `register_buffer`（B13 的 2.3 节）。

## T3 残差块

```python
class Residual(nn.Module):
    def __init__(self, in_channels: int, num_channels: int,
                 use_1x1conv: bool = False, strides: int = 1):
        """Conv3×3 → BN → ReLU → Conv3×3 → BN，加跳跃连接后过 ReLU。

        结构见讲义 2.2 节。use_1x1conv 为 True 时，跳跃路径上一个 1×1 卷积
        （步幅与 strides 一致）。
        """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X 形状 (batch, in_channels, H, W)。

        返回 (batch, num_channels, H // strides, W // strides)。
        """
```

| 检查 | 期望 |
|---|---|
| `Residual(3, 3)` 对 `(4,3,6,6)` | `(4,3,6,6)` |
| `Residual(3,6,use_1x1conv=True,strides=2)` 对 `(4,3,6,6)` | `(4,6,3,3)` |
| 不用 1×1 时 | 2 个卷积 + 2 个 BN |
| 用 1×1 时 | 3 个卷积 |
| 输出值域 | **全部非负**（末尾过 ReLU）|

## T4 DenseNet 的两个块

```python
def conv_block(num_channels: int, in_channels: int) -> nn.Module:
    """BN → ReLU → 3×3 卷积（填充 1）。"""


class DenseBlock(nn.Module):
    def __init__(self, num_convs: int, num_channels: int, in_channels: int):
        """num_convs 个 conv_block，每个的输入是这一路累积的通道数，
        输出后与输入在通道维拼接（见讲义 3.1）。"""

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X 形状 (batch, in_channels, H, W)。

        返回 (batch, in_channels + num_convs * num_channels, H, W)。
        """


def transition_block(num_channels: int, in_channels: int) -> nn.Module:
    """BN → ReLU → 1×1 卷积 → 平均池化 2×2 步幅 2。"""
```

| 检查 | 期望 |
|---|---|
| `DenseBlock(2, 10, in_channels=3)` 对 `(4,3,8,8)` | `(4, 23, 8, 8)` |
| `DenseBlock(4, 32, in_channels=64)` | 通道变 `64 + 4×32 = 192` |
| `transition_block(10, in_channels=23)` 对 `(4,23,8,8)` | `(4,10,4,4)` |
| `transition_block` 里的卷积 | 1×1 |

**注意 `DenseBlock` 里每个 `conv_block` 的输入通道数在变**：第一个吃 `in_channels`，
第二个吃 `in_channels + num_channels`，以此类推。

## T5 改错

下面这段想实现一个 `BatchNorm` 层。**三处**问题。

```python
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        shape = (num_features,) if num_dims == 2 else (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if X.dim() == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3))
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3))
        X_hat = (X - mean) / torch.sqrt(var + 1e-5)
        self.moving_mean = 0.9 * self.moving_mean + 0.1 * mean
        self.moving_var = 0.9 * self.moving_var + 0.1 * var
        return self.gamma * X_hat + self.beta
```

三处分别指：**四维情况下的统计量形状**、**两个 moving 张量的登记方式**、
**训练与预测没有区分**。

三段提示：第一处修好前会报广播错误；第二处不报错，但保存模型时统计量丢失；
第三处不报错，预测结果会依赖同批其他样本。
