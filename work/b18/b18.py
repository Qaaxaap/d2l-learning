"""
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        shape = (num_features,) if num_dims == 2 else (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape) 要用register_buffer,这样state_dict里没有
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if X.dim() == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0) 没有keepdim,对不上
        else:
            mean = X.mean(dim=(0, 2, 3))
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3))
        X_hat = (X - mean) / torch.sqrt(var + 1e-5)
        self.moving_mean = 0.9 * self.moving_mean + 0.1 * mean
        self.moving_var = 0.9 * self.moving_var + 0.1 * var
        return self.gamma * X_hat + self.beta 需要区分训练和预测。
"""
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def batch_norm(
    X: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    moving_mean: torch.Tensor,
    moving_var: torch.Tensor,
    eps: float,
    momentum: float,
    training: bool,
):
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
    if training:
        if X.dim() == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        with torch.no_grad():
            moving_mean += momentum * moving_mean + (1 - momentum) * mean - moving_mean
            moving_var += momentum * moving_var + (1 - momentum) * var - moving_var
        X_hat = (X - mean) / torch.sqrt(var + eps)
    else:
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    Y = gamma * X_hat + beta
    return (Y, moving_mean, moving_var)


class BatchNorm(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_dims: int = 4,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        """num_dims=2 用于全连接（参数形状 (num_features,)），
        num_dims=4 用于卷积（参数形状 (1, num_features, 1, 1)）。"""
        super().__init__()
        if num_dims == 2:
            self.gamma = nn.Parameter(torch.ones((num_features,), requires_grad=True))
            self.beta = nn.Parameter(torch.zeros((num_features,), requires_grad=True))
            self.register_buffer("moving_mean", torch.zeros((num_features,)))
            self.register_buffer("moving_var", torch.ones((num_features,)))
        else:
            self.gamma = nn.Parameter(
                torch.ones((1, num_features, 1, 1), requires_grad=True)
            )
            self.beta = nn.Parameter(
                torch.zeros((1, num_features, 1, 1), requires_grad=True)
            )
            self.register_buffer("moving_mean", torch.zeros((1, num_features, 1, 1)))
            self.register_buffer("moving_var", torch.ones((1, num_features, 1, 1)))
        self.eps = eps
        self.momentum = momentum

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X 形状 (batch, num_features) 或 (batch, num_features, H, W)。

        返回同形状的标准化结果。
        """
        train = False
        if self.training:
            train = True
        ret = batch_norm(
            X,
            self.gamma,
            self.beta,
            self.moving_mean,
            self.moving_var,
            self.eps,
            self.momentum,
            train,
        )
        self.moving_mean = ret[1]
        self.moving_var = ret[2]
        return ret[0]


class Residual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        use_1x1conv: bool = False,
        strides: int = 1,
    ):
        """Conv3×3 → BN → ReLU → Conv3×3 → BN，加跳跃连接后过 ReLU。

        结构见讲义 2.2 节。use_1x1conv 为 True 时，跳跃路径上一个 1×1 卷积
        （步幅与 strides 一致）。
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, 3, strides, 1),
            BatchNorm(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, 1, 1),
            BatchNorm(num_channels),
        )
        if use_1x1conv:
            self.net1x1 = nn.Sequential(
                nn.Conv2d(in_channels, num_channels, 1, strides)
            )
        else:
            self.net1x1 = nn.Sequential()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X 形状 (batch, in_channels, H, W)。

        返回 (batch, num_channels, H // strides, W // strides)。
        """
        return F.relu(self.net(X) + self.net1x1(X))


def conv_block(num_channels: int, in_channels: int) -> nn.Sequential:
    """BN → ReLU → 3×3 卷积（填充 1）。"""
    return nn.Sequential(
        BatchNorm(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, num_channels, 3, 1, 1)
    )


class DenseBlock(nn.Module):
    def __init__(self, num_convs: int, num_channels: int, in_channels: int):
        """num_convs 个 conv_block，每个的输入是这一路累积的通道数，
        输出后与输入在通道维拼接（见讲义 3.1）。"""
        super().__init__()
        self.net = nn.Sequential(conv_block(num_channels, in_channels))
        for i in range(num_convs - 1):
            self.net += conv_block(num_channels * (i + 2), num_channels * (i + 1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X 形状 (batch, in_channels, H, W)。

        返回 (batch, in_channels + num_convs * num_channels, H, W)。
        """
        return torch.cat([self.net(X), X], dim=1)


def transition_block(num_channels: int, in_channels: int) -> nn.Module:
    """BN → ReLU → 1×1 卷积 → 平均池化 2×2 步幅 2。"""
    return nn.Sequential(
        BatchNorm(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, num_channels, 1),
        nn.AvgPool2d(2, 2)
    )