"""B18 参考答案。做完再看。

跑一遍：
    python3 solutions/b18/b18.py
或者拿它验断言：
    python3 tools/check_b18.py solutions/b18/b18.py
"""

import torch
import torch.nn as nn


# ---------- T1 手写 BN 前向 ----------

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum, training):
    """返回 (Y, moving_mean, moving_var)。moving 两个张量原地更新。"""
    if not training:
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert X.dim() in (2, 4), f"只支持 2 维或 4 维输入，得到 {X.dim()} 维"
        if X.dim() == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        with torch.no_grad():
            moving_mean.mul_(momentum).add_((1 - momentum) * mean)
            moving_var.mul_(momentum).add_((1 - momentum) * var)
    return gamma * X_hat + beta, moving_mean, moving_var


# ---------- T2 BatchNorm 层 ----------

class BatchNorm(nn.Module):
    def __init__(self, num_features: int, num_dims: int = 4,
                 eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        if num_dims == 2:
            shape = (num_features,)
        elif num_dims == 4:
            shape = (1, num_features, 1, 1)
        else:
            raise ValueError("num_dims 只能是 2 或 4")
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.register_buffer("moving_mean", torch.zeros(shape))
        self.register_buffer("moving_var", torch.ones(shape))
        self.eps = eps
        self.momentum = momentum

    def forward(self, X):
        Y, _, _ = batch_norm(X, self.gamma, self.beta, self.moving_mean,
                             self.moving_var, self.eps, self.momentum,
                             self.training)
        return Y


# ---------- T3 残差块 ----------

class Residual(nn.Module):
    def __init__(self, in_channels: int, num_channels: int,
                 use_1x1conv: bool = False, strides: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3,
                               padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=1,
                                   stride=strides)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = torch.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        return torch.relu(Y + X)


# ---------- T4 DenseNet 的两个块 ----------

def conv_block(num_channels: int, in_channels: int) -> nn.Module:
    """BN → ReLU → 3×3 卷积（填充 1）。"""
    return nn.Sequential(
        nn.BatchNorm2d(in_channels), nn.ReLU(),
        nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1),
    )


class DenseBlock(nn.Module):
    def __init__(self, num_convs: int, num_channels: int, in_channels: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = in_channels
        for _ in range(num_convs):
            self.blocks.append(conv_block(num_channels, ch))
            ch += num_channels

    def forward(self, X):
        for blk in self.blocks:
            Y = blk(X)
            X = torch.cat([X, Y], dim=1)
        return X


def transition_block(num_channels: int, in_channels: int) -> nn.Module:
    """BN → ReLU → 1×1 卷积 → 平均池化 2×2 步幅 2。"""
    return nn.Sequential(
        nn.BatchNorm2d(in_channels), nn.ReLU(),
        nn.Conv2d(in_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2),
    )


"""T5 改错

第 1 处：卷积分支的 `mean` 少了 `keepdim=True`
    `X.mean(dim=(0,2,3))` 得到 `(C,)`，从 `(N,C,H,W)` 里减会被广播成
    `(N,C,H,W) - (C,)` —— 第 0 维对不上，报形状错误（或静默算错）。
    改法：`X.mean(dim=(0,2,3), keepdim=True)`。

第 2 处：`moving_mean` 与 `moving_var` 用普通张量保存
    不是 buffer，就不会进 `state_dict`，保存模型后读回来统计量丢失。
    改法：`self.register_buffer("moving_mean", ...)`。

第 3 处：`self.training` 没有传给 `batch_norm`
    训练与预测用同一套统计量，预测结果会依赖同批里的其他样本。
    改法：把 `self.training` 传进去。
"""


if __name__ == "__main__":
    X = torch.randn(4, 3, 8, 8)
    bn = BatchNorm(3, 4)
    print("BN 前向:", tuple(bn(X).shape), " 输出均值:", round(bn(X).mean().item(), 6))
    r = Residual(3, 6, use_1x1conv=True, strides=2)
    print("Residual:", tuple(r(X).shape))
    d = DenseBlock(2, 10, in_channels=3)
    print("DenseBlock:", tuple(d(X).shape), "（期望 (4, 23, 8, 8)）")
    t = transition_block(10, 23)
    print("transition:", tuple(t(d(X)).shape))
