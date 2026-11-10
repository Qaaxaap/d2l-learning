"""
class Inception(nn.Module):
    def __init__(self, c1, c2, c3, c4, in_channels):
        super().__init__()
        self.p1 = nn.Sequential(nn.Conv2d(in_channels, c1, 1), nn.ReLU())
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], 1), nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], 3), nn.ReLU()) 需要填充1。
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], 1), nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], 5), nn.ReLU()) 需要填充2。
        self.p4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(in_channels, c4, 1), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.p1(x), self.p2(x), self.p3(x), self.p4(x)])
        不应该在batch维拼接而是在channel维。
"""
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def make_alexnet(in_channels: int = 1, num_classes: int = 10) -> nn.Module:
    """结构见讲义 1.2 节。输入 (N, in_channels, 224, 224)。"""
    net = nn.Sequential(
        nn.Conv2d(in_channels, 96, 11, 4),
        nn.ReLU(),
        nn.MaxPool2d(3, 2),
        nn.Conv2d(96, 256, 5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(3, 2),
        nn.Conv2d(256, 384, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 384, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 256, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(3, 2),
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(6400, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, num_classes),
    )
    return net


def vgg_block(
    num_convs: int, num_channels: int, in_channels: int | None = None
) -> nn.Module:
    """num_convs 个 3×3 卷积（填充 1、ReLU）+ 一个 2×2 步幅 2 的最大池化。

    in_channels 为 None 时，块内所有卷积的输入输出都是 num_channels。
    """
    if in_channels is not None:
        net1 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, 3, padding=1), nn.ReLU()
        )
    else:
        net1 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.ReLU()
        )
    return (
        net1
        + nn.Sequential(
            *[
                copy.deepcopy(
                    nn.Sequential(
                        nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.ReLU()
                    )
                )
                for _ in range(num_convs - 1)
            ]
        )
        + nn.Sequential(nn.MaxPool2d(2, 2))
    )


def make_vgg(conv_arch, num_classes: int = 10, in_channels: int = 1) -> nn.Module:
    """conv_arch 形如 ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))。

    全连接部分：Flatten → Linear(4096) → ReLU → Dropout(0.5)
                → Linear(4096) → ReLU → Dropout(0.5)
                → Linear(num_classes)
    """
    conv_arch = list(conv_arch)
    net = nn.Sequential()
    for num_convs, num_channels in conv_arch:
        net += vgg_block(num_convs, num_channels, in_channels)
        in_channels = num_channels
    net += nn.Sequential(
        nn.Flatten(),
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, num_classes),
    )
    return net


def nin_block(
    num_channels: int,
    kernel_size: int,
    strides: int,
    padding: int,
    in_channels: int | None = None,
) -> nn.Sequential:
    """一个普通卷积（核大小、步幅、填充可配）+ 两个 1×1 卷积，每个后接 ReLU。"""
    return nn.Sequential(
        nn.Conv2d(
            in_channels or num_channels, num_channels, kernel_size, strides, padding
        ),
        nn.ReLU(),
        nn.Conv2d(num_channels, num_channels, 1),
        nn.ReLU(),
        nn.Conv2d(num_channels, num_channels, 1),
        nn.ReLU(),
    )


def make_nin(in_channels: int = 1, num_classes: int = 10) -> nn.Module:
    """结构见讲义 3.4 节。最后用 AdaptiveAvgPool2d(1) + Flatten。"""
    pool = nn.Sequential(nn.MaxPool2d(3, 2))
    return (
        nin_block(96, 11, 4, 0, in_channels)
        + pool
        + nin_block(256, 5, 1, 2, 96)
        + pool
        + nin_block(384, 3, 1, 1, 256)
        + pool
        + nn.Sequential(nn.Dropout(0.5))
        + nin_block(num_classes, 3, 1, 1, 384)
        + nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
    )


class Inception(nn.Module):
    def __init__(self, c1, c2, c3, c4, in_channels: int):
        """c1、c4 是整数；c2、c3 是二元组 (降维通道, 输出通道)。

        四条线路见讲义 4.1 节，最后在通道维拼接。
        """
        super().__init__()
        self.p1 = nn.Sequential(nn.Conv2d(in_channels, c1, 1))
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], 1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], 3, padding=1),
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], 1),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], 5, padding=2),
        )
        self.p4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(in_channels, c4, 1))

    def forward(self, x):
        """x 形状 (batch, in_channels, H, W)。

        返回四条线路在通道维拼接后的结果，形状
        (batch, c1 + c2[1] + c3[1] + c4, H, W)。
        """
        return F.relu(torch.cat([self.p1(x), self.p2(x), self.p3(x), self.p4(x)], dim=1))


def make_googlenet(in_channels: int = 1, num_classes: int = 10) -> nn.Module:
    """五个模块，见讲义 4.3 节。最后接 Linear(1024, num_classes)。"""
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 7, 2, 3),
        nn.ReLU(),
        nn.MaxPool2d(3, 2, 1),
        nn.Conv2d(64, 64, 1),
        nn.ReLU(),
        nn.Conv2d(64, 192, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(3, 2, 1),
        Inception(64, (96, 128), (16, 32), 32, 192),
        Inception(128, (128, 192), (32, 96), 64, 256),
        nn.MaxPool2d(3, 2, 1),
        Inception(192, (96, 208), (16, 48), 64, 480),
        Inception(160, (112, 224), (24, 64), 64, 512),
        Inception(128, (128, 256), (24, 64), 64, 512),
        Inception(112, (144, 288), (32, 64), 64, 512),
        Inception(256, (160, 320), (32, 128), 128, 528),
        nn.MaxPool2d(3, 2, 1),
        Inception(256, (160, 320), (32, 128), 128, 832),
        Inception(384, (192, 384), (48, 128), 128, 832),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(1024, num_classes)
    )
