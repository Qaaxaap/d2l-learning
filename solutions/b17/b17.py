"""B17 参考答案。做完再看。

跑一遍：
    python3 solutions/b17/b17.py
或者拿它验断言：
    python3 tools/check_b17.py solutions/b17/b17.py
"""

import torch
import torch.nn as nn


# ---------- T1 AlexNet ----------

def make_alexnet(in_channels: int = 1, num_classes: int = 10) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, 96, kernel_size=11, stride=4), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        nn.Dropout(0.5), nn.Linear(256 * 5 * 5, 4096), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
        nn.Linear(4096, num_classes),
    )


# ---------- T2 VGG ----------

def vgg_block(num_convs: int, num_channels: int,
              in_channels: int | None = None) -> nn.Module:
    """num_convs 个 3×3 卷积（填充 1）+ 一个 2×2 池化。

    in_channels 为 None 时按 num_channels 处理（块内各层输入输出相同）。
    """
    first_in = in_channels if in_channels is not None else num_channels
    layers = [nn.Conv2d(first_in, num_channels, kernel_size=3, padding=1), nn.ReLU()]
    for _ in range(num_convs - 1):
        layers += [nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
                   nn.ReLU()]
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def make_vgg(conv_arch, num_classes: int = 10,
             in_channels: int = 1) -> nn.Module:
    """conv_arch 形如 ((1, 64), (1, 128), (2, 256), ...)。输入固定 224×224。"""
    blocks = []
    ch = in_channels
    for num_convs, num_channels in conv_arch:
        blocks.append(vgg_block(num_convs, num_channels, in_channels=ch))
        ch = num_channels
    last = conv_arch[-1][1]
    side = 224 // (2 ** len(conv_arch))
    return nn.Sequential(
        *blocks,
        nn.Flatten(),
        nn.Linear(last * side * side, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, num_classes),
    )


# ---------- T3 NiN ----------

def nin_block(num_channels: int, kernel_size: int, strides: int,
              padding: int, in_channels: int | None = None) -> nn.Module:
    """第一个卷积的输入通道在 in_channels 为 None 时按 num_channels 推断。"""
    first_in = in_channels if in_channels is not None else num_channels
    return nn.Sequential(
        nn.Conv2d(first_in, num_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.Conv2d(num_channels, num_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(num_channels, num_channels, kernel_size=1), nn.ReLU(),
    )


def make_nin(in_channels: int = 1, num_classes: int = 10) -> nn.Module:
    return nn.Sequential(
        nin_block(96, 11, 4, 0, in_channels=in_channels),
        nn.MaxPool2d(3, 2),
        nin_block(256, 5, 1, 2, in_channels=96),
        nn.MaxPool2d(3, 2),
        nin_block(384, 3, 1, 1, in_channels=256),
        nn.MaxPool2d(3, 2), nn.Dropout(0.5),
        nin_block(num_classes, 3, 1, 1, in_channels=384),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )


# ---------- T4 GoogLeNet ----------

class Inception(nn.Module):
    def __init__(self, c1, c2, c3, c4, in_channels: int):
        super().__init__()
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1), nn.ReLU())
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1), nn.ReLU())
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2), nn.ReLU())
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.p1(x), self.p2(x), self.p3(x), self.p4(x)], dim=1)


def make_googlenet(in_channels: int = 1, num_classes: int = 10) -> nn.Module:
    b1 = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
        nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b3 = nn.Sequential(
        Inception(64, (96, 128), (16, 32), 32, 192),
        Inception(128, (128, 192), (32, 96), 64, 256),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b4 = nn.Sequential(
        Inception(192, (96, 208), (16, 48), 64, 480),
        Inception(160, (112, 224), (24, 64), 64, 512),
        Inception(128, (128, 256), (24, 64), 64, 512),
        Inception(112, (144, 288), (32, 64), 64, 512),
        Inception(256, (160, 320), (32, 128), 128, 528),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b5 = nn.Sequential(
        Inception(256, (160, 320), (32, 128), 128, 832),
        Inception(384, (192, 384), (48, 128), 128, 832),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten())
    return nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, num_classes))


if __name__ == "__main__":
    x = torch.randn(1, 1, 224, 224)
    for name, net in [
        ("AlexNet", make_alexnet()),
        ("VGG-11", make_vgg(((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)))),
        ("NiN", make_nin()),
        ("GoogLeNet", make_googlenet()),
    ]:
        out = net(x)
        n = sum(p.numel() for p in net.parameters())
        print(f"{name}: 输出 {tuple(out.shape)}, 参数 {n:,}")
