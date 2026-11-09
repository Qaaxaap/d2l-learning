"""
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
) net还在cpu上，要to(device)

device = torch.device("cuda")
transform = transforms.Compose([]) 没有转成张量 需要ToTensor
train_set = datasets.FashionMNIST("data", True, transform=transform)
train_iter = DataLoader(train_set, batch_size=256, shuffle=True)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    for X, y in train_iter: 没有清理梯度，要zero_grad
        X, y = X.to(device), y.to(device)
        loss = loss_fn(net(X), y)
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch + 1}, loss {loss.item():.4f}")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
    net = nn.Sequential(
        nn.Conv2d(in_channels, 6, (5, 5)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), 2),
        nn.Conv2d(6, 16, (5, 5)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), 2),
        nn.Flatten(),
        nn.Linear(256, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, num_classes)
    )
    return net

def evaluate(net: nn.Module, loader:DataLoader, device) -> float:
    """返回模型在 loader 上的分类准确率，范围 [0, 1]。不改变 net 的参数。"""
    acc, sum = 0, 1e-15
    net = net.to(device)
    net.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        sum += x.shape[0]
        acc += (net(x).argmax(dim=1) == y).sum().item()
    return acc / sum

def train_lenet(net: nn.Module, train_iter, test_iter, num_epochs: int,
                lr: float, device) -> dict:
    """训练 net，每个 epoch 记一次指标。

    返回 {"train_loss": [...], "train_acc": [...], "test_acc": [...]}
    三个列表的长度都等于 num_epochs。
    """
    train_loss, train_acc, test_acc = [], [], []
    loss_fn = nn.CrossEntropyLoss()
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr)
    for _ in range(num_epochs):
        total_size, total_loss = 0, 0
        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            optimizer.step()
            total_size += x.shape[0]
            total_loss += loss.item() * x.shape[0]
        with torch.no_grad():
            train_loss.append(total_loss/total_size)
            train_acc.append(evaluate(net, train_iter, device))
            test_acc.append(evaluate(net, test_iter, device))
        net.train()
    return {
        "train_loss" : train_loss,
        "train_acc" : train_acc,
        "test_acc" : test_acc 
    }
        
             