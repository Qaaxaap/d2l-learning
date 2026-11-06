"""B16 参考答案。做完再看。

跑一遍：
    python3 solutions/b16/b16.py
或者拿它验断言：
    python3 tools/check_b16.py solutions/b16/b16.py
"""

import torch
import torch.nn as nn


# ---------- T1 搭 LeNet ----------

def make_lenet(in_channels: int = 1, num_classes: int = 10) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, 6, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, num_classes),
    )


# ---------- T2 评估准确率 ----------

def evaluate(net: nn.Module, loader, device) -> float:
    net.eval()
    net.to(device)
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            correct += (net(X).argmax(dim=1) == y).sum().item()
            total += y.numel()
    return correct / total if total else 0.0


# ---------- T3 训练 ----------

def train_lenet(net: nn.Module, train_iter, test_iter, num_epochs: int,
                lr: float, device) -> dict:
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    hist = {"train_loss": [], "train_acc": [], "test_acc": []}
    for _ in range(num_epochs):
        net.train()
        loss_sum = correct = total = 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = net(X)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.numel()
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.numel()

        hist["train_loss"].append(loss_sum / total)
        hist["train_acc"].append(correct / total)
        hist["test_acc"].append(evaluate(net, test_iter, device))
    return hist


"""T4 改错

第 1 处：`transforms.Compose([])`
    空的变换链条，数据保持 PIL 图像对象。DataLoader 的默认 collate 只能拼张量与数字，
    于是报 `TypeError: default_collate: batch must contain tensors, numpy arrays, numbers,
    dicts or lists; found <class 'PIL.Image.Image'>`。
    改法：`transforms.Compose([transforms.ToTensor()])` 或直接 `transforms.ToTensor()`。
    换成 ToTensor 之后还会顺带把像素缩放到 [0,1]，这一步不做会让输入量级太大。

第 2 处：`net` 没有搬到 device
    数据 `.to(device)` 了，模型还在 CPU。第一层卷积会报
    `Expected all tensors to be on the same device`。
    改法：`net.to(device)`。

第 3 处：缺 `optimizer.zero_grad()`
    不报错，但梯度在每个批次上累加，等于用越来越大的梯度更新，训练效果很差。
    改法：在每个批次的 `loss.backward()` 之前调 `optimizer.zero_grad()`。
"""


if __name__ == "__main__":
    net = make_lenet()
    print("T1 参数量:", sum(p.numel() for p in net.parameters()))
    print("T1 前向:", tuple(net(torch.randn(4, 1, 28, 28)).shape))
