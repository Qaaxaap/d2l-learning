"""B06 参考答案。做完再看。

跑一遍：
    python3 solutions/b06/b06.py
或者拿它验断言：
    python3 tools/check_b06.py solutions/b06/b06.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_data(n: int, w: torch.Tensor, b: float, noise: float = 0.01,
              seed: int = 0) -> tuple:
    """X 从标准正态抽，y = Xw + b + 高斯噪声。"""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, w.numel(), generator=g)
    y = X @ w.reshape(-1, 1) + b
    y = y + torch.randn(n, 1, generator=g) * noise
    return X, y


def data_iter(batch_size: int, features: torch.Tensor, labels: torch.Tensor,
              seed: int = 0):
    """打乱下标后按批量逐个 yield。最后一个批量可以不满。"""
    n = features.shape[0]
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=g)
    for i in range(0, n, batch_size):
        j = indices[i : i + batch_size]
        yield features[j], labels[j]


def linreg(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return X @ w + b


def squared_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """逐元素平方损失，除以 2，不取平均。"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params: list, lr: float, batch_size: int) -> None:
    """小批量随机梯度下降。更新完顺手清零，为下一轮做准备。"""
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad / batch_size
            p.grad.zero_()


def train_scratch(X: torch.Tensor, y: torch.Tensor, lr: float = 0.03,
                  num_epochs: int = 3, batch_size: int = 10,
                  seed: int = 0) -> tuple:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(X.shape[1], 1, generator=g) * 0.01
    w.requires_grad_(True)
    b = torch.zeros(1, requires_grad=True)

    losses = []
    for epoch in range(num_epochs):
        for bx, by in data_iter(batch_size, X, y, seed=seed + epoch):
            l = squared_loss(linreg(bx, w, b), by)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = squared_loss(linreg(X, w, b), y)
        losses.append(train_l.mean().item())
    return w, b, losses


def train_concise(X: torch.Tensor, y: torch.Tensor, lr: float = 0.03,
                  num_epochs: int = 3, batch_size: int = 10,
                  seed: int = 0) -> tuple:
    torch.manual_seed(seed)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net = nn.Sequential(nn.Linear(X.shape[1], 1))
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    losses = []
    for epoch in range(num_epochs):
        for bx, by in loader:
            l = loss(net(bx), by)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        with torch.no_grad():
            losses.append(loss(net(X), y).item())
    return net, losses


"""T7 改错

第 1 处：`l.backward()`
    `l` 是这一批所有样本的损失，形状 (10, 1)，不是标量。torch 对非标量输出要求
    显式提供梯度权重，直接调会报
    `RuntimeError: grad can be implicitly created only for scalar outputs`。
    书上写的是 `l.backward()`，因为 MXNet 会隐式先求和；torch 里要写成 `l.sum().backward()`。

第 2 处：`w = w - lr * w.grad / batch_size`
    这是重新绑定名字，不是就地更新。新张量不是叶子，`requires_grad` 也不带，
    下一轮 `w.grad` 会是 None 或者直接报 `does not require grad`。
    而且它有梯度的话，会顺带建出一张越来越长的计算图。
    正确写法是在 `no_grad` 下原地改：
        with torch.no_grad():
            w -= lr * w.grad / batch_size
    `b` 那一行同样的问题。

第 3 处：整个循环没有清零梯度
    torch 的 `.grad` 是累加的，不清零的话每轮都会把上一轮的梯度加进来，
    等效学习率一轮比一轮大。症状是损失震荡或发散，但不报错。
    清零要放在这一轮的 `backward()` 之前，或者在更新参数时顺手清。
"""
