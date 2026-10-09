"""
import torch
import torch.nn as nn

w = torch.randn(2, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr, batch_size = 0.5, 10
net = lambda X: X @ w + b
loss_fn = lambda y_hat, y: (y_hat - y) ** 2

for epoch in range(3):
    for X, y in batches:
        l = loss_fn(net(X), y) 没有清空上次的梯度，会叠加
        l.backward() 这里l是张量，.sum()再反向传播
        w = w - lr * w.grad / batch_size 被计入了计算图，应当 no_grad 包裹
        b = b - lr * b.grad / batch_size
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def make_data(n: int, w: torch.Tensor, b: float, noise: float = 0.01,
              seed: int = 0) -> tuple:
    """生成线性回归的人造数据。

    - X 形状 (n, len(w))，每个元素从标准正态分布抽
    - 真实关系是 y = X @ w + b，再加一个噪声项
    - 噪声从均值 0、标准差 noise 的正态分布抽，形状与 y 相同
    - 同一个 seed 必须给出完全相同的结果
    - y 的形状是 (n, 1)

    返回 (X, y)，都是 float32 张量。
    """
    torch.manual_seed(seed)
    w = w.reshape(-1, 1)
    X = torch.randn(n, len(w), dtype=torch.float32)
    y = X @ w + b + torch.randn(n, 1, dtype=torch.float32) * noise
    return (X, y)

def data_iter(batch_size: int, features: torch.Tensor, labels: torch.Tensor,
              seed: int = 0):
    """按批量大小随机切分数据，逐批产出 (X, y)。

    要求：
    - 是生成器（用 yield），不是返回一个列表
    - 每轮把样本顺序打乱后再切分
    - 最后一个批量可以不满（样本数不被批量大小整除时）
    - 同一个 seed 给出相同的切分顺序
    """
    torch.manual_seed(seed)
    n = features.size(0)
    perm = torch.randperm(n)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield features[idx], labels[idx]
        
def linreg(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """线性回归的预测：X @ w + b。"""
    w = w.reshape(-1, 1);
    return X @ w + b

def squared_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """平方损失：(y_hat - y)² / 2，逐元素，不取平均。"""
    y = y.reshape(y_hat.shape)
    return (y_hat - y) ** 2 / 2

def sgd(params: list, lr: float, batch_size: int) -> None:
    """小批量随机梯度下降。对 params 里的每个参数做：

        p ← p - lr * p.grad / batch_size

    要求：
    - 就地更新，不要返回新张量
    - 更新后该参数的 .grad 要清零
    - 更新过程中不许产生新的计算图
    """
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad / batch_size
            p.grad = None

def train_scratch(X: torch.Tensor, y: torch.Tensor, lr: float = 0.03,
                  num_epochs: int = 3, batch_size: int = 10,
                  seed: int = 0) -> tuple:
    """用手写的零件训练线性回归。

    用 T2 的 data_iter 取批、T3 的 linreg 与 squared_loss 算损失、T4 的 sgd 更新参数。
    参数自己初始化：w 用均值 0、标准差 0.01 的正态随机数，形状 (d, 1)；b 用全零，形状 (1,)。
    两者都要开 requires_grad。

    返回 (w, b, losses)：
    - w、b 是训练完的参数
    - losses 是每个 epoch 结束时的训练损失，Python 浮点数组成的列表，长度 num_epochs
    """
    w = torch.normal(0., 0.01, size=(X.numel() // X.size(0), 1), requires_grad=True)
    b = torch.zeros(1 , requires_grad=True)
    losses = []
    for i in range(num_epochs):
        for Xi, yi in data_iter(batch_size, X, y, seed):
            y_hat = linreg(Xi, w, b)
            loss = squared_loss(y_hat, yi).sum()
            loss.backward()
            sgd([w, b], lr, batch_size)
        y_hat = linreg(X, w, b)
        loss = squared_loss(y_hat, y).mean()
        losses.append(loss)
    return w, b, losses

def train_concise(X: torch.Tensor, y: torch.Tensor, lr: float = 0.03,
                  num_epochs: int = 3, batch_size: int = 10,
                  seed: int = 0) -> tuple:
    """用框架的封装重写一遍，返回 (net, losses)。

    - net 是 nn.Sequential(nn.Linear(d, 1)) 这样的模块
    - 数据用 TensorDataset 包起来，交给 DataLoader 分批
    - 损失用 nn.MSELoss()
    - 优化器用 torch.optim.SGD
    """
    torch.manual_seed(seed)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    net = nn.Sequential(nn.Linear(X.numel() // X.size(0), 1))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
    loss = nn.MSELoss()
    losses = []
    for epoch in range(num_epochs):
        for Xi, yi in loader:
            l = loss(net(Xi), yi)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        losses.append(loss(net(X), y))
    return net, losses


