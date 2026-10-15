"""B10 参考答案。做完再看。

跑一遍：
    python3 solutions/b10/b10.py
或者拿它验断言：
    python3 tools/check_b10.py solutions/b10/b10.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


# ---------- T1 高维数据 ----------

def make_high_dim(n_train: int = 20, n_test: int = 100,
                  num_inputs: int = 200, seed: int = 0) -> tuple:
    """y = 0.05 + 0.01 * sum(x_i) + 噪声，噪声标准差 0.01。"""
    g = torch.Generator().manual_seed(seed)
    n = n_train + n_test
    x = torch.randn(n, num_inputs, generator=g)
    noise = torch.randn(n, 1, generator=g) * 0.01
    y = 0.05 + 0.01 * x.sum(dim=1, keepdim=True) + noise
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


# ---------- T2 惩罚项 ----------

def l2_penalty(w: torch.Tensor, lam: float) -> torch.Tensor:
    return lam / 2 * (w**2).sum()


# ---------- T3 带惩罚项的训练 ----------

def _mean_loss(loader, w, b) -> float:
    """遍历 loader 算平均平方损失，不含惩罚项。"""
    total, n = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            total += ((X @ w + b - y) ** 2).sum().item()
            n += y.numel()
    return total / n


def train_high_dim(train_iter, test_iter, lam: float = 0.0,
                   num_epochs: int = 3, lr: float = 0.003,
                   seed: int = 0) -> dict:
    torch.manual_seed(seed)
    num_inputs = train_iter.dataset.tensors[0].shape[1]
    w = (torch.randn(num_inputs, 1) * 0.01).requires_grad_(True)
    b = torch.zeros(1, requires_grad=True)

    train_loss, test_loss = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            # 训练损失 = 这一批的平方误差之和 + lam/2*||w||^2 / 批量大小
            loss = ((X @ w + b - y) ** 2).sum() + l2_penalty(w, lam) / X.shape[0]
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
                w.grad = None
                b.grad = None
        # 记录的是不含惩罚项的平均损失
        train_loss.append(_mean_loss(train_iter, w, b))
        test_loss.append(_mean_loss(test_iter, w, b))
    return {"train_loss": train_loss, "test_loss": test_loss}


# ---------- T4 丢弃函数 ----------

def dropout_layer(X: torch.Tensor, dropout: float) -> torch.Tensor:
    if dropout == 0:
        return X
    if dropout == 1:
        return torch.zeros_like(X)
    # 保留概率 1-dropout 的位置置 1，其余置 0，再按 1/(1-dropout) 缩放
    mask = (torch.rand(X.shape) > dropout).to(X.dtype)
    return mask * X / (1.0 - dropout)


# ---------- T5 带丢弃层的 MLP ----------

def make_dropout_net(num_inputs: int = 784, num_hiddens1: int = 256,
                     num_hiddens2: int = 256, num_outputs: int = 10,
                     dropout1: float = 0.2, dropout2: float = 0.5,
                     seed: int = 0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, num_hiddens1), nn.ReLU(), nn.Dropout(dropout1),
        nn.Linear(num_hiddens1, num_hiddens2), nn.ReLU(), nn.Dropout(dropout2),
        nn.Linear(num_hiddens2, num_outputs),
    )


def evaluate(net, loader) -> float:
    """评估时切到推理模式，评估完恢复原来所处的模式。"""
    was_training = net.training
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            correct += (net(X).argmax(dim=1) == y).sum().item()
            total += y.numel()
    if was_training:
        net.train()
    return correct / total


def train_dropout(train_iter, test_iter, use_dropout: bool = True,
                  num_epochs: int = 5, lr: float = 0.5,
                  batch_size: int = 256, seed: int = 0) -> dict:
    net = make_dropout_net(
        dropout1=0.2 if use_dropout else 0.0,
        dropout2=0.5 if use_dropout else 0.0,
        seed=seed,
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    train_loss, train_acc, test_acc = [], [], []
    for _ in range(num_epochs):
        total, n = 0.0, 0
        for X, y in train_iter:
            optimizer.zero_grad()
            loss = loss_fn(net(X), y)
            loss.backward()
            optimizer.step()
            # CrossEntropyLoss 返回的是这一批的平均，乘回样本数再统一除
            total += loss.item() * y.numel()
            n += y.numel()
        train_loss.append(total / n)
        train_acc.append(evaluate(net, train_iter))
        test_acc.append(evaluate(net, test_iter))
    return {"train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc}


"""T6 改错

第 1 处：`nn.Dropout(0.2)` 写在 `nn.ReLU()` 之前
    丢弃应当作用在激活之后的输出上。写在前面会先把一部分线性输出清零，
    再经过 ReLU——被清零的位置 ReLU 后还是 0，看起来"也丢弃了"，
    但概率与语义都变了：ReLU 会把负值也变成 0，被丢弃的元素于是混在
    "本来就为 0" 的元素里，`1/(1-p)` 的缩放也对不上。
    改法：调换两行，先 `nn.ReLU()` 再 `nn.Dropout(0.2)`。

第 2 处：`test_acc` 没有切到推理模式
    它在训练循环里被调用，此时 `net` 处于训练模式（`nn.Module` 的默认），
    丢弃层仍然在随机清零。于是报出来的测试精度偏低，而且每次跑都不一样。
    改法：进函数先 `net.eval()`，算完按原来的状态恢复（`net.train()`）。

第 3 处：优化器上带了 `weight_decay=0.01`
    这个实验要比较"有丢弃"与"无丢弃"，权重衰减是另一种正则化手段，
    两者混在一起就分不清精度变化是谁带来的。
    改法：去掉 `weight_decay`，让它保持默认的 0。
"""


if __name__ == "__main__":
    # 权重衰减实验
    x_tr, y_tr, x_te, y_te = make_high_dim()
    g = torch.Generator().manual_seed(0)
    tr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=5, shuffle=True, generator=g)
    te = DataLoader(TensorDataset(x_te, y_te), batch_size=100)
    for lam in (0.0, 3.0):
        res = train_high_dim(tr, te, lam=lam, num_epochs=3)
        print(f"lam={lam}: 训练 {res['train_loss'][-1]:.5f}  测试 {res['test_loss'][-1]:.5f}")

    # 丢弃函数
    print("dropout=0 原样返回：", torch.allclose(dropout_layer(torch.ones(4), 0.0), torch.ones(4)))
    print("dropout=1 全零：    ", dropout_layer(torch.ones(4), 1.0).tolist())

    # 丢弃法实验
    tf = transforms.ToTensor()
    ftr = datasets.FashionMNIST(root="data", train=True, download=False, transform=tf)
    fte = datasets.FashionMNIST(root="data", train=False, download=False, transform=tf)
    g2 = torch.Generator().manual_seed(0)
    fitr = DataLoader(ftr, batch_size=256, shuffle=True, generator=g2)
    fite = DataLoader(fte, batch_size=256)
    for use in (True, False):
        res = train_dropout(fitr, fite, use_dropout=use, num_epochs=5)
        print(f"丢弃={use}: 测试精度 {[round(v, 4) for v in res['test_acc']]}")
