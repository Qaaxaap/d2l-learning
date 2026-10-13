"""
import torch
import torch.nn as nn

net = nn.Sequential(nn.Linear(784, 10))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

for epoch in range(5):
    for X, y in train_iter: 梯度未清零 会累加
        X = X.reshape(X.shape[0], -1)
        out = torch.softmax(net(X), dim=1)
        l = loss_fn(out, y) 交叉熵内部含有softmax,不需要再做一次，直接用net(X), y跑loss_fn
        optimizer.step()
        l.backward() 求梯度过晚，无法正确更新参数，应互换更新和求梯度两行
"""

from numpy import argmax
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def softmax(o: torch.Tensor) -> torch.Tensor:
    """按最后一维做 softmax，返回同形状的概率。

    要求数值稳定：logits 里有 1000 这样的数时不能出现 nan。
    input 形状 (n, q) 时按第 1 维归一化，每一行之和为 1。
    """
    mx = o.max(dim = -1, keepdim=True).values
    e = torch.exp(o - mx)
    return e / e.sum(dim = -1, keepdim=True)

def cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """交叉熵损失。

    - logits 形状 (n, q)，是未经 softmax 的分数
    - y 形状 (n,)，元素是 0 到 q-1 的整数标签（不是独热）
    - 返回标量：这一批样本的交叉熵**之和**（不取平均）

    不许调用 nn.CrossEntropyLoss、torch.log_softmax 或 torch.logsumexp。
    """
    mx = logits.max(dim = -1, keepdim=True).values
    lse = mx + torch.log(torch.exp(logits - mx).sum(dim=1, keepdim=True))
    n = logits.size(0)
    idx = torch.arange(n)
    lt = logits - lse
    return -(lt[idx, y]).sum()

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """返回命中率，Python 浮点数。

    logits 形状 (n, q)，y 形状 (n,) 是整数标签。
    不需要先算 softmax（想想为什么）。
    """
    return float((logits.argmax(dim=1) == y).sum() / logits.size(0))

def evaluate_accuracy(net, loader) -> float:
    """net 是可调用对象，接受 (n, 784) 返回 (n, 10) 的 logits。

    在整个 loader 上算准确率，返回 Python 浮点数。
    评估过程不应建立计算图。
    """
    t, s = 0, 0
    with torch.no_grad():
        for X, y in loader:
           ya = net(X.reshape(X.size(0), -1)) 
           t += int((ya.argmax(dim=1) == y).sum())
           s += ya.size(0)
    return t / s
           
def make_net(num_inputs: int = 784, num_outputs: int = 10, seed: int = 0):
    """返回 (W, b) 两个叶子张量：W 形状 (784, 10) 用均值 0、标准差 0.01 的正态随机数，
    b 形状 (10,) 用全零。都用 seed 保证可复现，都开 requires_grad。"""
    torch.manual_seed(seed)
    W, b = torch.normal(size=(num_inputs, num_outputs), mean=0, std=0.01, requires_grad=True), torch.zeros((num_outputs,), requires_grad=True)
    return (W, b)

def sgd(params: list, lr: float, batch_size: int) -> None:
    """小批量随机梯度下降，与 B06 的 T4 相同，这一章自己重写一遍。

    对 params 里的每个参数做 p ← p - lr * p.grad / batch_size。
    要求：就地更新、更新后就地清零该参数的梯度、过程中不建计算图。
    注意这里的除法不能省：T1 的 cross_entropy 返回的是和，不是平均。
    """
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad / batch_size
            p.grad = None

def train_from_scratch(train_iter, test_iter, num_epochs: int = 5,
                       lr: float = 0.1, batch_size: int = 256,
                       seed: int = 0) -> dict:
    torch.manual_seed(seed)
    """用手写的 softmax、交叉熵与 sgd 训练。

    返回一个字典，键固定为：
        train_loss -> 每个 epoch 的平均训练损失，浮点列表
        train_acc  -> 每个 epoch 结束时的训练集准确率，浮点列表
        test_acc   -> 每个 epoch 结束时的测试集准确率，浮点列表
    三个列表长度都是 num_epochs。
    """
    W, b = make_net(seed=seed)
    d = {}
    d["train_loss"], d["train_acc"], d["test_acc"] = [], [], []
    for _ in range(num_epochs):
        loss, sum = 0, 0
        for X, y in train_iter:
            X = X.reshape(X.size(0), -1)
            ya = X @ W + b
            #l = accuracy(ya, y)
            l = cross_entropy(ya, y)
            l.backward()
            sgd([W, b], lr, ya.size(0))
            with torch.no_grad():
                loss += l
                sum += X.size(0)
        net = lambda X: X @ W + b
        d["train_loss"].append(float(loss / sum))
        d["train_acc"].append(evaluate_accuracy(net, train_iter))
        d["test_acc"].append(evaluate_accuracy(net, test_iter))
    return d

def confusion_matrix(net, loader, q: int = 10) -> torch.Tensor:
    """返回 (q, q) 的整数张量。

    第 i 行第 j 列 = 真实类别是 i 而被预测成 j 的样本数。
    对角线上是预测正确的数量，整张表所有元素之和等于样本总数。
    """
    ret = torch.zeros(size=(q, q), dtype=torch.int64)
    for X, y in loader:
        ya = net(X.reshape(X.size(0), -1)).argmax(dim=-1)
        for t, p in zip(y, ya):
            ret[t, p] += 1
    return ret

def train_concise(train_iter, test_iter, num_epochs: int = 5,
                  lr: float = 0.1, seed: int = 0) -> dict:
    """用 nn.Linear、nn.CrossEntropyLoss、torch.optim.SGD 重写一遍。

    返回的字典键与 T4 相同。
    """
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(784, 10))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    d = {}
    d["train_loss"], d["train_acc"], d["test_acc"] = [], [], [] 
    for _ in range(num_epochs):
        total_loss, total_size = 0, 0
        for X, y in train_iter:
            optimizer.zero_grad()
            X = X.reshape(X.size(0), -1)
            l = loss_fn(net(X), y)
            l.backward()
            optimizer.step()
            total_loss += l
            total_size += X.size(0)
        d["train_loss"].append(float(total_loss / total_size))
        d["train_acc"].append(evaluate_accuracy(net, train_iter)) 
        d["test_acc"].append(evaluate_accuracy(net, test_iter))
    return d
