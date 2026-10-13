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
    t, sum = 0, 0
    with torch.no_grad():
        for X, y in loader:
           ya = net(X) 
           t += int((ya.argmax(dim=1) == y).sum())
           sum += ya.size(0)
    return t / sum
           

