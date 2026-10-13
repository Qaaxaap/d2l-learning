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
    p = softmax(logits)
    n = logits.size(0)
    idx = torch.arange(n)
    return -torch.log(p[idx, y].clamp_min(1e-5)).sum()

