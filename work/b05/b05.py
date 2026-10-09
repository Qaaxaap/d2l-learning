"""
import torch
import torch

p_a = 0.3
p_b = 0.4

# 下面这行想算出 P(A 且 B)，也就是联合概率
p_a_and_b = p_a + p_b

counts = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
probs = counts

values = torch.tensor([1.0, 2.0, 3.0])
weights = torch.tensor([0.2, 0.3, 0.5])
E = values.mean()


"""

import torch
from torch.distributions import multinomial

def roll_and_estimate(n_rolls: int, seed: int = 0) -> torch.Tensor:
    """掷 n_rolls 次六面公平骰子，返回每个面出现的频率。

    返回值形状 (6,)，六个元素之和为 1（允许浮点误差）。
    同一个 seed 必须给出完全相同的结果。
    """
    torch.manual_seed(seed)
    m = multinomial.Multinomial(n_rolls, torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32)).sample()
    return m / n_rolls

def mean_var(values: torch.Tensor, probs: torch.Tensor) -> tuple:
    """给定取值和对应概率，返回 (期望, 方差)。

    两个参数形状都是 (n,)，probs 之和为 1。
    方差用 E[X²] - E[X]² 计算，不许用 torch.var（它默认按无偏估计，语义不同）。
    返回两个 0 维张量。
    """
    x = values * probs
    v = torch.tensor(x.sum())
    e = torch.tensor((values ** 2 * probs).sum() - v.item() ** 2)
    return (v, e)

def posterior(p_d_given_h1: float, p_d_given_h0: float, p_h1: float) -> float:
    """给定 P(D=1|H=1)、P(D=1|H=0)、P(H=1)，返回 P(H=1|D=1)。

    返回 Python 浮点数。
    """
    return p_h1 * p_d_given_h1 / (p_d_given_h1 * p_h1 + p_d_given_h0 * (1 - p_h1))

def cumulative_means(n_max: int, seed: int = 0) -> torch.Tensor:
    """掷 n_max 次六面骰子，返回前 k 次的平均值构成的序列。

    返回值形状 (n_max,)，第 k 个元素是前 k+1 次的平均（k 从 0 开始）。
    """
    torch.manual_seed(seed)
    m = multinomial.Multinomial(1, torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32)).sample((n_max,)).argmax(dim=1) + 1
    m = m.cumsum(dim=0) / (torch.arange(n_max) + 1)
    return m