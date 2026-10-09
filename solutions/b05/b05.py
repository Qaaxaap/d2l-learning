"""B05 参考答案。做完再看。

跑一遍：
    python3 solutions/b05/b05.py
或者拿它验断言：
    python3 tools/check_b05.py solutions/b05/b05.py
"""

import torch


def roll_and_estimate(n_rolls: int, seed: int = 0) -> torch.Tensor:
    """用 torch.multinomial 掷骰子，bincount 统计每个面的次数，再归一化。"""
    g = torch.Generator().manual_seed(seed)
    probs = torch.ones(6) / 6
    rolls = torch.multinomial(probs, n_rolls, replacement=True, generator=g)
    counts = torch.bincount(rolls, minlength=6).float()
    return counts / n_rolls


def mean_var(values: torch.Tensor, probs: torch.Tensor) -> tuple:
    """E[X] 与 E[X²] - E[X]²。"""
    mean = (values * probs).sum()
    var = (values * values * probs).sum() - mean * mean
    return mean, var


def posterior(p_d_given_h1: float, p_d_given_h0: float, p_h1: float) -> float:
    """先对 H 边际化算出 P(D=1)，再代贝叶斯公式。"""
    p_h0 = 1.0 - p_h1
    p_d = p_d_given_h1 * p_h1 + p_d_given_h0 * p_h0
    return p_d_given_h1 * p_h1 / p_d


def cumulative_means(n_max: int, seed: int = 0) -> torch.Tensor:
    """累积和除以 1..n，一次算完，不用循环。"""
    g = torch.Generator().manual_seed(seed)
    probs = torch.ones(6) / 6
    rolls = torch.multinomial(probs, n_max, replacement=True, generator=g).float() + 1.0
    return rolls.cumsum(0) / torch.arange(1, n_max + 1, dtype=torch.float32)


"""T5 改错

第 1 处：`PAB = PA + PB`
    PA 与 PB 是"独立"事件的概率，不是互斥事件。独立时联合概率是**相乘**：
    P(A,B) = P(A)P(B) = 0.12。加法法则对应的是互斥事件求并集 P(A∪B) = P(A) + P(B)，
    两者条件相反（互斥时 P(A,B) = 0），混起来用是最常见的概率错误之一。

第 2 处：`probs = counts`
    计数不是概率，六个计数之和是 21，不是 1。要归一化：`probs = counts / counts.sum()`。
    讲义第 1 节的规范性 P(S) = 1 就是这条要求。

第 3 处：`E = values.mean()`
    期望是**按概率加权**的平均，不是算术平均。正确写法是 `E = (values * weights).sum()`，
    得到 0.2×1 + 0.3×2 + 0.5×3 = 2.3，而 `values.mean()` 给的是 2.0。
    只有概率均匀时两者才相等。
"""


if __name__ == "__main__":
    print("骰子 6000 次的频率 =", roll_and_estimate(6000, seed=0))
    print("骰子的期望与方差 =", mean_var(torch.arange(1.0, 7.0), torch.full((6,), 1 / 6)))
    print("HIV 后验 =", posterior(1.0, 0.01, 0.0015))
    print("累积均值最后三个 =", cumulative_means(1000, seed=0)[-3:])
