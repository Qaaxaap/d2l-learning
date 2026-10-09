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


