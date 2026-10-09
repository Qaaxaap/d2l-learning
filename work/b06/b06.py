import torch
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
    