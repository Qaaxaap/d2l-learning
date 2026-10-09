import torch
from torch._dynamo.variables import nn_module
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