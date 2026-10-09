"""B03 参考答案。做完再看。

跑一遍：
    python3 solutions/b03/b03.py
或者拿它验断言：
    python3 tools/check_b03.py solutions/b03/b03.py
"""

import torch


def reduce_shape(shape: tuple, dim: int, keepdim: bool) -> tuple:
    """沿 dim 求和之后的形状。dim 为负时先换算成正的下标。"""
    ndim = len(shape)
    if dim < 0:
        dim += ndim
    if keepdim:
        return shape[:dim] + (1,) + shape[dim + 1 :]
    return shape[:dim] + shape[dim + 1 :]


def matmul_kind(a: torch.Tensor, b: torch.Tensor) -> str:
    """按"能用的最专用函数"分类。"""
    if a.dim() == 1 and b.dim() == 1:
        return "dot"
    if a.dim() == 2 and b.dim() == 1:
        return "mv"
    if a.dim() == 2 and b.dim() == 2:
        return "mm"
    return "matmul"


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """平方和再开根号。先展平，任意形状都成立。"""
    flat = x.reshape(-1)
    return (flat * flat).sum() ** 0.5


def normalize_rows(X: torch.Tensor) -> torch.Tensor:
    """每行除以该行的 L2 范数。零行单独处理，避免除零。"""
    norms = (X * X).sum(dim=1, keepdim=True) ** 0.5
    safe = torch.where(norms == 0, torch.ones_like(norms), norms)
    return X / safe


"""T5 改错

第 1 处：`y = A * x`
    `*` 是按元素乘法。A 形状 (2, 3)、x 形状 (3,)，广播后得到 (2, 3)，
    每一行都逐元素乘了 x，这不是矩阵乘向量。它不报错，结果却是错的。
    改成 `y = A @ x` 或 `y = torch.mv(A, x)`，得到形状 (2,)。

第 2 处：`s = A.sum(dim=1)` 之后的 `z = A / s`
    sum 之后形状是 (2,)，除法右对齐时最后一维 3 与 2 对不上，报
    `The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1`。
    想按行除，统计量必须待在"行"那个位置上，改成 `s = A.sum(dim=1, keepdim=True)`，
    形状 (2, 1)，广播成 (2, 3)。

第 3 处：`C = B.T`
    `.T` 反转所有维度，B 形状 (2, 3, 4) 得到 (4, 3, 2)。
    想只交换最后两维应当写 `C = B.transpose(1, 2)`，得到 (2, 4, 3)。
    二维时"转置"恰好等于交换两维，所以这个直觉在三维上会出错。
"""


if __name__ == "__main__":
    A = torch.arange(6.0).reshape(2, 3)
    print("A =", A)
    print("A @ ones(3) =", A @ torch.ones(3))
    print("A / A.sum(dim=1, keepdim=True) =", A / A.sum(dim=1, keepdim=True))
    B = torch.arange(24.0).reshape(2, 3, 4)
    print("B.T.shape =", tuple(B.T.shape), " B.transpose(1,2).shape =", tuple(B.transpose(1, 2).shape))
    print("normalize_rows =", normalize_rows(torch.tensor([[3.0, 4.0], [0.0, 0.0]])))
