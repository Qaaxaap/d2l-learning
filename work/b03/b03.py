"""
import torch

A = torch.arange(6.0).reshape(2, 3)
x = torch.ones(3)

# 想算 A 乘以向量 x
y = A * x 这是逐元素乘法，而不是矩阵乘向量，换成 @

# 想每一行除以该行元素之和
s = A.sum(dim=1) 列被压掉了，会导致右端对齐错误，用 keepdim=True
z = A / s

# 想交换 B 的最后两维
B = torch.arange(24.0).reshape(2, 3, 4)
C = B.T 三维都翻转了，用 .transpose(1,2)

"""
import torch

def reduce_shape(shape: tuple, dim: int, keepdim: bool) -> tuple:
    """算出「对形状 shape 的张量沿 dim 求和」之后的结果形状。

    不调用 torch，纯推导。
    """
    if (shape == ()):
        return ()
    shape_list = list(shape)
    while (dim < 0):
        dim += len(shape_list)
    dim %= len(shape_list)
    if (keepdim):
        shape_list[dim] = 1
    else:
        shape_list.pop(dim)
    return tuple(shape_list)

def matmul_kind(a: torch.Tensor, b: torch.Tensor) -> str:
    """判断这两个张量相乘时应当用哪个函数，返回最专用的那个。"""
    if (a.dim() < b.dim()):
        a, b = b, a
    if (a.dim() == 1):
        return "dot"
    if (a.dim() == 2 and b.dim() == 1):
        return "mv"
    if (a.dim() == 2 and b.dim() == 2):
        return "mm"
    return "matmul"

def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """用基本运算算出 x 的 L2 范数。

    不许用 torch.linalg.vector_norm、torch.linalg.norm、torch.norm、
    torch.hypot 等任何现成的范数函数。
    """
    return torch.sqrt((x * x).sum())

def normalize_rows(X: torch.Tensor) -> torch.Tensor:
    """把 X 的每一行除以该行的 L2 范数，返回新张量。

    要求：
    - 不修改 X
    - 某行的元素全是 0 时，结果里那一行保持全 0，不许出现 nan
    - 返回的 dtype 与 X 相同
    """
    x = X.clone()
    l2_x = torch.linalg.norm(x, dim=-1, keepdim=True)
    return x / (l2_x + 1e-12)