import torch
def tensor_info(x: torch.Tensor) -> dict:
    """返回关于 x 的一份描述，键固定为：
    shape           -> tuple，例如 (3, 4)
    numel           -> int，元素总数
    dtype           -> str，例如 'float32'
    device          -> str，例如 'cpu' 或 'cuda:0'
    is_contiguous   -> bool
    """
