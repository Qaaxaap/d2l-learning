import torch
def tensor_info(x: torch.Tensor):
    """返回关于 x 的一份描述，键固定为：
    shape           -> tuple，例如 (3, 4)
    numel           -> int，元素总数
    dtype           -> str，例如 'float32'
    device          -> str，例如 'cpu' 或 'cuda:0'
    is_contiguous   -> bool
    """
    d = {}
    d["shape"] = tuple(x.shape)
    d["numel"] = x.numel()
    d["dtype"] = str(x.dtype)[6:]
    d["device"] = str(x.device)
    d["is_contiguous"] = x.is_contiguous()
    return d
def safe_broadcast_add(a: torch.Tensor, b: torch.Tensor):
    