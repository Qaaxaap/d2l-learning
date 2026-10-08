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
    da = tensor_info(a)
    db = tensor_info(b)
    sp_a = list(da["shape"])
    sp_b = list(db["shape"])
    if (len(sp_a) < len(sp_b)):
        sp_a, sp_b = sp_b, sp_a
    for i in range(len(sp_a) - len(sp_b)):
        sp_b.insert(0,1)
    flag = True
    first = -1
    for ia, ib in zip(sp_a, sp_b):
        if (flag == False):
            break
        first += 1
        if (ia == ib):
            continue
        if ((ia <= 1) or (ib <= 1)):
            continue
        flag = False
    if (flag):
        return a + b;
    raise ValueError(f"第{first}位不可广播")
def slice_is_view() -> bool:
    """构造一个张量，取它的一个切片，修改切片，返回原张量是否跟着变。"""
    x = torch.arange(12).reshape(3,4)
    cx = x.clone()
    y = x[1:3]
    y[0, 0] = 0
    return int((cx == x).sum()) != x.numel()

def clone_is_copy() -> bool:
    """同上，但中间做一次 clone，返回原张量是否跟着变。"""
    x = torch.arange(12).reshape(3,4)
    cx = x.clone()
    y = x[1:3].clone()
    y[0, 0] = 0
    return int((cx == x).sum()) != x.numel()

def load_csv(path: str) -> tuple[torch.Tensor, list[str], dict]:
    """读取 CSV，返回 (特征张量, 特征名列表, 列统计信息)。
