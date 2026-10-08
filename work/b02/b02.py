"""
import torch

# 造一批数据，准备拿去做浮点运算
x = torch.arange(12) dtype错误，加个dtype=torch.float32

# 变形成 3 行 4 列
X = x.reshape(3, 4)

# 打印它的形状
print(X.size) size是方法不是属性，需要括号。

# 取第一行改一改，认为 X 不会受影响
row = X[0]
row[0] = 100.0 这个切片是视图不是副本，不想受影响在上一行加.clone()
print(X[0, 0]) 

"""
import torch
import numpy as np
import pandas as pd

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
        if ((ia == 1) or (ib == 1)):
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
    """读取 CSV，返回 (特征张量, 特征名列表, 列统计信息)。"""
    data = pd.read_csv(path)
    num_cols = [
        col for col in data.columns
        if pd.api.types.is_numeric_dtype(data[col])
    ]
    cat_cols = [
        col for col in data.columns
        if col not in num_cols
    ]
    d = {
        "数值列": num_cols.copy(), 
        "类别列": {}
    }
    for col in cat_cols:
        d["类别列"][col] = data[col].value_counts().to_dict()
    for col in num_cols:
        data[col] = data[col].fillna(data[col].mean())
    if cat_cols:
        data = pd.get_dummies(data,columns=cat_cols,dummy_na=False)
    x = torch.Tensor(data.to_numpy(dtype=np.float32))
    name = list(data.columns)
    return (x, name, d)