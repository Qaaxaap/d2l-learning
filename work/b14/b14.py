import torch
import torch.nn as nn
import torch.nn.functional as F

class CenteredLayer(nn.Module):
    """把输入的均值减掉。

    前向：返回 x - x.mean()，其中 x.mean() 是**全部元素的均值**（一个标量），
    不是每个样本自己的均值。
    """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x - x.mean()
    
class MyDense(nn.Module):
    """自己的全连接层，最后一个 ReLU。

    构造：MyDense(in_units, units)
    参数：self.weight 形状 (in_units, units)，self.bias 形状 (units,)
          两者都必须是 nn.Parameter
    前向：relu(x @ self.weight + self.bias)
    """
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(in_units, units))
        self.bias = nn.Parameter(torch.zeros(units, ))
    def forward(self, x):
        return F.relu(x @ self.weight + self.bias)
    
def save_and_load(net: nn.Module, path: str) -> dict:
    """把 net 的参数存到 path，再用一个**新建的、结构相同的**模型读回来。

    返回 {"saved_keys": [...], "loaded_ok": bool, "max_diff": float}
    - saved_keys：state_dict 的键名列表（顺序按 state_dict 给的顺序）
    - loaded_ok：load_state_dict 是否成功（不抛异常就是 True）
    - max_diff：新模型参数与旧模型参数的最大绝对差（应该接近 0）
    """
    torch.save(net.state_dict(), path)
    net2 = net
    net2.load_state_dict(torch.load(path))
    return {torch.load(path), True, }