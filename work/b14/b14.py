"""
import torch
import torch.nn as nn

class Scale(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.scale = torch.ones(n) 没有Parameters包裹

    def forward(self, x):
        return x * self.scale

net = nn.Sequential(nn.Linear(4, 4), Scale(4))
torch.save(net, "model.pt") 保存了全部网络，通常保存state_dict

X = torch.randn(2, 4)
net2 = nn.Sequential(nn.Linear(4, 4), Scale(4))
net2.load_state_dict(torch.load("model.pt")) 保存的不是state_dict所以读出来也不是，对不上
print(net2(X))
"""
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
    
def save_params(net: nn.Module, path: str) -> list[str]:
    """把 net 的 state_dict 存到 path，返回键名列表（按 state_dict 给的顺序）。"""
    torch.save(net.state_dict(), path)
    return list(net.state_dict().keys())

def load_params(net: nn.Module, path: str) -> None:
    """从 path 读参数，装进 net（原地改 net 的参数）。"""
    net.load_state_dict(torch.load(path))

def to_device(x: torch.Tensor, device) -> torch.Tensor:
    """把张量搬到 device。设备相同时返回同一个对象（不是副本）。"""
    return x.to(device)

def move_net(net: nn.Module, device) -> nn.Module:
    """把模型的所有参数搬到 device，返回搬完之后的模型。

    返回的对象应当**就是传进来的那个模型**（nn.Module.to() 是原地操作）。
    """
    return net.to(device)