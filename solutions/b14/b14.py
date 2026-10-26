"""B14 参考答案。做完再看。

跑一遍：
    python3 solutions/b14/b14.py
或者拿它验断言：
    python3 tools/check_b14.py solutions/b14/b14.py
"""

import copy
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path.cwd()


# ---------- T1 不含参数的层 ----------

class CenteredLayer(nn.Module):
    def forward(self, x):
        return x - x.mean()


# ---------- T2 含参数的层 ----------

class MyDense(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        # 必须用 nn.Parameter 包住，否则 net.parameters() 收不到
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units))

    def forward(self, x):
        return nn.functional.relu(x @ self.weight + self.bias)


# ---------- T3 保存与加载 ----------

def make_twin(net: nn.Module) -> nn.Module:
    """深拷贝出同结构的模型，再把参数重新随机化，保证与原来的不同。"""
    twin = copy.deepcopy(net)
    with torch.no_grad():
        for p in twin.parameters():
            p.normal_()
    return twin


def save_and_load(net: nn.Module, path: str) -> dict:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    keys = list(net.state_dict().keys())
    torch.save(net.state_dict(), str(path_obj))

    twin = make_twin(net)
    twin.load_state_dict(torch.load(str(path_obj)))
    twin.eval()

    max_diff = 0.0
    for a, b in zip(net.parameters(), twin.parameters()):
        max_diff = max(max_diff, (a - b).abs().max().item())
    return {"saved_keys": keys, "loaded_ok": True, "max_diff": max_diff}


# ---------- T4 搬到 GPU ----------

def to_device(x: torch.Tensor, device) -> torch.Tensor:
    # .to() 在设备相同时返回 self，不同时返回新张量
    return x.to(device)


def move_net(net: nn.Module, device) -> nn.Module:
    # nn.Module.to() 是原地操作，返回的还是同一个对象
    return net.to(device)


"""T5 改错

第 1 处：`self.scale = torch.ones(n)`
    这是普通张量属性，不会被登记成参数。`net.parameters()` 收不到它，
    优化器不会更新它——训练时它一直保持初始值，而损失照样下降，属于静默出错。
    改法：`self.scale = nn.Parameter(torch.ones(n))`。

第 2 处：`torch.save(net, "model.pt")`
    存的是整个模型对象（pickle）。它绑定代码结构——`torch.load` 时要能 import 到
    定义 Scale 的模块；换目录、改类名就读不出来。
    改法：`torch.save(net.state_dict(), "model.pt")`。

第 3 处：`net2.load_state_dict(torch.load("model.pt"))`
    与第 2 处连在一起：`torch.load` 读回来的是**模型对象**，不是 state_dict，
    直接传给 `load_state_dict` 会报类型错。
    改法：先存 state_dict，再 `net2.load_state_dict(torch.load("model.pt"))`。
    注意那是"先改第 2 处"之后才成立的写法。
"""


if __name__ == "__main__":
    layer = CenteredLayer()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print("T1 CenteredLayer:", layer(x).tolist(), " 均值:", round(layer(x).mean().item(), 6))

    dense = MyDense(5, 3)
    print("T2 MyDense 参数:", [(n, tuple(p.shape)) for n, p in dense.named_parameters()])
    print("T2 前向:", tuple(dense(torch.randn(2, 5)).shape))

    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2), CenteredLayer())
    out = save_and_load(net, str(REPO / "work" / "b14" / "demo.pt"))
    print("T3 键名:", out["saved_keys"])
    print("T3 max_diff:", out["max_diff"])

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("T4 设备:", dev)
    t = torch.randn(3)
    print("T4 同设备返回 self:", to_device(t, t.device) is t)
    moved = move_net(nn.Linear(2, 3), dev)
    print("T4 模型参数设备:", next(moved.parameters()).device)
