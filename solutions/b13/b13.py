"""B13 参考答案。做完再看。

跑一遍：
    python3 solutions/b13/b13.py
或者拿它验断言：
    python3 tools/check_b13.py solutions/b13/b13.py
"""

import torch
import torch.nn as nn


# ---------- T1 自定义 MLP ----------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(20, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 3)

    def forward(self, x):
        x = nn.functional.relu(self.hidden1(x))
        x = nn.functional.relu(self.hidden2(x))
        return self.out(x)


# ---------- T2 共享参数 ----------

class SharedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(8, 8)

    def forward(self, x):
        x = nn.functional.relu(self.shared(x))
        return self.shared(x)


# ---------- T3 参数统计 ----------

def count_by_kind(net: nn.Module) -> tuple[int, int]:
    weights = sum(1 for n, _ in net.named_parameters() if n.endswith("weight"))
    biases = sum(1 for n, _ in net.named_parameters() if n.endswith("bias"))
    return weights, biases


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


# ---------- T4 自定义初始化 ----------

def half_zero_init(t: torch.Tensor) -> torch.Tensor:
    # 叶子张量在开梯度的状态下不能原地改，所以包 no_grad
    with torch.no_grad():
        t.uniform_(-10, 10)
        t.mul_((t.abs() >= 5).float())
    return t


# ---------- T5 用 ModuleList 堆层 ----------

class StackedMLP(nn.Module):
    def __init__(self, num_inputs, h, num_hidden, num_outputs):
        super().__init__()
        self.first = nn.Linear(num_inputs, h)
        # num_hidden - 1 个 h -> h 的隐藏层，用 ModuleList 才能被登记
        self.middle = nn.ModuleList(
            [nn.Linear(h, h) for _ in range(num_hidden - 1)]
        )
        self.out = nn.Linear(h, num_outputs)

    def forward(self, x):
        x = nn.functional.relu(self.first(x))
        for layer in self.middle:
            x = nn.functional.relu(layer(x))
        return self.out(x)


"""T6 改错

第 1 处：`__init__` 里少了 `super().__init__()`
    后果：创建实例时就报 `AttributeError: cannot assign module before Module.__init__() call`，
    报在 `self.lin = nn.Linear(4, 4)` 那一行。
    改法：`__init__` 的第一行写 `super().__init__()`。

第 2 处：`len(net.parameters())`
    `parameters()` 返回的是**生成器**，没有 `len`。`named_parameters()` 同理。
    改法：`len(list(net.parameters()))`，或者 `sum(1 for _ in net.parameters())`。

第 3 处：`net2.lin.weight.data = torch.ones_like(net2.lin.weight)`
    这行能跑、前向也对，但 `.data = ...` 换的是"这个属性指向哪块存储"。
    如果外面有别的对象持有原存储（例如优化器），它还在更新旧的那块，
    新存储永远不会被更新。
    改法：原地改写内容。
        with torch.no_grad():
            net2.lin.weight.copy_(torch.ones_like(net2.lin.weight))
    或者 fill_：`net2.lin.weight.fill_(1.0)`。
"""


if __name__ == "__main__":
    import math

    net = MLP()
    print("T1 参数名：", [n for n, _ in net.named_parameters()])
    print("T1 前向：", tuple(net(torch.randn(2, 20)).shape))

    shared = SharedMLP()
    print("T2 参数个数：", len(list(shared.parameters())), "（应为 2）")
    with torch.no_grad():
        shared.shared.weight.zero_()
    print("T2 权重置零后输出是否全零：", bool((shared(torch.randn(4, 8)) == 0).all()))

    print("T3 MLP 的 (权重, 偏置)：", count_by_kind(MLP()))
    print("T3 MLP 参数量：", count_params(MLP()), "（应为 3411）")
    print("T3 SharedMLP 参数量：", count_params(SharedMLP()), "（应为 72）")

    t = torch.empty(10000)
    out = half_zero_init(t)
    nonzero = out[out != 0]
    print("T4 零的比例：", round((out == 0).float().mean().item(), 3))
    print("T4 非零元素的范围：", round(nonzero.min().item(), 2), "~", round(nonzero.max().item(), 2))
    print("T4 返回的是同一对象：", out is t)

    for num_hidden in (1, 3, 5):
        net5 = StackedMLP(20, 16, num_hidden, 5)
        n_tensors = len(list(net5.parameters()))
        print(f"T5 num_hidden={num_hidden}: 参数张量 {n_tensors} 个，前向 {tuple(net5(torch.randn(2, 20)).shape)}")
