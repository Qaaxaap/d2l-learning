"""
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self): 没有调用父类的同名函数。需要super().__init__()
        self.lin = nn.Linear(4, 4)
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        x = nn.functional.relu(self.lin(x))
        x = nn.functional.relu(self.lin(x))
        return self.out(x)

net = Net()
print("参数个数:", len(net.parameters())) 这个只能访问到有多少张量。访问个数的话需要累加生成器里的张量大小，前面题里有。
print("权重个数:", sum(1 for n, p in net.named_parameters() if "weight" in n))

net2 = Net()
net2.lin.weight.data = torch.ones_like(net2.lin.weight) 换了存储，可能导致优化器更新旧的张量，用原地操作。
loss = net2(torch.randn(2, 4)).sum()
loss.backward()
print("lin.weight.grad:", net2.lin.weight.grad)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """三层全连接：20 → 64 → 32 → 3。

    - 两个隐藏层后各接一个 ReLU（用 nn.functional.relu 或 nn.ReLU 都行）
    - 层的属性名分别是 hidden1、hidden2、out（断言会按名字查参数）
    - 前向的输入形状 (batch, 20)，输出 (batch, 3)
    """
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(20, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 3)
    def forward(self, x:torch.Tensor):
        return self.out(nn.functional.relu(self.hidden2(nn.functional.relu(self.hidden1(x)))))

class SharedMLP(nn.Module):
    """一个 nn.Linear(8, 8) 在前向里被用两次。

    - 那个层叫 `shared`，形状 (8, 8) 的 Linear
    - 前向：x → shared → relu → shared → 输出
    - 输入 (batch, 8)，输出 (batch, 8)
    """
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(8, 8)
    def forward(self, x):
        return self.shared(nn.functional.relu(self.shared(x)))
    
def count_by_kind(net: nn.Module) -> tuple[int, int]:
    """统计一个模型里权重与偏置各有多少个张量，返回 (权重个数, 偏置个数)。

    判断依据是参数名字：以 "weight" 结尾算权重、以 "bias" 结尾算偏置。
    其他名字的参数不计入这两个数。
    """
    weight, bias = 0, 0
    for name, _ in net.named_parameters():
        if name.endswith("weight"):
            weight += 1
        if name.endswith("bias"):
            bias += 1
    return (weight, bias)

def count_params(net: nn.Module) -> int:
    """模型里所有可训练参数的元素总数（不是张量个数）。"""
    sum = 0
    for p in net.parameters():
        sum += p.numel()
    return sum

def half_zero_init(t: torch.Tensor) -> torch.Tensor:
    """把 t 原地改成一个特殊的分布：一半元素是 0，另一半是 [-10,-5] 与 [5,10] 上的均匀分布。

    做法提示：先整个填成 [-10, 10] 上的均匀分布，再把绝对值小于 5 的元素置成 0。
    返回传入的那个张量（原地改）。
    """
    with torch.no_grad():
        t.uniform_(-10, 10)
        t[t.abs() < 5] = 0
    return t

class StackedMLP(nn.Module):
    """用 nn.ModuleList 堆 num_hidden 个隐藏层，每层 (h, h) 的 Linear + ReLU，
    最后接一个 (h, num_outputs) 的 Linear 输出。

    构造签名：StackedMLP(num_inputs, h, num_hidden, num_outputs)
    前向：输入 (batch, num_inputs)，先过第一层 Linear(num_inputs, h)，
    再过 num_hidden-1 个 Linear(h, h)，最后过 Linear(h, num_outputs)。
    每个隐藏层之后都有 ReLU。

    注意：`first` 算一层，`middle` 里有 num_hidden-1 层，`out` 算一层，
    所以总层数是 num_hidden+1。属性名用 first、middle、out。
    """
    def __init__(self, num_inputs, h, num_hidden, num_outputs):
        super().__init__()
        self.first = nn.Linear(num_inputs, h)
        self.middle = nn.ModuleList([
            nn.Sequential(nn.Linear(h, h), nn.ReLU())
            for _ in range(num_hidden - 1)
        ])
        self.out = nn.Linear(h, num_outputs)
    def forward(self, x):
        x = F.relu(self.first(x))
        for block in self.middle:
            x = block(x)
        return self.out(x)