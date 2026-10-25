# B13 代码题

把你的实现写在 `work/b13/b13.py` 里，函数名与签名必须和下面一致。

验收：`just check b13`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章做什么

前面每一章都用 `nn.Sequential(...)` 一行堆模型。这一章换成自己写类，并且把配套的三件事
（参数访问、初始化、共享）各练一遍。

**不涉及训练**——这一章全是"模型怎么组织"，跑起来都是几毫秒的构造与前向。
训练在下一章（B14 的自定义层）与 B16 的 LeNet 里。

验收分三类：**精确**（参数个数、形状、初始化后的分布）、
**范围**（初始化后落在哪个区间）、**趋势**（前向输出随输入变化）。

---

## T1 自定义一个多层感知机

```python
class MLP(nn.Module):
    """三层全连接：20 → 64 → 32 → 3。

    - 两个隐藏层后各接一个 ReLU（用 nn.functional.relu 或 nn.ReLU 都行）
    - 层的属性名分别是 hidden1、hidden2、out（断言会按名字查参数）
    - 前向的输入形状 (batch, 20)，输出 (batch, 3)
    """
```

| 检查 | 期望 |
|---|---|
| 参数名 | `hidden1.weight`、`hidden1.bias`、`hidden2.weight`、`hidden2.bias`、`out.weight`、`out.bias` |
| 参数个数 | 6 个张量 |
| 前向 | 输入 `(2, 20)` 得到 `(2, 3)` |
| 是 `nn.Module` 子类 | 是 |

## T2 共享参数

```python
class SharedMLP(nn.Module):
    """一个 nn.Linear(8, 8) 在前向里被用两次。

    - 那个层叫 `shared`，形状 (8, 8) 的 Linear
    - 前向：x → shared → relu → shared → 输出
    - 输入 (batch, 8)，输出 (batch, 8)
    """
```

| 检查 | 期望 |
|---|---|
| 参数个数 | **2** 个张量（`shared.weight`、`shared.bias`），不是 4 个 |
| 同一个层是不是被用了两次 | 断言的检查方式：把 `shared.weight` 改成全零，输出应当也是全零 |
| 前向 | 输入 `(4, 8)` 得到 `(4, 8)` |

## T3 参数统计

```python
def count_by_kind(net: nn.Module) -> tuple[int, int]:
    """统计一个模型里权重与偏置各有多少个张量，返回 (权重个数, 偏置个数)。

    判断依据是参数名字：以 "weight" 结尾算权重、以 "bias" 结尾算偏置。
    其他名字的参数不计入这两个数。
    """


def count_params(net: nn.Module) -> int:
    """模型里所有可训练参数的元素总数（不是张量个数）。"""
```

| 检查 | 期望 |
|---|---|
| `count_by_kind(MLP())` | `(3, 3)` |
| `count_by_kind(SharedMLP())` | `(1, 1)` |
| `count_params(MLP())` | `20*64 + 64 + 64*32 + 32 + 32*3 + 3` = `3411` |
| `count_params(SharedMLP())` | `8*8 + 8` = `72` |

## T4 自定义初始化

```python
def half_zero_init(t: torch.Tensor) -> torch.Tensor:
    """把 t 原地改成一个特殊的分布：一半元素是 0，另一半是 [-10,-5] 与 [5,10] 上的均匀分布。

    做法提示：先整个填成 [-10, 10] 上的均匀分布，再把绝对值小于 5 的元素置成 0。
    返回传入的那个张量（原地改）。
    """
```

| 检查 | 期望 |
|---|---|
| 大张量（10000 个元素）上 | 0 的比例约 0.5（0.45 到 0.55）|
| 非零元素 | 都在 [-10,-5] 与 [5,10] 里 |
| 返回的是同一个对象 | `half_zero_init(t) is t` |
| 输入是叶子张量、`requires_grad=True` 时 | 不报错（说明用了 `no_grad`）|

## T5 用 `nn.ModuleList` 堆层

```python
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
```

| 检查 | 期望 |
|---|---|
| `StackedMLP(20, 16, 3, 5)` 的参数张量个数 | `2*(1+2+1)` = **8**（first、2 个 middle、out）|
| 前向 | 输入 `(2, 20)` 得到 `(2, 5)` |
| 隐藏层个数 | 就是传进去的 `num_hidden`（3 时应该有 3 个 h→h 的层）|

## T6 改错

下面这段代码想定义一个带共享层的模型，并统计它的参数个数。**三处**问题。

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        self.lin = nn.Linear(4, 4)
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        x = nn.functional.relu(self.lin(x))
        x = nn.functional.relu(self.lin(x))
        return self.out(x)

net = Net()
print("参数个数:", len(net.parameters()))
print("权重个数:", sum(1 for n, p in net.named_parameters() if "weight" in n))

net2 = Net()
net2.lin.weight.data = torch.ones_like(net2.lin.weight)
loss = net2(torch.randn(2, 4)).sum()
loss.backward()
print("lin.weight.grad:", net2.lin.weight.grad)
```

三处分别指三个位置：**`__init__` 的第一行**、**参数个数的统计**、**换权重的方法**。

三段提示：第一处会直接报错，报错点在给 `self.lin` 赋值那一行；
第二处 `parameters()` 与 `named_parameters()` 返回的都是生成器，`len()` 用不了；
第三处能跑通，但 `.data = ...` 换的是存储的指向，要用原地改写的写法。
