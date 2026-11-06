# B14 代码题

把你的实现写在 `work/b14/b14.py` 里，函数名与签名必须和下面一致。

验收：`just check b14`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章做什么

三件配套的事，都与训练循环无关：

1. **写一个层**（T1、T2）——它是模型里的零件，比"写一个模型"小一号
2. **存与读**（T3）——训练完要能保存
3. **搬到 GPU**（T4）——训练慢了要加速

T5 是改错题。

**这一章不训练**，全是几十毫秒的操作。T4 需要一个可用的 CUDA 设备；
断言会在没有 GPU 时跳过那一项并说明。

验收分三类：**精确**（形状、键名、设备）、**范围**（初始化后的分布）、
**趋势**（存了再读回来是否一致）。

---

## T1 不含参数的层：`CenteredLayer`

```python
class CenteredLayer(nn.Module):
    """把输入的均值减掉。

    前向：返回 x - x.mean()，其中 x.mean() 是**全部元素的均值**（一个标量），
    不是每个样本自己的均值。
    """
```

| 检查 | 期望 |
|---|---|
| 是否含参数 | `parameters()` 为空 |
| `(2, 2)` 输入 `[[1,2],[3,4]]` | 输出全元素均值为 0，且等于 `[[-1.5,-0.5],[0.5,1.5]]` |
| 放进 `nn.Sequential` 里 | 能正常前向 |

## T2 含参数的层：`MyDense`

```python
class MyDense(nn.Module):
    """自己的全连接层，最后一个 ReLU。

    构造：MyDense(in_units, units)
    参数：self.weight 形状 (in_units, units)，self.bias 形状 (units,)
          两者都必须是 nn.Parameter
    前向：relu(x @ self.weight + self.bias)
    """
```

| 检查 | 期望 |
|---|---|
| `parameters()` | 2 个张量，形状分别为 `(in_units, units)` 与 `(units,)` |
| 两个参数都是 `nn.Parameter` | 是 |
| 前向形状 | 输入 `(batch, in_units)` → 输出 `(batch, units)` |
| 输出非负 | 全 `>= 0`（因为最后有 ReLU）|
| 在 `state_dict` 里 | 键名 `weight`、`bias` |

**注意 `weight` 的形状**：`(in_units, units)`，因为算的是 `x @ weight`。
这与 `nn.Linear(in, out).weight` 是 `(out, in)` 相反。

## T3 保存与加载

只写两个动作：

```python
def save_params(net: nn.Module, path: str) -> list[str]:
    """把 net 的 state_dict 存到 path，返回键名列表（按 state_dict 给的顺序）。"""


def load_params(net: nn.Module, path: str) -> None:
    """从 path 读参数，装进 net（原地改 net 的参数）。"""
```

`path` 是调用方给的字符串，指向 `work/b14/` 下的文件。

| 检查 | 期望 |
|---|---|
| 返回的键名 | 与 `net.state_dict().keys()` 一致 |
| 存完文件存在 | 是 |
| 读进一个**参数不同**的同结构模型后 | 该模型参数与原模型一致 |
| 读进一个**结构不同**的模型 | 抛 `RuntimeError` |

**断言怎么验的**（这部分不用你写）：断言自己 `copy.deepcopy` 造一个同结构模型、
把参数重随机、记下加载前的差，再调你的 `load_params`，比较加载后的差。
加载前必须明显不同、加载后必须接近 0，两个数都看——只看后者的话，
如果那个模型一开始就一样，加载与不加载都是 0，测不出加载有没有发生。

## T4 搬到 GPU

```python
def to_device(x: torch.Tensor, device) -> torch.Tensor:
    """把张量搬到 device。设备相同时返回同一个对象（不是副本）。"""


def move_net(net: nn.Module, device) -> nn.Module:
    """把模型的所有参数搬到 device，返回搬完之后的模型。

    返回的对象应当**就是传进来的那个模型**（nn.Module.to() 是原地操作）。
    """
```

| 检查 | 期望 |
|---|---|
| `to_device(x, x.device) is x` | `True` |
| `to_device(x, "cpu")` 后的设备 | `cpu` |
| 有 GPU 时 `to_device(x, "cuda")` | 设备为 `cuda:0`，原张量仍在 `cpu` |
| `move_net(net, device)` 的返回值 | 是同一个对象；参数设备都是 `device` |
| 没 GPU 时 | 这一项跳过并提示 |

## T5 改错

下面这段代码想实现"一个含参数的自定义层 + 保存到磁盘"。**三处**问题。

```python
import torch
import torch.nn as nn

class Scale(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.scale = torch.ones(n)

    def forward(self, x):
        return x * self.scale

net = nn.Sequential(nn.Linear(4, 4), Scale(4))
torch.save(net, "model.pt")

X = torch.randn(2, 4)
net2 = nn.Sequential(nn.Linear(4, 4), Scale(4))
net2.load_state_dict(torch.load("model.pt"))
print(net2(X))
```

三处分别指三个位置：**参数的声明**、**保存的方式**、**加载的目标**。

三段提示：第一处不在报错——训练时这个参数不会更新，而损失照样下降；
第二处能存、也能读回来，但换目录就读不出来，而且`torch.load` 的结果不是 `state_dict`；
第三处与第二处连在一起看。
