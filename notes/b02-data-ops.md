# B02 数据操作与数据预处理

对应 v1 第 2.2 节（纸质书，MXNet）与 v2 第 2.1、2.2 节（电子版，看 PyTorch tab）。
本单元解决数据怎么进程序、怎么改形状、怎么算；后面每一章都在写张量运算，这里的形状规则和
内存语义没弄明白，之后的维度错误只能靠试错定位。

torch 行为在 torch 2.14 上跑过（CPU 与远程 4070S 各测一遍），pandas 在 2.3.3 上跑过，报错文本照抄实测输出。

## 1. 两版代码的命名差别

v1 用 `nd` 模块（`NDArray`），v2 电子版的 MXNet tab 换成 numpy 风格的 `np` 并先调 `npx.set_np()`。
两套名字指同一类对象，认第二列：

| 书上（v1，`nd.`） | PyTorch |
|---|---|
| `nd.arange(12)` | `torch.arange(12)`，dtype 推断规则不同，见 2.1 |
| `nd.zeros((2, 3, 4))`、`nd.ones((3, 4))` | `torch.zeros(2, 3, 4)`、`torch.ones(3, 4)` |
| `nd.array([[2, 1], [1, 2]])` | `torch.tensor([[2, 1], [1, 2]])` |
| `nd.random.normal(0, 1, shape=(3, 4))` | `torch.normal(0., 1., size=(3, 4))`，标准正态用 `torch.randn(3, 4)` |
| `x.size`（属性，元素总数） | `x.numel()`；torch 的 `x.size()` 是方法且返回形状，见 2.3 |
| `x.reshape((3, 4))`、`Y.exp()` | `x.reshape(3, 4)`、`torch.exp(Y)` |
| `nd.concat(X, Y, dim=0)`、`nd.dot(X, Y.T)` | `torch.cat([X, Y], dim=0)`、`X @ Y.T`，矩阵乘法在 B03 |
| `X.asnumpy()`、`nd.elemwise_add(X, Y, out=Z)` | `X.numpy()`、`torch.add(X, Y, out=Z)`，内存语义见第 7 节 |

## 2. 创建张量与形状

### 2.1 dtype 是第一个坑

```python
import torch

torch.arange(12)                        # int64
torch.arange(12, dtype=torch.float32)   # 要浮点就显式指定
torch.zeros(2, 3, 4); torch.ones(3, 4); torch.randn(3, 4)   # randn 是标准正态，参数就是形状
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4]])
```

**`arange` 的默认元素类型是整数**：`torch.arange(12)` 给出 `int64`。要浮点必须显式写出来：

```python
torch.arange(12, dtype=torch.float32)
```

为什么不能省：整数张量后面按浮点用会出问题。B02 里能直接观察到的是类型提升带来的意外结果
（`int64 + float32` 悄悄变成 `float32`，结果类型和你写下的不一样），后面几章会遇到更硬的报错。

一处对照：MXNet 的 `nd.arange(12)` 默认给 `float32`（它的默认浮点类型叫 `mx_real_t`），
和 torch 不一样。**判断依据是你需要什么类型，不是书上写没写 `dtype`。**
（依据：MXNet 源码 `ndarray.py` 的 `def arange(..., dtype=mx_real_t)`。）

### 2.2 `torch.tensor` 与 `torch.Tensor`

| 写法 | 结果 |
|---|---|
| `torch.tensor([1, 2])` / `torch.Tensor([1, 2])` | 前者由数据推断得 `int64`，后者恒为 `float32` |
| `torch.Tensor(2, 3)` | 形状 `(2, 3)` 的**未初始化**张量，内容是内存残留值（实测出现过 `3.83e+24`） |
| `torch.tensor(2, 3)` | `TypeError: tensor() takes 1 positional argument but 2 were given` |
| `torch.Tensor([1, 2], dtype=torch.int64)` | `TypeError`，该构造器不接受 `dtype` 关键字 |

`torch.Tensor` 是历史遗留的类构造器，既当数据构造又当形状构造，还改不了 dtype；要空张量写 `torch.empty(2, 3)`，其余场合统一用 `torch.tensor`。

### 2.3 形状与元素个数

```python
x = torch.arange(12)
x.shape        # torch.Size([12])，tuple 的子类
x.numel()      # 12，元素总数，对应书上的 x.size
x.size()       # 返回形状；x.size(0) 是第 0 轴长度；x.dim() 是轴数；len(x) 也是第 0 轴长度
```

`x.size` 在书上是属性，在 torch 里是方法；写成 `x.size` 而不调用拿到的是内置方法对象，报错位置离出错位置很远。数元素个数用 `numel()`。

### 2.4 reshape 与 view

```python
x = torch.arange(12)
X = x.reshape(3, 4)      # 也可以写 x.reshape((3, 4))、x.reshape(-1, 4)、x.reshape(3, -1)
```

`-1` 表示该维长度由元素总数和其余维度推断，推断不出来会报错。`reshape` 与 `view` 的差别在于
是否允许重新分配内存：

```python
Y = torch.arange(12).reshape(3, 4)
Y.t().view(-1)      # RuntimeError: view size is not compatible with input tensor's size and stride
Y.t().reshape(-1)   # 成功，形状 (12,)
```

`view` 只接受能在原存储上重新解释的情况，要求内存连续（这里 `Y.t().is_contiguous()` 为 `False`）。
转置把行优先的 stride `(4, 1)` 换成 `(1, 4)`，展平后下标不再单调，无法用一个 stride 描述；
`reshape` 遇到这种情况会复制一份，实测连续的 `Y.reshape(-1)` 与源共享内存，`Y.t().reshape(-1)` 的 `data_ptr()` 与源不同。要展平不连续张量，先 `.contiguous()` 再 `view`。

### 2.5 形状对不上时怎么报告

T2 要求"形状不能广播时抛 `ValueError`，信息里含第一个不匹配的维度"。
`raise` 的完整语法在 A1 讲义第 9 节，这里只说要点：

```python
raise ValueError(f"形状 {sa} 与 {sb} 无法广播：左对齐后第 {i} 维长度分别为 {da} 和 {db}")
```

- `raise` 抛出异常。调用者不捕获的话程序停在那里，终端上打印你写的信息
- `ValueError` 表示"值不对"，形状不匹配属于这一类
- 信息用 f-string 把具体数值拼进去。只写"形状不对"没有用，调用者需要知道是哪里不对

## 3. 逐元素运算与拼接

```python
X = torch.ones(3, 4); Y = torch.zeros(3, 4)
x = torch.tensor([1.0, 2, 4, 8]); y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y   # 按元素，形状不变
torch.exp(x)                         # x.exp() 也能跑，函数形式更常见
X == Y                               # 逐元素比较
X.sum()                              # 所有元素求和，返回 0 维张量
torch.cat([X, Y], dim=0)             # 沿轴 0 拼接，其余各轴长度必须相同
```

- `X == Y` 返回 `torch.bool` 张量，书上描述为值为 0 或 1 的张量。bool 参与算术会提升类型：`(X == Y) + X` 得 `int64`，`(X == Y).sum()` 是 int64 的计数。
- `X.sum()` 的结果可以放进 `if`；多元素张量做布尔判断报 `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`。
- 按元素运算和 `cat` 会做类型提升（`float32 + int64` 得 `float32`，`float32 + float64` 得 `float64`）；矩阵乘法两侧 dtype 必须一致，否则报 `expected m1 and m2 to have the same dtype`。

## 4. 广播机制

规则三条：两个形状从最右边一维开始向左对齐；每一维上长度相等、其中一方为 1、或其中一方缺失（视为 1）才允许广播；结果的每一维取两者在该维上的最大长度。

```python
a = torch.arange(3).reshape(3, 1); b = torch.arange(2).reshape(1, 2)
(a + b).shape                              # (3, 2)

torch.ones(3, 4) + torch.ones(4)           # (3, 4)，一维向量按最后一维对齐
torch.ones(2, 1, 4) + torch.ones(1, 3, 1)  # (2, 3, 4)
torch.ones(3, 4) + torch.ones(3)           # RuntimeError: The size of tensor a (4) must match
                                           # the size of tensor b (3) at non-singleton dimension 1
```

最后一行是常见错误：长度为 3 的一维张量右对齐到最后一维，与长度 4 冲突；想按行运算，长度 3 的量必须先变成 `(3, 1)`。

广播不复制数据，`t.expand(3, 4)` 的 stride 实测是 `(1, 0)`，第 0 步长为 0 表示同一份数据被重复读，所以改扩展结果等于改原张量：

```python
t = torch.ones(3, 1); e = t.expand(3, 4)
e[0, 0] = 5.0
t            # tensor([[5.], [1.], [1.]])
```

要真正复制一份用 `t.repeat(1, 4)`。`keepdims` 与广播直接相关：让被消掉的那一维以长度 1 的形式留在原位，好让两边对齐，`A / A.sum(axis=1)` 报错而 `A / A.sum(axis=1, keepdims=True)` 正确，原因在 B03 第 4 节。

## 5. 索引与切片

```python
X = torch.arange(12).reshape(3, 4)
X[-1]          # 最后一行
X[1:3]         # 第 1、2 行，左闭右开
X[1, 2] = 9    # 写单个元素；X[0:2, :] = 12 写一块
```

torch 的切片默认返回**视图**，这一点和 Python 的 list 正好相反（list 的切片会复制一份）：

```python
X = torch.arange(12).reshape(3, 4)
s = X[1:3]
s[0, 0] = 99
X              # 第 1 行第 0 列变成 99，原张量被改了
```

只要后面还要用原张量，任何“取一部分改一改”的写法都要先 `.clone()`，实测 `X[1:3].clone()` 之后的修改不影响 `X`。返回副本的只有布尔索引 `X[X > 5]` 和整数数组索引 `X[[0, 2]]`，它们的结果在内存里不连续，无法用 stride 描述。

## 6. 内存复用与 in-place 操作

书里用 `id()` 演示的结论在 torch 上一样：`Y = Y + X` 会分配新内存再让 `Y` 指向它。

```python
X = torch.ones(3, 4)
Y = torch.zeros(3, 4)
Y = Y + X                 # id 变了，新内存
Y[:] = X + Y              # id 不变，写回原内存；Y += X 也是原地；torch.add(X, Y, out=Y) 返回 Y 本身
```

**这里有一个坑，现在只要知道，B04 会展开**：如果张量是用 `requires_grad=True` 造出来的
（也就是之后要参与求导的），在它上面做 in-place 操作有风险，轻则报错，重则静默算错梯度。
B02 这一章的张量都不求导，暂时碰不到。完整规则与实测数据在 B04 讲义第 5.6 节。

## 7. 与 NumPy 互转

```python
import numpy as np

D = torch.tensor(np.ones((2, 3)))   # 对应书上的 nd.array(P)
D.numpy()                           # 对应书上的 D.asnumpy()
```

两版书在这里的描述相反：v2 的 MXNet 段落写“转换后的结果不共享内存”，PyTorch 段落写“torch 张量和 numpy 数组将共享它们的底层内存”。torch 侧实测：

```python
T = torch.arange(6, dtype=torch.float32).reshape(2, 3)
N = T.numpy(); N[0, 0] = 99.0
T[0, 0]        # tensor(99.)，numpy 侧的修改进了张量
```

反方向是否共享取决于函数：`torch.tensor(N)` 复制，`torch.from_numpy(N)` 与 `torch.as_tensor(N)` 共享。
`.numpy()` 有一个硬条件：张量必须在 CPU 上。

```python
torch.arange(3, device=0).numpy()
# TypeError: can't convert cuda:0 device type tensor to numpy.
# Use Tensor.cpu() to copy the tensor to host memory first.（开发机实测）
```

还有一个条件与"梯度"有关，B04 学完自动微分才会碰到，那时再回来看这一段。

dtype 跟着 NumPy 走：`to_numpy(dtype=float)` 给出 NumPy 的 `float64`，转回 torch 张量就是 `torch.float64`。
想统一到 `float32`，用 `.float()` 显式转。

### 7.1 `.item()` 与 `.numpy()` 什么时候用

| 场景 | 用法 |
|---|---|
| 把一个单元素张量取成 Python 数值 | `.item()` |
| 交给 NumPy、pandas、matplotlib | `.cpu().numpy()` |

`.item()` 要求张量只有一个元素：0 维、`(1,)`、`(1, 1)` 都可以，
`torch.tensor([1, 2]).item()` 报 `RuntimeError: a Tensor with 2 elements cannot be converted to Scalar`。

`.detach()` 与这两个方法有关，它断开张量与"计算图"的联系、但共享内存。
B02 的张量都不求导，用不上它；B04 学完自动微分再回来看。

## 8. 数据预处理

这一节 v1 没有对应内容，看 v2 的 2.2 节。

```python
import pandas as pd

with open('house_tiny.csv', 'w') as f:   # 每行一个样本，NA 表示缺失
    f.write('NumRooms,Alley,Price\nNA,Pave,127500\n2,NA,106000\n4,NA,178100\nNA,NA,140000\n')

data = pd.read_csv('house_tiny.csv')
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
```

`pd.read_csv` 把 `NA` 识别成缺失值，数值列 `NumRooms` 是 `float64`，字符串列 `Alley` 是 `object`。

### 8.1 书中这一句在新版 pandas 上会失败

v2 原文写 `inputs = inputs.fillna(inputs.mean())`，在 pandas 2.3.3 上实测报
`TypeError: can only concatenate str (not "int") to str`：`inputs` 里同时有数值列和字符串列，
`mean()` 会去对字符串列求均值。改成 `inputs.fillna(inputs.mean(numeric_only=True))` 即可。

`fillna` 收到 Series 时按**列名**对齐，Series 里没有的列保持原样，所以这一句只填了 `NumRooms`，`Alley` 的缺失值留给下一步；实测 `pd.Series({'A': 100.0})` 只影响 A 列。删除法对应 `data.dropna()`（删行）与 `data.dropna(axis=1, how='any')`（删列）；插值还是删除要看缺失比例，盲目填均值会把分布压窄。

### 8.2 类别列与独热编码

```python
inputs = pd.get_dummies(inputs, dummy_na=True)
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
```

`dummy_na=True` 把缺失值本身当作一个类别，`Alley` 生成 `Alley_Pave` 和 `Alley_nan` 两列。
当前 pandas 的 `get_dummies` 返回 bool 列，`to_numpy(dtype=float)` 把 True/False 转成 1.0/0.0。
得到的 `X` 是 `torch.float64`、形状 `(4, 3)`，喂给模型前通常再转一次 `float32`。电子版这里的
MXNet 写法是 `np.array(inputs.to_numpy(dtype=float))`，对应 torch 的 `torch.tensor(...)`。

## 自测题

1. `torch.arange(6)` 的 dtype 是什么？要让它变成 `float32` 有哪两种写法？把一个 `int64` 张量和一个 `float32` 张量相加，结果的 dtype 是什么？这个行为为什么值得注意？
2. `torch.tensor([1, 2])` 与 `torch.Tensor([1, 2])` 的 dtype 各是什么？`torch.Tensor(2, 3)` 造出来的张量内容是什么？为什么讲义建议统一用 `torch.tensor`？
3. `X = torch.arange(12).reshape(3, 4)`，执行 `s = X[1:3]; s[0, 0] = 99` 之后 `X` 是什么？写成 `s = X[1:3].clone()` 呢？`X[X > 5]` 返回视图还是副本，为什么？
4. 为什么 `Y.t().view(-1)` 报错而 `Y.t().reshape(-1)` 不报错？两者的返回值与 `Y` 共享内存吗？
5. `A` 的形状是 `(5, 4)`。`A / A.sum(dim=1)` 报什么错？加上 `keepdim=True` 之后为什么就对了？把广播的三条规则写出来。
6. `Y = torch.ones(3, 4)`。`Z = Y + 1`、`Y[:] = Y + 1`、`Y += 1` 三种写法里，哪些会新建张量、哪些是原地写回？`id(Y)` 在哪几种写法之后不变？
7. 写一段代码把形状 `(3, 4)` 的 `float32` 张量每一行除以该行元素之和，结果仍是 `float32`，且不修改原张量，不用循环。
8. 一个 0 维张量、一个形状 `(1,)` 的张量、一个形状 `(1, 1)` 的张量，`.item()` 都能用吗？形状 `(2,)` 的呢？

## 答案（做完再看）

1. `int64`。写成 `torch.arange(6, dtype=torch.float32)`，或者生成之后调 `.float()`。
结果是 `float32`，规则是取两者里"更宽"的那个。值得注意是因为它不报错就悄悄换了类型，
后面依赖具体 dtype 的地方会出问题。
2. 分别是 `int64`（从数据推断）和 `float32`。`torch.Tensor(2, 3)` 给未初始化张量，内容是内存残留值，该构造器还不接受 `dtype` 关键字。
3. `X` 第 1 行第 0 列变成 99，基本切片返回视图；`clone()` 之后 `X` 不变。`X[X > 5]` 是副本，布尔索引的结果在内存里不连续，无法用 stride 描述。
4. `Y.t()` 不连续，stride 变成 `(1, 4)`，`view` 无法描述展平后的下标；`reshape` 在不能重新解释时复制一份，返回值与 `Y` 不共享内存（`data_ptr()` 不同）。
5. 报 `RuntimeError: The size of tensor a (4) must match the size of tensor b (5) at non-singleton dimension 1`。`A.sum(dim=1)` 形状是 `(5,)`，右对齐到最后一维与 4 冲突；`keepdim=True` 得 `(5, 1)`，第 1 维长度为 1，可广播到 `(5, 4)`。三条规则见第 4 节。
6. `Z = Y + 1` 新建张量，`Y` 不变；`Y[:] = Y + 1` 与 `Y += 1` 都是原地写回，
    `id(Y)` 在这两种写法之后不变。判断依据是"有没有把结果写回同一块内存"。
7. `B = A / A.sum(dim=1, keepdim=True)`。要点是 `keepdim=True`（或先 `A.sum(1).reshape(-1, 1)`）、输入已是 `float32`、`/` 产生新张量不改 `A`。
8. 前三个都能用，它们的元素总数都是 1；`(2,)` 报 `RuntimeError: a Tensor with 2 elements cannot be converted to Scalar`。
