# B14 自定义层、保存与加载、GPU

B13 讲了"怎么写一个模型"。这一章是三件配套的事：

1. **写一个层**（而不是一个完整模型）——它是模型里的零件
2. **把张量或模型存到磁盘**——训练完要能保存
3. **把计算放到 GPU 上**——训练慢了要加速

三件事各自独立，共同点是都在"模型之外"。

## 与书对应

| 讲义小节 | 纸质书（第一版） | 电子版（第二版） |
|---|---|---|
| 1 自定义层 | 4.4 全节 | [5.4](https://zh.d2l.ai/chapter_deep-learning-computation/custom-layer.html) |
| 2 保存与加载 | 4.5 全节 | [5.5](https://zh.d2l.ai/chapter_deep-learning-computation/read-write.html) |
| 3 GPU 计算（含 3.5 GPU 不是永远快，本节补充实测）| 4.6 全节 | [5.6](https://zh.d2l.ai/chapter_deep-learning-computation/use-gpu.html) |
| 4 实现会踩的坑 | 散在正文里，本节汇总 | 同上 |
| 自测题 | 无 | 无 |

**解锁点**：4.4 里有一个 `CenteredLayer` 与一个 `MyDense` 要自己写；
4.5、4.6 没有解锁点（都是 API 用法）。

**两版差异**：

- **自定义层**：v1 用 `Block` + `self.params.get(...)`，torch 用 `nn.Module` + `nn.Parameter`
- **保存**：v1 是 `net.save_parameters()` / `load_parameters()`，
  torch 是 `torch.save(net.state_dict(), ...)` / `net.load_state_dict(...)`
- **GPU**：v1 用 `ctx=mx.gpu()`，torch 用 `device="cuda"`；模型是 `net.to(device)`

**这一章的 GPU 部分可以实测**：本机有 CUDA 可用的显卡。

## 0 这一单元解决什么问题

B13 讲了"继承 `nn.Module` 写一个模型"。这一章的第一个问题更小：**只写一个层**。

层的写法与模型没有区别——`nn.Module` 的子类，有 `__init__` 与 `forward`。
区别只在于**它会被放进更大的模型里**，所以两件事要留意：

1. **参数必须能被外层看见**：用 `nn.Parameter` 包起来
2. **形状要对**：外层不知道你有什么参数，只会把张量传进来

第二、三部分（保存与 GPU）是训练完成后与训练中途都要打交道的事，与模型结构无关。

## 1 自定义层

### 1.1 不含参数的层

书上第一个例子是 `CenteredLayer`：把输入的均值减掉。

```python
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x - x.mean()
```

**`x.mean()` 是对所有元素求平均**，得到一个零维张量，然后广播减掉。
输入 `(batch, features)` 时，它减的是"所有样本所有特征的总平均"，不是"每个样本减去
自己的平均"。

**如果想每个样本减自己的均值**，要 `x - x.mean(dim=1, keepdim=True)`。
两者不同，书上用的是前者。

**这个层没有参数**，可以直接放进 `nn.Sequential`：

```python
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
```

### 1.2 含参数的层

书上实现了一个自己的全连接层。torch 的写法：

```python
class MyDense(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units))

    def forward(self, x):
        return nn.functional.relu(x @ self.weight + self.bias)
```

**三处与"不含参数的层"不同**：

| 位置 | 做了什么 |
|---|---|
| `__init__` | 用 `nn.Parameter` 包住张量 |
| `parameters()` | 自动包含这两个参数（因为它们是 `nn.Parameter`）|
| `state_dict()` | 自动包含它们，键名是 `weight`、`bias` |

**`nn.Parameter` 包起来是必须的**。直接写 `self.weight = torch.randn(...)` 的话，
`net.parameters()` 收不到它，优化器不会更新它——B13 的 2.3 节讲过这张表。

**形状约定**：`x @ self.weight` 要求 `x` 的最后一维等于 `in_units`。
这与 `nn.Linear` 的 `weight` 形状**相反**——`nn.Linear(in, out).weight` 是 `(out, in)`，
因为它内部算的是 `x @ weight.T`。

**这是个容易混的地方**：自己写的层用 `x @ W`（`W` 是 `(in, out)`），
框架的 `nn.Linear` 用 `x @ W.T`（`W` 是 `(out, in)`）。两者都对，只是约定不同。

### 1.3 一组参数用 `nn.ParameterList`

一层里有多个同类型的参数时（例如多个卷积核），用 `nn.ParameterList`：

```python
class MultiBias(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(n)])

    def forward(self, x):
        for b in self.biases:
            x = x + b
        return x
```

**与 B13 的 `nn.ModuleList` 对应**：`ModuleList` 装的是**层**，`ParameterList` 装的是**参数**。
两者都是容器，都会被自动登记。

## 2 保存与加载

### 2.1 存张量

```python
torch.save(x, "x.pt")          # 存一个张量
y = torch.load("x.pt")         # 读回来

torch.save([x, y], "xy.pt")                    # 存列表
torch.save({"x": x, "y": y}, "dict.pt")        # 存字典
```

**`torch.save` 接受任何 Python 对象**（张量、列表、字典、数字……），
内部用 pickle 序列化。**`torch.load` 读回来的是同一个类型**。

**文件名**：torch 没有强制后缀，习惯用 `.pt` 或 `.pth`。

**安全提醒**：`torch.load` 会执行 pickle 里的代码。只读自己生成的文件；
读别人的模型文件时，优先用 `weights_only=True`（PyTorch 2.6 起是默认值）。

### 2.2 存模型参数

**标准做法是存 `state_dict`**：

```python
torch.save(net.state_dict(), "mlp.params")

net2 = MLP()                                    # 先建好同样结构的模型
net2.load_state_dict(torch.load("mlp.params"))
```

`state_dict()` 是一个**名字到张量的字典**：

```python
{'hidden.weight': tensor(...), 'hidden.bias': tensor(...),
 'output.weight': tensor(...), 'output.bias': tensor(...)}
```

**`load_state_dict` 会检查键名与形状**。键名对不上会报
`RuntimeError: Error(s) in loading state_dict for MLP: Missing key(s) in state_dict: ...`；
形状对不上也会报。这是好事——能立刻发现"模型结构变了但旧参数还在用"。

**`save_parameters` / `load_parameters`** 是 v1 书上的 Gluon API，torch 里没有。
torch 的对应就是上面这两行。

### 2.3 存整个模型（不推荐）

```python
torch.save(net, "whole.pt")        # 把模型对象整个 pickle 下来
net = torch.load("whole.pt")
```

**能跑，但不推荐**，原因有两个：

1. **它绑定代码结构**。`torch.load` 时要能 import 到定义这个类的模块。换个目录、
   改了类名，就读不出来了
2. **参数与结构混在一起**。想复用参数、或者只换结构比较，都不方便

**约定**：**存 `state_dict`，不存模型对象**。

## 3 GPU 计算

### 3.1 设备的表示

**先查有没有 GPU**：

```python
torch.cuda.is_available()        # True 或 False
torch.cuda.device_count()        # 几块
torch.cuda.get_device_name(0)    # 显卡型号
```

**torch 的"设备"用 `torch.device` 表示**：

```python
torch.device("cpu")
torch.device("cuda")             # 等价于 "cuda:0"
torch.device("cuda:1")           # 第二块
```

**写一个通用的设备变量**是常见做法：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**v1 的 `mx.cpu()` / `mx.gpu()` 对应 `"cpu"` 与 `"cuda"`**。

### 3.2 把张量放到设备上

三种写法：

```python
x = torch.tensor([1, 2, 3], device="cuda")     # 创建时就在显存
y = x.to("cuda")                               # 从别处搬过去
z = x.cuda()                                   # 简写，等价于 .to("cuda")
```

**`.to(device)` 在设备相同时返回 `self`，不同时返回一个新张量**：B06 的 5.1.1.2 节
实测过这件事，它导致 `.to("cuda")` 的结果不是叶子张量（如果原来在 CPU 上）。

**没有 `as_in_context` 这类"共享存储"的函数**。v1 书上的 `as_in_context` 在设备相同时
返回同一个对象、`copyto` 总是新建——torch 里 `.to()` 的行为就是前者，`.` 没有后者的对应物。
要复制用 `.clone()`。

**`.cpu()`**：从显存搬回内存，与 `.to("cpu")` 等价。

### 3.3 把模型放到 GPU

```python
net = MLP()
net = net.to(device)          # 把参数搬到 device 上
```

**`.to()` 作用在模型上时会递归处理所有子模块的参数与缓冲区**，
并且是**原地改**（返回的还是同一个模型对象），所以 `net = net.to(device)` 里的赋值
不是必须的——但写上更安全，因为有些人会不习惯。

**一行之后所有参数都在 `device` 上了**：

```python
print(next(net.parameters()).device)      # cuda:0
```

**数据与模型必须在同一个设备上**：

```python
net = net.to("cuda")
X = X.to("cuda")              # 数据也要搬
y = net(X)                    # 否则报错
```

报错信息大致是：

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

### 3.4 混合设备会报错，而且是好事

**torch 不会自动搬数据**。把 CPU 张量与 GPU 张量相加会直接报错。

这与 v1 书上说的一致。**为什么不允许自动搬**：CPU 与 GPU 之间的传输很慢，
自动搬会让"为什么这段代码慢"变得难以定位。显式搬迫使人知道自己在做什么。

### 3.5 GPU 不是永远快

实测（RTX 4070 SUPER，本机）：

| 运算 | CPU | GPU |
|---|---|---|
| 2000×2000 矩阵乘法 | 16 ms | 34.8 ms |

**GPU 反而慢一倍。** 原因是 CUDA 调用有固定开销——内核启动、传输、
以及这次要先把两块 2000×2000 的矩阵搬到显存（16 MB）。规模小时这些开销压过了计算速度。

**GPU 的优势在规模**：

- 几万维的矩阵乘、卷积、批量图像处理
- 同一组数字要做的重复运算多（并行的粒度够粗）
- 数据已经在显存里、不用来回搬

**实践含义**：小数据集上先用 CPU 跑通，确认逻辑对了再把 `device` 切到 `cuda`。
不要一上手就上 GPU——小矩阵上它只会更慢，还多一层调试的复杂度。

**一个判断方法**：把 `device` 改成 `cpu` 与 `cuda` 各跑一次，看总时间。
如果 GPU 更慢，说明这个任务的规模还压不过开销。

**常见的一个坑**：在批循环里忘了搬数据。

```python
for X, y in loader:
    X, y = X.to(device), y.to(device)     # 这句不能少
    ...
```

**更难查的一种**：损失函数里自己造的张量没搬。例如 `torch.tensor(0.0)` 在 CPU 上，
与 GPU 上的损失相加就会报错。写成 `torch.tensor(0.0, device=device)`。

## 4 实现会踩的坑

| 坑 | 症状 | 原因 |
|---|---|---|
| 参数没用 `nn.Parameter` 包 | `parameters()` 收不到，训练不更新 | 普通张量属性不登记（1.2）|
| 自己写的层照搬 `nn.Linear` 的形状约定 | 矩阵乘法形状错 | `nn.Linear` 的 `weight` 是 `(out, in)`（1.2）|
| `CenteredLayer` 以为是逐样本中心化 | 结果与预期不同 | `x.mean()` 是对全部元素求平均（1.1）|
| 用 `torch.save(net)` 存整个模型 | 换目录后读不出来 | 绑定代码结构（2.3）|
| `load_state_dict` 键名对不上 | `RuntimeError: Missing key(s)` | 模型结构与保存时不同（2.2）|
| 忘了 `X.to(device)` | `Expected all tensors to be on the same device` | 数据还在 CPU（3.3）|
| 损失里临时造的张量没搬 | 同上，但报错位置奇怪 | `torch.tensor(0.0)` 默认在 CPU（3.4）|
| 以为 `.to("cuda")` 是原地改 | 原张量没变 | `.to()` 返回新张量（3.2）|

## 命令速查

| 命令 | 作用 | 讲义哪一节 |
|---|---|---|
| `class X(nn.Module)` + `nn.Parameter` | 含参数的自定义层 | 1.2 |
| `nn.ParameterList([...])` | 装一组参数的容器 | 1.3 |
| `torch.save(obj, path)` | 存任意 Python 对象（张量、字典等）| 2.1 |
| `torch.load(path)` | 读回来 | 2.1 |
| `net.state_dict()` | 名字到张量的字典 | 2.2 |
| `net.load_state_dict(d)` | 按名字加载，检查键与形状 | 2.2 |
| `torch.cuda.is_available()` | 有没有可用 GPU | 3.1 |
| `torch.device("cuda")` | 设备对象 | 3.1 |
| `t.to(device)` / `t.cuda()` / `t.cpu()` | 搬张量 | 3.2 |
| `net.to(device)` | 把模型全部参数搬到设备 | 3.3 |

## 自测题

1. 写一个含参数的自定义层，`__init__` 里必须用哪个东西包住参数？不包会怎样？
2. 自己写的层算 `x @ W`，`nn.Linear` 内部算 `x @ W.T`。两者的 `W` 形状分别是多少？
3. 存模型的标准做法是存什么？为什么不推荐 `torch.save(net)`？
4. `load_state_dict` 会检查什么？检查出来对不上说明什么？
5. 把模型放到 GPU 用哪一句？数据呢？
6. `.to("cuda")` 是原地改还是返回新张量？
7. 为什么 torch 不允许 CPU 张量与 GPU 张量直接相加？

## 答案（做完再看）

**1.** 必须用 `nn.Parameter` 包住。不包的话 `net.parameters()` 收不到它，
优化器不会更新它——训练时它一直保持初始值，而且不报错。

**2.** 自己的层用 `x @ W`，`W` 形状 `(in_units, units)`。
`nn.Linear(in, out).weight` 形状是 `(out, in)`，因为它内部算 `x @ W.T`。

**3.** 存 `net.state_dict()`（名字到张量的字典）。不推荐 `torch.save(net)` 的原因有两条：
它绑定代码结构（换目录、改类名就读不出来），参数与结构混在一起也不好复用。

**4.** 检查键名与每个张量的形状。对不上说明模型结构与保存时不同（加了层、改了维度，
或者名字变了）。这是好事——能立刻发现"结构变了但还在用旧参数"。

**5.** 模型：`net = net.to(device)`；数据：`X = X.to(device)`。两者必须在同一个设备上。

**6.** 返回新张量（设备不同的时候）。设备相同时返回 `self`。

**7.** 因为 CPU 与 GPU 之间的传输很慢。自动搬会让性能问题难以定位；
显式搬迫使人知道数据在哪。
