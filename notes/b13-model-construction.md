# B13 模型构造与参数管理

从 B06 到现在，模型一直是 `nn.Sequential(...)` 一行堆出来的。这一章换成另一种写法：
**自己定义一个类**。然后是配套的三件事——访问参数、初始化参数、共享参数。

这一章不加新模型，只把"模型是什么"这件事说透。后面 B15 起的卷积网络、B17 的
各种复杂结构，都要靠自己定义模块才能拼出来。

## 与书对应

| 讲义小节 | 纸质书（第一版） | 电子版（第二版） |
|---|---|---|
| 1 继承 `nn.Module` | 4.1 的"继承 `Block` 类来构造模型" | [5.1](https://zh.d2l.ai/chapter_deep-learning-computation/model-construction.html) 的对应小节 |
| 2 构造复杂模型 | 4.1 的"`Sequential` 类"、"构造复杂的模型" | 5.1 的"自定义块"与"顺序块" |
| 3 访问参数 | 4.2 的"访问模型参数" | [5.2](https://zh.d2l.ai/chapter_deep-learning-computation/parameters.html) 的"参数访问" |
| 4 初始化参数 | 4.2 的"初始化模型参数"、"自定义初始化方法" | 5.2 的"参数初始化" |
| 5 共享参数 | 4.2 的"共享模型参数" | 5.2 的"参数绑定" |
| 6 延后初始化 | 4.3 全节 | 5.1 末与 5.2 提到 |
| 7 实现会踩的坑 | 散在正文里，本节汇总 | 同上 |
| 自测题 | 无 | 无 |

**解锁点**：v1 的 4.1 里有一个 `MySequential` 要自己写，4.2 里有一个 `MyInit`。
这一章不涉及 d2lzh 的解锁点（4.1–4.3 的正文都在讲怎么写类，不是封装好的函数）。

**两版差异很大**：v1 用 MXNet 的 `Block`，v2 用 torch 的 `nn.Module`，两边的 API 名字
几乎都不一样。这一章以 torch 为主，把 v1 的说法作为对照。

## 0 这一单元解决什么问题

`nn.Sequential` 能表达的是"一层接一层"。有三种情况它表达不了：

1. **层之间不是简单串联**：有分支、有跳连（ResNet）、有跨层拼接（DenseNet）
2. **同一个层用两次**（共享参数）
3. **前向里有控制流**：`while`、`if` 这类

要表达这些，得自己写一个类。这一章的三节各管一件事：

| 节 | 问题 |
|---|---|
| 1、2 | 怎么写这个类 |
| 3、4 | 类里的参数怎么访问、怎么初始化 |
| 5 | 两个层怎么共用同一份参数 |
| 6 | 书上的"延后初始化"是什么，torch 里为什么没有 |

## 1 继承 `nn.Module`

### 1.1 两个方法

自己定义一个模型，就是**继承 `nn.Module` 并写两个方法**：

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super().__init__()                        # 必须调用
        self.hidden = nn.Linear(num_inputs, num_hiddens)
        self.output = nn.Linear(num_hiddens, num_outputs)

    def forward(self, x):
        return self.output(nn.functional.relu(self.hidden(x)))
```

**`__init__`**：声明这个模型有哪些层。写法上就是把层赋给 `self` 的属性。

**`forward`**：定义"输入怎么一路算到输出"。

**`super().__init__()` 必须有。** `nn.Module` 的构造函数会初始化几个内部字典
（记录子模块与参数）。不调用它的后果是**创建实例时就报错**，实测：

```
AttributeError: cannot assign module before Module.__init__() call
```

报错点是 `self.hidden = nn.Linear(...)` 那一行——`nn.Module.__setattr__` 要往内部字典里
登记，而字典还没建。这个错还算友好，一看就知道。

**不需要写反向传播。** autograd 会从 `forward` 里用到的运算自动建图。

### 1.2 `__call__` 与 `forward` 的关系

**调用时写 `net(x)`，不是 `net.forward(x)`。**

`net(x)` 触发的是 `nn.Module.__call__`，它内部做几件事再调 `forward`：

- 处理 forward hook（挂在前向上面的回调，调试与可视化用）
- 设置 `_called` 之类的内部状态

**直接调 `net.forward(x)` 会跳过这些**，一般也能算对，但 hook 不生效。
约定是 `net(x)`。

### 1.3 与 `Sequential` 的关系

`nn.Sequential` 本身也是 `nn.Module` 的子类。它的 `forward` 就是"按顺序把每一层算一遍"。

所以两种写法可以互相嵌套：`Sequential` 里可以放自定义模块，自定义模块里也可以放
`Sequential`。书上专门写了一个 `MySequential` 来演示这件事的内部机制，
torch 里的对应是：

```python
class MySequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        for idx, layer in enumerate(layers):
            self.add_module(f"layer{idx}", layer)   # 登记子模块

    def forward(self, x):
        for layer in self.children():               # 按添加顺序遍历
            x = layer(x)
        return x
```

**`self.add_module(name, module)`** 是"把子模块登记到 `self` 名下"的显式写法。
平时写 `self.hidden = nn.Linear(...)` 时，`nn.Module` 会通过 `__setattr__` 自动做这件事，
所以不需要手写 `add_module`。只有在名字是**动态生成**的（比如上面 `f"layer{idx}"`）时，
才需要它——`self.layer0 = ...` 这种写死了名字的不用。

**`self.children()`** 返回子模块的迭代器，顺序与添加顺序一致。
torch 内部记子模块用的是 `OrderedDict`（v1 的书上叫 `_children`），所以顺序可靠。

## 2 构造复杂模型

### 2.1 嵌套

模块可以嵌套：自定义模块里放自定义模块。

```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.dense = nn.Linear(32, 16)

    def forward(self, x):
        return self.dense(self.net(x))
```

**嵌套之后，参数会自动递归收集。** `NestMLP().parameters()` 会给出全部四个 `Linear`
的权重与偏置，不需要手动遍历。

### 2.2 复用同一个层 = 共享参数

```python
class Shared(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(20, 20)

    def forward(self, x):
        x = nn.functional.relu(self.dense(x))
        return self.dense(x)          # 同一个层用第二次
```

`self.dense` 在前向里出现两次，**它是同一个对象**，所以两次用到的是同一份参数。
反向传播时两处的梯度会**加起来**（B04 讲的梯度累加）。

**这是最常用的共享参数写法**。v1 的书上还介绍了另一种用 `params=shared.params` 的写法，
那是 Gluon 的 API，torch 里没有——复用同一个层对象就够了。

### 2.1.1 层的列表要用 `nn.ModuleList`

想动态生成若干层时（"堆 n 个隐藏层"），不能在 `__init__` 里写一个普通 list：

```python
class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(2, 3), nn.Linear(3, 4)]    # 普通 list
```

**实测 `parameters()` 返回 0 个参数**——普通 list 不是 `nn.Module`，
`nn.Module.__setattr__` 不会把里面的层登记成子模块。前向里照样能用
（`for layer in self.layers: x = layer(x)` 算得对），但优化器收不到这些层的参数，
训练时它们不更新。

**正确写法**是 `nn.ModuleList`：

```python
class Good(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 3), nn.Linear(3, 4)])
```

实测参数 4 个。`nn.ModuleList` 是一个"装模块的 list"，它会登记里面的每个层；
它自己**没有 `forward`**，要在自定义 `forward` 里显式遍历。

**`nn.Sequential` 与 `nn.ModuleList` 的区别**：前者自己定义前向（按顺序算），
后者只负责装，前向要自己写。有控制流、有跳连时用后者。

### 2.3 常数不是参数

书上的 `FancyMLP` 里用 `params.get_constant` 造了一个"训练中不更新的权重"。

torch 里的两种对应：

| 做法 | 会被优化器更新吗 | 会被存进 `state_dict` 吗 |
|---|---|---|
| `self.w = nn.Parameter(torch.randn(20, 20))` | 会 | 会 |
| `self.w = torch.randn(20, 20)` | **不会** | 不会 |
| `self.register_buffer("w", torch.randn(20, 20))` | **不会** | **会** |

**普通张量属性**最接近书上的"常数"：它不参与训练，也不在 `state_dict` 里——
保存模型的代码里 `torch.save(net.state_dict(), ...)` 不会带上它。

**`register_buffer`** 是第三种：不训练，但**会**存进 `state_dict`。适合"需要保存的常量"
（例如 BatchNorm 的均值与方差，B18 会用到）。

**什么时候必须用 `nn.Parameter`**：这个张量要被优化器看到、要跟着训练更新。

### 2.4 控制流

`forward` 里可以写 Python 控制流：

```python
def forward(self, x):
    x = self.dense(x)
    while x.norm() > 1:
        x = x / 2
    if x.norm() < 0.8:
        x = x * 10
    return x.sum()
```

**注意比较要用 `.item()` 或让张量是标量**：`x.norm() > 1` 这个比较本身会得到
一个布尔**张量**（零维），`while` 接受它是因为零维张量能转换成布尔值。
多元素张量不行，会报 `Boolean value of Tensor with more than one value is ambiguous`。

## 3 访问参数

### 3.1 按层取

```python
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net[0].weight          # 第 0 层的权重，(256, 20)
net[0].bias            # 第 0 层的偏置，(256,)
net[2].weight          # 第 2 层的权重，(10, 256)
```

**`net[0]` 拿到的是那个层对象**，然后按属性取权重与偏置。

**名字从哪来**：`nn.Linear` 内部把权重注册成 `"weight"`、偏置注册成 `"bias"`，
所以能直接按这两个名字取。自定义模块里自己起的名字（`self.hidden`）也是同样道理。

### 3.2 遍历全部参数

| 方法 | 返回什么 |
|---|---|
| `net.parameters()` | 所有参数的迭代器（只有张量）|
| `net.named_parameters()` | `(名字, 张量)` 的迭代器 |
| `net.state_dict()` | 名字到张量的字典（保存模型用）|
| `net.modules()` | 所有子模块（含自己）|

**例**：

```python
for name, p in net.named_parameters():
    print(name, tuple(p.shape))
# 输出类似：
# 0.weight (256, 20)
# 0.bias (256,)
# 2.weight (10, 256)
# 2.bias (10,)
```

**名字是按"层在 Sequential 里的下标"拼出来的**，这与 `self.hidden = ...` 的名字
在自定义模块里会变成 `hidden.weight`。

### 3.3 按名字筛选

书上用 `collect_params('.*weight')` 按正则筛。torch 里没有这个方法，标准做法是：

```python
for name, p in net.named_parameters():
    if name.endswith("weight"):
        ...
```

**这一招在实践里很有用**：想给权重和偏置设不同的 `weight_decay` 时，
就是按名字把它们分到两个参数组里。

## 4 初始化参数

### 4.1 默认初始化

`nn.Linear` 创建时就完成了初始化（B11 的 5.2 节实测过：`nn.Linear(100, 50)` 的权重
在 ±0.1 之内，用的是 `kaiming_uniform_`）。

**所以 torch 不需要调用 `initialize()`**——那是 Gluon 的 API。

### 4.2 手动改

三种写法：

```python
# 写法一：in-place 修改（推荐）
with torch.no_grad():
    net[0].weight.normal_(0, 0.01)      # 均值 0、标准差 0.01 的正态
    net[0].weight.fill_(1.0)            # 全部填 1
    nn.init.xavier_uniform_(net[0].weight)

# 写法二：直接换掉 .data
net[0].weight.data = torch.ones_like(net[0].weight)

# 写法三：copy_（形状必须一致）
with torch.no_grad():
    net[0].weight.copy_(torch.ones_like(net[0].weight))
```

**为什么必须包 `no_grad()`**：权重是叶子张量，在开启梯度的模式下原地改会报
`RuntimeError: a leaf Variable that requires grad is being used in an in-place operation`
（B04 的 5.5 节讲过）。

**写法二为什么不用 `no_grad`**：`weight.data = ...` 换的是 `.data` 这个属性的指向，
绕过了原地操作的限制。但它有副作用——优化器持有的仍然是原来那个张量对象，
如果换掉 `.data` 指向的存储，优化器可能还在更新旧的那块。**平时用写法一**。

### 4.3 自定义初始化

`nn.init` 里没有的方法，自己写一个函数就行：

```python
def my_init(t):
    # 一半概率是 0，另一半是 [-10,-5] 与 [5,10] 上的均匀分布
    with torch.no_grad():
        t.uniform_(-10, 10)
        t.mul_((t.abs() >= 5).float())
    return t

my_init(net[0].weight)
```

**torch 不需要继承一个 `Initializer` 类**——那套是 Gluon 的设计。torch 里初始化就是
"拿一个张量、原地改它"，写成普通函数即可。

**`nn.init` 里的常见函数**（B11 用过两个）：

| 函数 | 分布 |
|---|---|
| `nn.init.normal_(t, mean, std)` | 正态 |
| `nn.init.uniform_(t, a, b)` | 均匀 |
| `nn.init.constant_(t, v)` | 常数 |
| `nn.init.xavier_uniform_(t)` | Xavier 均匀 |
| `nn.init.kaiming_uniform_(t)` | He 均匀 |

**它们都是原地操作**，返回的还是传入的那个张量。

## 5 共享参数

**torch 里共享参数只有一种做法：让两个地方引用同一个 `nn.Module` 对象。**

```python
class TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(8, 8)
        self.other = nn.Linear(8, 10)

    def forward(self, x):
        x = nn.functional.relu(self.shared(x))     # 第一次
        x = nn.functional.relu(self.shared(x))     # 第二次，同一个层
        return self.other(x)
```

**验证确实是共享的**：

```python
print(net.shared is net.shared)                          # True，当然是同一个
print(sum(1 for _ in net.parameters()))                  # 2 + 2 = 4 个张量，不是 6
```

`parameters()` **对同一个模块只遍历一次**，不会因为前向里用了两次就多出一份。

**反向时的梯度**：两处用到的梯度会累加到同一份 `.grad` 上，这与 B04 讲的
"梯度是累加的"一致。

## 6 延后初始化

书上 4.3 讲了一件 Gluon 特有的事：创建层时**不指定输入维度**，第一次前向时才推断。

```python
# Gluon
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))   # 只给输出维度
net.add(nn.Dense(10))
net.initialize()        # 这时什么都没初始化
net(X)                  # 第一次前向时才创建参数 (256, 20)
```

**torch 没有这个行为**。`nn.Linear(20, 256)` 创建时就要求两个维度都给，
参数立刻存在，可以立刻访问与修改：

```python
layer = nn.Linear(20, 256)
print(layer.weight.shape)     # (256, 20)，马上就能看
```

**torch 里唯一类似的东西是 `nn.LazyLinear`**：只给输出维度，第一次前向时推断输入维度。
它内部的实现机制（`UninitializedParameter`）比书的描述更复杂，这一章不用它。

**所以这一节在 torch 里只有一条要点**：书上的"延后初始化"是 Gluon 的设计，
torch 里创建层就要给全维度，参数立刻可访问。这一条在 B06 就能看出来——
`nn.Linear(2, 1)` 的 `weight` 从来都是直接取得到的。

## 7 实现会踩的坑

| 坑 | 症状 | 原因 |
|---|---|---|
| `__init__` 里忘了 `super().__init__()` | `net.parameters()` 是空的，训练不更新 | 子模块没被登记（1.1）|
| 调 `net.forward(x)` 而不是 `net(x)` | 能算，但 hook 不生效 | 绕过了 `__call__`（1.2）|
| 把层存进普通 list | `parameters()` 收不到它 | list 不是 `nn.Module`，不会登记（2.1.1）|
| 以为普通张量属性会被训练 | 它一直不变 | 只有 `nn.Parameter` 进优化器（2.3）|
| 用 `weight.data = ...` 换存储 | 优化器还在更新旧的 | `.data` 换了指向（4.2）|
| 改参数忘了 `no_grad()` | `RuntimeError: a leaf Variable ...` | 叶子张量不能原地改（4.2）|
| 以为共享参数要特殊 API | — | 同一个层对象用两次就够（5）|
| 期待 `nn.Linear(256)` 只给输出维度 | `TypeError` | torch 没有延后初始化（6）|

## 命令速查

| 命令 | 作用 | 讲义哪一节 |
|---|---|---|
| `class X(nn.Module)` | 自定义模块的写法 | 1.1 |
| `super().__init__()` | 必须调用，否则参数不登记 | 1.1 |
| `def forward(self, x)` | 定义前向 | 1.1 |
| `self.add_module(name, m)` | 显式登记子模块（名字动态生成时用）| 1.3 |
| `self.children()` | 遍历子模块，按添加顺序 | 1.3 |
| `nn.Parameter(t)` | 把张量变成可训练参数 | 2.3 |
| `self.register_buffer(name, t)` | 登记不训练但要保存的张量 | 2.3 |
| `net[0].weight` / `.bias` | 按层下标取参数 | 3.1 |
| `net.named_parameters()` | 遍历全部参数与名字 | 3.2 |
| `net.state_dict()` | 名字到张量的字典，保存模型用 | 3.2 |
| `nn.init.normal_(t, mean, std)` | 原地改成正态分布 | 4.3 |
| `nn.init.xavier_uniform_(t)` | 原地做 Xavier 初始化 | 4.3 |
| `nn.ModuleList([...])` | 装一组层的容器，会登记子模块 | 2.1.1 |
| `nn.functional.relu(x)` | 函数式的 ReLU（不建模块）| 1.1 |

## 自测题

1. 自定义一个模型要写哪两个方法？`super().__init__()` 忘了会怎样？
2. 为什么调用时写 `net(x)` 而不是 `net.forward(x)`？
3. 想把两个层设置成共享参数，torch 里怎么做？
4. 一个张量要跟着训练更新、另一个不更新但要存在模型文件里。分别用什么？
5. `net.named_parameters()` 输出什么？拿它做"权重与偏置分开设 weight_decay"怎么做？
6. 为什么改参数要包 `no_grad()`？
7. 书上的"延后初始化"在 torch 里对应什么？

## 答案（做完再看）

**1.** `__init__`（声明层）与 `forward`（定义前向）。
忘了 `super().__init__()` 的话，子模块不会被登记，`net.parameters()` 返回空，
优化器拿不到参数，训练时参数不更新——而且模型能创建、能前向，报错很晚。

**2.** `net(x)` 走的是 `nn.Module.__call__`，它会在调 `forward` 前后处理 hook 与内部状态。
直接调 `forward` 会跳过这些，hook 不生效。

**3.** 让两个地方引用**同一个 `nn.Module` 对象**。例如在前向里对 `self.shared` 调用两次，
或者把同一个层对象赋给两个属性。

**4.** 会训练更新的用 `nn.Parameter`；不训练但要进 `state_dict` 的用
`self.register_buffer("name", tensor)`。普通张量属性两样都不满足。

**5.** `(名字, 张量)` 的迭代器，名字是按模块层级拼出来的（`0.weight`、`hidden.bias`）。
按 `name.endswith("weight")` 分成两组，分别传给优化器的两个参数组。

**6.** 权重是叶子张量，在开启梯度的模式下原地改会报错。`no_grad()` 关掉梯度记录后就能原地改。

**7.** torch 里没有延后初始化——`nn.Linear` 创建时就要给全输入与输出维度，参数立刻存在、
立刻能访问。唯一类似的是 `nn.LazyLinear`，本章不用。
