# B13 口试题

关书作答。答不上来先想，说"不知道"我就讲。

**答题要求**：用中文和代码描述即可，**公式不必写 LaTeX**。

题目分两档：

- **必会**（Q1、Q2、Q4、Q5）：写后面那些复杂网络时天天用
- **了解**（Q3、Q6、Q7、Q8）：知道有这回事即可

## 必会

**Q1** 自己定义一个模型，要继承什么、写哪两个方法？`__init__` 里第一行必须是哪一句，
不写会怎样？

<details><summary>参考答点</summary>

继承 `nn.Module`，写 `__init__` 与 `forward`。

`__init__` 的第一行必须是 `super().__init__()`。不写的话，创建实例时就会报

```
AttributeError: cannot assign module before Module.__init__() call
```

报在 `self.hidden = nn.Linear(...)` 那一行——`nn.Module.__setattr__` 要往内部字典登记
子模块，而字典还没建出来。

</details>

**Q2** 调用时为什么写 `net(x)` 而不是 `net.forward(x)`？

<details><summary>参考答点</summary>

`net(x)` 走的是 `nn.Module.__call__`，它会在调 `forward` 前后处理 forward hook
（调试与可视化用的回调）与内部状态。

直接调 `forward` 会绕过这些。一般也能算对，但 hook 不生效，所以约定写 `net(x)`。

</details>

**Q4** 有三种张量属性：`nn.Parameter`、普通张量、`register_buffer`。分别会不会被训练更新、
会不会出现在 `state_dict` 里？

<details><summary>参考答点</summary>

| 做法 | 训练更新 | 在 `state_dict` 里 |
|---|---|---|
| `self.p = nn.Parameter(torch.randn(3))` | 会 | 会 |
| `self.t = torch.randn(3)` | 不会 | 不会 |
| `self.register_buffer("b", torch.randn(3))` | 不会 | **会** |

**要训练**用 `nn.Parameter`。**不训练但要跟着模型一起保存**（例如 BatchNorm 的均值方差）
用 `register_buffer`。普通张量属性两样都不满足。

</details>

**Q5** 想让两个层共享参数，torch 里怎么做？怎么验证确实是共享的？

<details><summary>参考答点</summary>

让两个地方引用**同一个 `nn.Module` 对象**。例如在 `forward` 里对 `self.shared` 调用两次，
或者把同一个层对象赋给两个属性。

**验证**：`len(list(net.parameters()))` 的个数。一个 `Linear(8, 8)` 有 2 个张量，
用两次如果参数个数还是 2（而不是 4），说明共享成功。

反向传播时两处的梯度会累加到同一份 `.grad` 上。

</details>

## 了解

**Q3** `nn.Sequential` 缺了什么能力，为什么还需要自己写模块？

<details><summary>参考答点</summary>

`nn.Sequential` 只能表达"一层接一层"。三种情况它做不到：

1. 层之间不是简单串联（有分支、有跳连）
2. 同一个层用两次（共享参数）
3. 前向里有控制流（`while`、`if`）

</details>

**Q6** 为什么改模型参数要包 `with torch.no_grad():`？

<details><summary>参考答点</summary>

权重是叶子张量、且 `requires_grad=True`。在开启梯度的模式下原地改它会报

```
RuntimeError: a leaf Variable that requires grad is being used in an in-place operation
```

`no_grad()` 关掉梯度记录之后就能原地改。

</details>

**Q7** `net.named_parameters()` 输出什么？怎么拿它把权重与偏置分开处理？

<details><summary>参考答点</summary>

输出 `(名字, 张量)` 的迭代器。名字按模块层级拼出来，例如 `0.weight`、`0.bias`、
`hidden.weight`。

按 `name.endswith("weight")` 分成两组，分别传给优化器的两个参数组——这是"给权重与偏置
设不同 `weight_decay`"的标准做法。

</details>

**Q8** 书上的"延后初始化"在 torch 里对应什么？

<details><summary>参考答点</summary>

**torch 里没有延后初始化。** `nn.Linear` 创建时就要求给全输入与输出维度，
参数立刻存在、立刻能访问。

书上的 `nn.Dense(256)` 只给输出维度、第一次前向时才推断输入维度——那是 Gluon 的设计。
torch 里唯一类似的是 `nn.LazyLinear`，本章不用。

</details>

## 评分

- **必会**四题（Q1、Q2、Q4、Q5）全对才算通过
- **了解**四题答不上来不扣分
- Q1 与 Q4 是写复杂网络时天天用的
