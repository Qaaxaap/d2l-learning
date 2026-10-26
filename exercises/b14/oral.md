# B14 口试题

关书作答。答不上来先想，说"不知道"我就讲。

**答题要求**：用中文和代码描述即可，**公式不必写 LaTeX**。

题目分两档：

- **必会**（Q1、Q2、Q3、Q5）：后面写卷积网络与保存模型时天天用
- **了解**（Q4、Q6、Q7、Q8）：知道有这回事即可

## 必会

**Q1** 写一个含参数的自定义层。`__init__` 里必须用哪个东西包住参数？
不包会怎样？这个错会报错吗？

<details><summary>参考答点</summary>

必须用 `nn.Parameter` 包住：

```python
self.weight = nn.Parameter(torch.randn(in_units, units))
```

不包的话（直接写 `self.weight = torch.randn(...)`），它是普通张量属性：
`net.parameters()` 收不到它，优化器不会更新它。

**不报错**——模型能创建、能前向、损失能下降（别的参数在学），
只是这一个参数一直保持初始值。属于"静默出错"。

</details>

**Q2** 自己写的层算 `x @ W`，而 `nn.Linear` 内部算 `x @ W.T`。
两者的 `W` 形状分别是多少？

<details><summary>参考答点</summary>

- **自己写的**：`W` 形状 `(in_units, units)`，因为算的是 `x @ W`
- **`nn.Linear(in, out).weight`**：形状 `(out, in)`，因为算的是 `x @ W.T`

两者都对，只是约定不同。混用会报形状错。

</details>

**Q3** 存模型的标准做法是存什么？为什么不推荐 `torch.save(net)`？

<details><summary>参考答点</summary>

存 **`net.state_dict()`**（名字到张量的字典），加载用
`net.load_state_dict(torch.load(path))`。

不推荐 `torch.save(net)` 的两条理由：

1. **绑定代码结构**：`torch.load` 时要能 import 到定义这个类的模块。
   换目录、改了类名就读不出来
2. **参数与结构混在一起**：想复用参数、或只比较结构，都不方便

</details>

**Q5** 把模型放到 GPU 用哪一句？数据呢？为什么 `torch` 不允许 CPU 与 GPU 张量直接相加？

<details><summary>参考答点</summary>

模型：`net = net.to(device)`。数据：`X = X.to(device)`。

不允许混合相加会报：

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

**为什么不允许自动搬**：CPU 与 GPU 之间的传输很慢，自动搬会让性能问题难以定位。
显式搬迫使人知道数据在哪。

</details>

## 了解

**Q4** `load_state_dict` 会检查什么？检查出对不上说明什么？

<details><summary>参考答点</summary>

检查**键名**与每个张量的**形状**。

对不上说明模型结构与保存时不同——加了层、改了维度、或者属性名变了。
这是好事：能立刻发现"结构变了但还在用旧参数"，而不是算出一堆无意义的数。

</details>

**Q6** `.to("cuda")` 是原地改还是返回新张量？设备相同时呢？

<details><summary>参考答点</summary>

**返回新张量**（设备不同的时候）。原来的张量不变。

**设备相同时返回 `self`**——同一个对象，不复制。

这一条 B06 的 5.1.1.2 节实测过：`.to("cuda")` 的结果不是叶子张量（带 `grad_fn`），
而 `.to("cpu")` 是常量操作，返回原张量。

</details>

**Q7** `CenteredLayer` 里 `x - x.mean()` 减的是哪个均值？

<details><summary>参考答点</summary>

**全部元素的均值**（一个标量），不是每个样本减自己的均值。

输入 `(batch, features)` 时，`x.mean()` 把所有样本所有特征加起来再平均，
得到零维张量，然后广播减掉。

要逐样本中心化得写 `x - x.mean(dim=1, keepdim=True)`。

</details>

**Q8** GPU 的运算一定比 CPU 快吗？

<details><summary>参考答点</summary>

**不一定。** 实测（本机 RTX 4070 SUPER）：

| 运算 | CPU | GPU |
|---|---|---|
| 2000×2000 矩阵乘法 | 16 ms | 34.8 ms |

GPU 反而慢一倍。CUDA 调用有固定开销（内核启动、传输），规模小时压过计算速度。

**GPU 的优势在规模**：几万维的矩阵、卷积、批量图像，以及数据已经在显存里不用来回搬。
小任务上先用 CPU 跑通再切。

</details>

## 评分

- **必会**四题（Q1、Q2、Q3、Q5）全对才算通过
- **了解**四题答不上来不扣分
- Q1 的"不报错但静默出错"是最要紧的一条
