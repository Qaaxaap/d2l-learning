# B14 阅读任务书

## 读什么

| 材料 | 位置 | 预估 |
|---|---|---|
| 纸质书 | 4.4 自定义层、4.5 读取和存储、4.6 GPU 计算 | 30 分钟 |
| 电子版 | [5.4 自定义层](https://zh.d2l.ai/chapter_deep-learning-computation/custom-layer.html)、[5.5 读写文件](https://zh.d2l.ai/chapter_deep-learning-computation/read-write.html)、[5.6 GPU](https://zh.d2l.ai/chapter_deep-learning-computation/use-gpu.html) | 30 分钟 |
| 讲义 | `notes/b14-custom-layer-device.md` | 35 分钟 |

## 读的时候要能回答

1. 写一个含参数的自定义层，`__init__` 里必须用哪个东西包住参数？不包会怎样？
2. 自己写的层算 `x @ W`，`nn.Linear` 内部算 `x @ W.T`。两者的 `W` 形状分别是多少？
3. 存模型的标准做法是存什么？为什么不推荐 `torch.save(net)`？
4. `load_state_dict` 会检查什么？对不上说明什么？
5. 把模型放到 GPU 用哪一句？数据呢？
6. `.to("cuda")` 是原地改还是返回新张量？
7. 为什么 torch 不允许 CPU 张量与 GPU 张量直接相加？
8. GPU 的运算一定比 CPU 快吗？

## 顺序

4.4（自定义层）与 B13 是同一件事，区别只是"写层"与"写模型"。
读的时候留意 `self.params.get(...)` 与 torch 的 `nn.Parameter` 的对应。

4.5 与 4.6 是两条独立的线，先读哪条都行。4.6 的代码在本机能跑
（有 CUDA 的显卡），可以边读边试。

## 三处要留意

**4.4 的 `CenteredLayer` 减的是"全部元素的均值"**，不是"每个样本减自己的均值"。
书上写的是 `x - x.mean()`，对一个 `(batch, features)` 的输入，减掉的是一个标量。

**4.5 的 `save_parameters` 是 Gluon 的 API**。torch 里对应 `torch.save(net.state_dict(), ...)`
与 `load_state_dict`。两者的语义一样（存名字到张量的字典），名字不同。

**4.6 的 GPU 部分与这台机器能对上**：有 RTX 4070 SUPER。但讲义 3.5 节有一条实测：
2000×2000 的矩阵乘 GPU 比 CPU 还慢——规模小时 CUDA 的固定开销压过计算速度。
读的时候知道"GPU 不是越快越好"就够了。
