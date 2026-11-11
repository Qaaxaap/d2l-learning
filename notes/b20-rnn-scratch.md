# B20 RNN 从零实现与 BPTT

B19 讲完了数据管道与 RNN 的前向公式。本章把两部分接起来，训练一个能生成文本的模型。

本章要解决四件事：

1. 把字符索引变成网络能吃的输入。
2. 写出完整的前向。
3. 理解梯度沿时间传播时为什么会爆炸或消失。
4. 让训练跑起来。

## 与书对应

| 讲义小节 | 纸质书（第一版） | 电子版（第二版） |
|---|---|---|
| 1 one-hot 输入 | 6.4 的"one-hot向量" | [8.5](https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html) 的同一处 |
| 2 参数初始化 | 6.4 的"初始化模型参数" | 8.5 的同一处 |
| 3 前向 | 6.4 的"定义模型" | 8.5 的同一处 |
| 4 预测函数 | 6.4 的"定义预测函数" | 8.5 的同一处 |
| 5 BPTT | 6.6 全节 | [8.7](https://zh.d2l.ai/chapter_recurrent-neural-networks/bptt.html) |
| 6 梯度裁剪 | 6.4 的"裁剪梯度" | 8.5 的同一处 |
| 7 困惑度 | 6.4 的"困惑度" | 8.5 的同一处 |
| 8 训练循环 | 6.4 的"定义模型训练函数" | 8.5 的同一处 |
| 9 实现会踩的坑 | 散在正文里，本节汇总 | 同上 |
| 自测题 | 无 | 无 |

**解锁点**：v1 的 6.4 标了"本函数已保存在 d2lzh 包中"，
涉及 `to_onehot`、`predict_rnn`、`grad_clipping`、`train_and_predict_rnn`。
本章要自己写。

## 1 one-hot 输入

### 1.1 问题：索引不能直接喂给矩阵乘

B19 的数据管道产出整数索引。一个索引是 Python 的 `int`，比如 7。

矩阵乘要的是向量。把索引 7 直接放进张量，得到的是一个标量。标量乘权重矩阵，
只会得到一个数，无法区分「第 7 个字符」与「第 7 个位置上的数值 7」。

整数还有一个问题：索引有大小关系。7 比 6 大，但「第 7 个字符」不比
「第 6 个字符」大。直接用索引会让模型学到不存在的关系。

### 1.2 one-hot 的做法

**one-hot** 把索引 $i$ 变成一个长度为词表大小的向量。只有第 $i$ 位是 1，其余为 0。

```
索引 2，词表大小 5  →  [0, 0, 1, 0, 0]
```

这样处理有三个效果：

- 所有字符在输入空间里等距。
- 没有大小关系。
- 任意两个不同字符的向量内积为 0。

torch 提供现成函数：`torch.nn.functional.one_hot(t, num_classes)`。
它接受一维整型张量，返回 `(len(t), num_classes)`。

### 1.3 从一个批量到多个时间步

数据管道产出的 `X` 形状是 `(batch_size, num_steps)`。网络要的是**每个时间步一个**输入矩阵。

转换顺序：

1. 按时间步切开，每步得到形状 `(batch_size,)` 的索引向量。
2. 对每步做 one-hot，得到 `(batch_size, vocab_size)`。
3. 一共得到 `num_steps` 个这样的矩阵。

```
索引矩阵 (batch, num_steps)
   ↓ 按时间步拆开
每步索引 (batch,)
   ↓ one-hot
每步输入 (batch, vocab_size)      共 num_steps 个
```

**注意拆分的维度**。`X` 是 `(batch, num_steps)`，时间步在**第 1 维**。
按第 1 维切开，得到 `num_steps` 个 `(batch,)`。

如果按第 0 维切，得到的是 `batch` 个 `(num_steps,)`，含义完全相反。

### 1.4 代价

one-hot 向量的长度等于词表大小。词表 5 万时，每个字符占一个 5 万维向量，
其中 49999 个位置是 0。

两个后果：

- 存储浪费。一个批次 `(32, 10000)` 的 one-hot 输入是 3.2 亿个元素。
- 输入层参数多。`(vocab_size, hidden)` 的权重矩阵，词表大时它最大。

B28 的词向量解决这个问题：把字符映射到几十到几百维的稠密向量。

## 2 参数初始化

### 2.1 五组参数

RNN 前向要五组参数。形状与 B19 的 3.3 节一致。

| 参数 | 形状 | 说明 |
|---|---|---|
| `W_xh` | `(vocab_size, num_hiddens)` | 输入到隐藏 |
| `W_hh` | `(num_hiddens, num_hiddens)` | 上一步隐藏到这一步隐藏 |
| `b_h` | `(num_hiddens,)` | 隐藏层偏置 |
| `W_hq` | `(num_hiddens, vocab_size)` | 隐藏到输出 |
| `b_q` | `(vocab_size,)` | 输出层偏置 |

注意 `W_xh` 的第一维是词表大小而不是嵌入维。字符级模型的输入是 one-hot，
所以输入维就是词表大小。

这五组必须是 `nn.Parameter`。不包起来优化器看不到它们，参数不会更新。

### 2.2 为什么乘 0.01

书上的初始化是 `scale=0.01` 的正态分布。偏置初始化为 0。

`W_hh` 要在时间步之间反复相乘。初始值大时，连乘几步就爆炸成 `nan`。

举一个极端的例子。设 `W_hh` 的每个元素都是 1，隐藏状态每步乘一次，
序列长度 100 时结果放大了 $1^{100}$ 倍。元素是 1.1 时结果放大 $1.1^{100} \approx 13780$ 倍。

乘 0.01 让连乘的结果在前若干步内保持数值稳定。

```python
W_hh = torch.randn(num_hiddens, num_hiddens) * 0.01
```

偏置不需要缩放。偏置不参与连乘。

## 3 前向

### 3.1 接口

```python
def rnn_forward(inputs: list[torch.Tensor], state: tuple[torch.Tensor],
                params: tuple[torch.Tensor, ...]):
    """返回 (outputs, new_state)。"""
```

三个参数：

- `inputs`：长度为 `num_steps` 的列表，每个元素形状 `(batch, vocab_size)`。
- `state`：含一个张量的元组，张量形状 `(batch, num_hiddens)`。
- `params`：五组参数的元组。

返回两个值：`outputs` 是长度 `num_steps` 的列表，每个元素形状
`(batch, vocab_size)`；`new_state` 是更新后的隐藏状态。

**状态为什么用元组包一层**：书上的写法是为了以后支持有多个隐藏状态的 RNN。
LSTM 有 `(H, C)` 两个状态。用元组的形式，换模型时接口不用改。

### 3.2 循环体

对每个时间步执行两步。

1. 更新隐藏状态：`H = tanh(X @ W_xh + H @ W_hh + b_h)`。
2. 算输出：`Y_t = H @ W_hq + b_q`。

`H = torch.tanh(X @ W_xh + H @ W_hh + b_h)`。

`H` 在每步之间传递。第 $t$ 步的 `H` 由第 $t-1$ 步的 `H` 与第 $t$ 步的 `X` 共同决定。

### 3.3 输出是 logits

`Y_t` 没有经过 softmax。它是未归一化的分数。

`nn.CrossEntropyLoss` 内部会做 log-softmax 与负对数似然。
传 logits 进去正好。

**在外面再做一次 softmax 会出错**。这一条在 B14 与 B16 都提过，这里再强调一次：
`CrossEntropyLoss(input, target)` 的 `input` 是 logits。

## 4 预测函数

### 4.1 做什么

给定一段前缀，逐字符生成后续文本。

```
前缀 "分开"
  第 1 步：输入 "分"，预测下一个字符 → "开"
  第 2 步：输入 "开"，预测 → ?
  ...
```

### 4.2 三个要点

**要点一：批量是 1。**

预测时一次只生成一条文本。初始隐藏状态形状 `(1, num_hiddens)`，
不是 `(batch_size, num_hiddens)`。

**要点二：前缀阶段不采信模型的预测。**

前几步把前缀里的字符**强制**作为输入，忽略模型给出的预测。

这样模型从给定的上下文出发，不会一开始就偏离。如果采信预测，
第一步错后面就全错，前缀失去引导作用。

**要点三：前缀结束后用上一步的预测作为输入。**

每步从 `Y` 里取 `argmax`，把这个索引作为下一步的输入。这叫**自回归**。

取索引写 `torch.argmax(Y, dim=1)`。`Y` 形状 `(1, vocab_size)`，
沿第 1 维取最大值，得到形状 `(1,)`。

### 4.3 argmax 的局限

贪心取每步概率最大的字符。这样生成的文本容易重复。

原因：一旦某个字符被选中，下一步的概率分布会强化它，容易陷入循环。

改进办法有温度采样与 beam search。B22 会讲 beam search。

## 5 BPTT：梯度怎么沿时间传播

### 5.1 计算图

RNN 展开后，每个时间步对应计算图里的一层。损失是各步损失之和（或平均）：

$$L = \frac{1}{T}\sum_{t=1}^{T} \ell(\boldsymbol{o}_t, y_t)$$

反向传播要沿时间反着走。这叫**通过时间反向传播**（BPTT，backpropagation through time）。

### 5.2 输出层的梯度

$L$ 通过每个 $\boldsymbol{o}_t$ 依赖 $\boldsymbol{W}_{hq}$。按链式法则求和：

$$\frac{\partial L}{\partial \boldsymbol{W}_{hq}} = \sum_{t=1}^{T} \frac{\partial L}{\partial \boldsymbol{o}_t} \boldsymbol{h}_t^\top$$

每一步的梯度都要算，全部加起来。这是因为 $T$ 个时间步共用同一个 $\boldsymbol{W}_{hq}$。

### 5.3 隐藏状态的梯度

从最后一步开始。$L$ 只通过 $\boldsymbol{o}_T$ 依赖最终隐藏状态：

$$\frac{\partial L}{\partial \boldsymbol{h}_T} = \boldsymbol{W}_{qh}^\top \frac{\partial L}{\partial \boldsymbol{o}_T}$$

对 $t < T$，$L$ 通过两条路依赖 $\boldsymbol{h}_t$：
一条经过 $\boldsymbol{o}_t$，一条经过 $\boldsymbol{h}_{t+1}$。

$$\frac{\partial L}{\partial \boldsymbol{h}_t}
= \boldsymbol{W}_{hh}^\top \frac{\partial L}{\partial \boldsymbol{h}_{t+1}}
+ \boldsymbol{W}_{qh}^\top \frac{\partial L}{\partial \boldsymbol{o}_t}$$

**第一项**来自「下一步的隐藏状态依赖这一步」。**第二项**来自「这一步的输出依赖这一步」。

两项相加就是两个方向来的梯度之和。

把这个递归展开，得到通项：

$$\frac{\partial L}{\partial \boldsymbol{h}_t}
= \sum_{i=t}^{T} \left(\boldsymbol{W}_{hh}^\top\right)^{T-i}
\boldsymbol{W}_{qh}^\top \frac{\partial L}{\partial \boldsymbol{o}_{T+t-i}}$$

### 5.4 梯度为什么会爆炸或消失

上式的关键部分是 $\left(\boldsymbol{W}_{hh}^\top\right)^{T-i}$。这是一个**矩阵的连乘**。

考虑 $\boldsymbol{W}_{hh}$ 的特征值：

- 特征值都小于 1：连乘 $T-t$ 次后趋近 0。**梯度消失**，远处的影响传不回来。
- 特征值都大于 1：连乘后指数增长。**梯度爆炸**，几步就变成 `nan`。

序列越长，$T-t$ 越大，问题越严重。这解释了 B19 提到的「RNN 学不到长距离依赖」。

这个现象与 B11 讲的前馈网络梯度消失是同一件事。
区别在于 RNN 的连乘次数等于时间步数，而前馈网络等于层数。

## 6 梯度裁剪

### 6.1 公式

把所有参数的梯度拼成一个向量 $\boldsymbol{g}$。设阈值为 $\theta$。裁剪后的梯度是：

$$\boldsymbol{g} \leftarrow \min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right) \boldsymbol{g}$$

裁剪后的范数不超过 $\theta$，方向不变。

### 6.2 为什么必须整体缩放

**整体缩放**：所有元素乘同一个系数。只改变长度，不改变方向。

**逐元素裁剪**：每个元素除以各自的绝对值。方向被改变，优化方向也被破坏。

举例。梯度是 `[3, 4]`，阈值是 1。

- 整体缩放：乘 $1/5 = 0.2$，得到 `[0.6, 0.8]`。方向仍是 `[3, 4]`。
- 逐元素裁剪：每个分量压到 1，得到 `[1, 1]`。方向完全变了。

所以要整体缩放。

### 6.3 实现

计算所有参数梯度的总范数，超过阈值时按比例缩放。

```python
norm = torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None))
if norm > theta:
    for p in params:
        if p.grad is not None:
            p.grad.mul_(theta / norm)
```

torch 提供现成实现：`torch.nn.utils.clip_grad_norm_(params, theta)`。
实际项目里用它。但要知道它在做什么。

## 7 困惑度

**困惑度**（perplexity）是语言模型的标准评价指标：

$$\text{perplexity} = \exp(\text{cross-entropy loss})$$

三种情况的结果：

| 情况 | 交叉熵 | 困惑度 |
|---|---|---|
| 完美预测（概率 1）| 0 | 1 |
| 均匀瞎猜（$1/V$）| $\ln V$ | $V$ |
| 最差（概率 0）| $\infty$ | $\infty$ |

**基线是词表大小**。词表 1000 时，困惑度 1000 等于瞎猜。
有效模型的困惑度必须小于词表大小。

**为什么用困惑度而不用交叉熵**：困惑度的单位是「等效的候选词个数」。
困惑度 10 的意思是「模型平均在 10 个候选里犹豫」。这比「损失 2.3」直观。

代码里写 `torch.exp(loss)`，`loss` 是该周期的平均交叉熵。

## 8 训练循环

### 8.1 与前面章节的差别

训练循环的结构与 B12 相同：取批量、前向、算损失、反向、更新。
本章多三件事。

1. 每个批量要处理隐藏状态的传递。
2. 每次更新前裁剪梯度。
3. 用困惑度报告。

### 8.2 两种采样的隐藏状态处理

**随机采样**：每个小批量开始前，隐藏状态重置为全 0。

理由见 B19 的 5.3 节：相邻两个批量在原始序列上不相邻，它们的隐藏状态没有连续性。

**相邻采样**：迭代周期开始时初始化一次隐藏状态。
之后每个批量开始前，把隐藏状态从计算图里切断。

切断用 `detach()`：

```python
state = tuple(s.detach() for s in state)
```

**`detach()` 做什么**：保留数值，切断梯度。反向传播到它为止，不再往前。

**不切断会怎样**：计算图沿着批量之间的连接一直增长。
显存越用越多，训练越来慢，最后内存不足。

### 8.3 输出的拼接与标签

`rnn_forward` 返回 `num_steps` 个 `(batch, vocab_size)` 的输出。
标签 `Y` 形状是 `(batch, num_steps)`。两者的形状对不上。

处理步骤：

1. 把 `num_steps` 个输出按第 0 维拼接，得到 `(num_steps * batch, vocab_size)`：
   `outs = torch.cat(outputs, dim=0)`。
2. 把 `Y` 转置成 `(num_steps, batch)`，再展平成 `(num_steps * batch,)`：
   `targets = Y.T.reshape(-1)`。
3. 两者一一对应，可以直接算交叉熵。

**顺序不能错**。如果先展平 `Y` 再转置，标签与输出的对应关系就乱了，
模型会学到错误的目标。

### 8.4 完整流程

对每个迭代周期：

1. 若用相邻采样，初始化隐藏状态。
2. 对每个批量：
   1. 若用随机采样，隐藏状态重置为全 0；否则 `detach`。
   2. 把 `X` 转成 one-hot。
   3. 前向，得到输出与新的隐藏状态。
   4. 拼接输出、整理标签、算交叉熵。
   5. 反向。
   6. 裁剪梯度。
   7. 更新参数。
3. 每若干周期打印困惑度并生成示例文本。

## 9 实现会踩的坑

| 坑 | 症状 | 原因 |
|---|---|---|
| one-hot 的维度搞反 | 形状不匹配 | `(batch, num_steps)` 按时间步（第 1 维）拆开（1.3）|
| `Y` 忘了过 softmax 之外的检查 | 损失不降 | `Y` 是 logits，`CrossEntropyLoss` 内部做 softmax（3.3）|
| 忘了裁剪梯度 | 损失变 `nan` | 梯度爆炸（6）|
| 裁剪写成逐元素 | 训练变慢、方向被改 | 要整体按比例缩放（6.2）|
| 相邻采样没 `detach` | 越训越慢、显存涨 | 计算图无限增长（8.2）|
| 参数没用 `nn.Parameter` | 返回结果但不更新 | B14 讲过（2.1）|
| 预测时批量当成 `num_steps` | 形状不匹配 | 一次只生成一条，批量是 1（4.2）|
| `W_hh` 初始化成 1 | 几步就 `nan` | 连乘放大（2.2）|
| 输出与标签的拼接顺序弄错 | 损失不降 | 先拼输出，再把 `Y` 转置展平（8.3）|

## 命令速查

| 命令 | 作用 | 讲义哪一节 |
|---|---|---|
| `torch.nn.functional.one_hot(t, num_classes)` | 索引转 one-hot | 1.2 |
| `torch.randn(shape) * 0.01` | 小初始值 | 2.2 |
| `nn.Parameter(tensor)` | 包成可训练参数 | 2.1 |
| `torch.tanh(x)` | RNN 激活 | 3.2 |
| `torch.argmax(y, dim=1)` | 取预测字符 | 4.2 |
| `torch.sqrt(sum((p.grad**2).sum() for p in params))` | 梯度范数 | 6.3 |
| `p.grad.mul_(scale)` | 按比例缩梯度 | 6.3 |
| `torch.nn.utils.clip_grad_norm_(params, theta)` | 现成的裁剪 | 6.3 |
| `s.detach()` | 切断计算图 | 8.2 |
| `torch.exp(loss)` | 困惑度 | 7 |
| `torch.cat(outs, dim=0)` | 拼接各步输出 | 8.3 |
| `Y.T.reshape(-1)` | 标签转置展平 | 8.3 |

## 自测题

1. 说明索引不能直接输入矩阵乘的两个原因，以及 one-hot 解决的两件事。
2. 写出 one-hot 的代价，并指出后来的哪项技术解决了它。
3. 说明 `W_hh` 的初始值乘 0.01 的原因。
4. 判断 RNN 的输出 `Y` 是否经过 softmax，并写出损失函数需要的输入类型。
5. 说明预测函数在前缀阶段不采信模型预测的原因。
6. 写出隐藏状态梯度的递推公式，说明两项各来自哪里。
7. 说明梯度爆炸与消失的根源，并解释为什么序列越长越严重。
8. 写出梯度裁剪的公式，并说明为什么不能逐元素裁剪。
9. 写出困惑度的定义与基线。
10. 说明相邻采样训练时对隐藏状态 `detach` 的原因。
11. 写出输出与标签的拼接顺序，并说明顺序弄错的后果。

## 答案（做完再看）

**1.** 索引是整数，有大小关系，而字符之间没有。one-hot 让所有字符等距、消除大小关系。
另外它还让两个不同字符的向量内积为 0。

**2.** 向量长度等于词表大小，词表大时存储与输入层参数都很贵。
B28 的词向量把它变成几十到几百维的稠密向量。

**3.** `W_hh` 在时间步之间反复相乘。初始值大时连乘几步就爆炸。
乘 0.01 让连乘结果在前若干步内保持稳定。

**4.** 没有。`Y` 是 logits。`nn.CrossEntropyLoss` 需要 logits，它内部做 log-softmax。

**5.** 前缀是要强制模型接受的上下文。采信预测会让模型一开始就偏离，
前缀失去引导作用。

**6.** $\frac{\partial L}{\partial \boldsymbol{h}_t}
= \boldsymbol{W}_{hh}^\top \frac{\partial L}{\partial \boldsymbol{h}_{t+1}}
+ \boldsymbol{W}_{qh}^\top \frac{\partial L}{\partial \boldsymbol{o}_t}$。

第一项来自「下一步的隐藏状态依赖这一步」。第二项来自「这一步的输出依赖这一步」。

**7.** 根源是 $\boldsymbol{W}_{hh}^\top$ 的连乘。
特征值小于 1 时连乘趋近 0（消失），大于 1 时指数增长（爆炸）。
序列越长连乘次数越多，问题越严重。

**8.** $\boldsymbol{g} \leftarrow \min(\theta/\|\boldsymbol{g}\|, 1)\cdot\boldsymbol{g}$。

逐元素裁剪会让不同元素缩不同比例，改变梯度方向。
整体缩放所有元素乘同一系数，只改长度。

**9.** 困惑度是 $\exp(\text{交叉熵})$。基线是词表大小。
均匀瞎猜时交叉熵是 $\ln V$，困惑度就是 $V$。

**10.** 相邻采样的隐藏状态跨批量传递。不 `detach` 时，
计算图沿所有历史批量累积，显存持续增长。
`detach` 保留数值、切断梯度，让每个批量只看当前序列。

**11.** 先把 `num_steps` 个输出按第 0 维拼接，再把 `Y` 转置并展平。
顺序弄错会让标签与输出的对应关系错位，模型学到错误的目标。
