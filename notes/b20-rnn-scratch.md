# B20 RNN 从零实现与 BPTT

B19 把数据管道和 RNN 的前向公式讲完了。这一章把它们接起来：**训练一个能生成文本的字符级语言模型**。

## 与书对应

| 讲义小节 | 纸质书（第一版） | 电子版（第二版） |
|---|---|---|
| 1 one-hot 输入 | 6.4 的"one-hot向量" | [8.5](https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html) 的同一处 |
| 2 参数初始化 | 6.4 的"初始化模型参数" | 8.5 的同一处 |
| 3 模型前向 | 6.4 的"定义模型" | 8.5 的同一处 |
| 4 预测函数 | 6.4 的"定义预测函数" | 8.5 的同一处 |
| 5 梯度裁剪 | 6.4 的"裁剪梯度" | 8.5 的同一处 |
| 6 困惑度 | 6.4 的"困惑度" | 8.5 的同一处 |
| 7 训练函数 | 6.4 的"定义模型训练函数" | 8.5 的同一处 |
| 8 BPTT | 6.6 全节 | [8.7](https://zh.d2l.ai/chapter_recurrent-neural-networks/bptt.html) |
| 9 实现会踩的坑 | 散在正文里，本节汇总 | 同上 |
| 自测题 | 无 | 无 |

**解锁点**：v1 的 6.4 里 `to_onehot`、`predict_rnn`、`grad_clipping`、`train_and_predict_rnn`
都标了"本函数已保存在 d2lzh 包中方便以后使用"。这些都要自己写。

## 1 one-hot 输入

字符的索引是一个整数，不能直接喂给矩阵乘法。**one-hot** 把索引 $i$ 变成
一个长度为词表大小的向量，只有第 $i$ 位是 1：

```python
idx = 2
vec = torch.zeros(vocab_size)
vec[idx] = 1.0
```

torch 有现成的：

```python
torch.nn.functional.one_hot(idx_tensor, num_classes=vocab_size)   # (N,) -> (N, vocab)
```

**为什么用 one-hot 而不是直接喂整数**：整数索引有大小关系，但字符之间没有
"第 3 个字比第 2 个字大"这种语义。one-hot 让所有字符在输入空间里等距。

**代价**：向量长度等于词表大小，词表大时很浪费。词向量的做法（B28）解决这个问题。

**数据管道的输出是 `(batch, num_steps)` 的索引**，要把它变成
`num_steps` 个 `(batch, vocab_size)` 的矩阵：

```
索引矩阵 (batch, num_steps)
   ↓ 按时间步拆开，每步 (batch,)
每步索引 (batch,)
   ↓ one-hot
每步输入 (batch, vocab_size)   共 num_steps 个
```

## 2 参数初始化

五组参数，与 B19 的 2.1 节对应：

```python
W_xh = torch.randn(vocab_size, num_hiddens) * 0.01
W_hh = torch.randn(num_hiddens, num_hiddens) * 0.01
b_h  = torch.zeros(num_hiddens)
W_hq = torch.randn(num_hiddens, vocab_size) * 0.01
b_q  = torch.zeros(vocab_size)
```

**`× 0.01`**：书上用 `scale=0.01` 的正态。为什么这么小——`W_hh` 要在时间步之间反复相乘，
初始值大了几步就爆炸。小初始值让连乘的结果在合理范围内。

**偏置初始化为 0**：偏置没有"相乘放大"的问题。

**这些必须是 `nn.Parameter`**（B14 讲过），否则不进优化器。

## 3 模型前向

```python
def rnn_forward(inputs, state, params):
    """inputs 是 num_steps 个 (batch, vocab) 的列表；
    state 是 (H,)，H 形状 (batch, num_hiddens)。

    返回 (outputs, (H_new,))，outputs 是 num_steps 个 (batch, vocab) 的列表。
    """
```

**循环体**：

```
H = tanh(X_t @ W_xh + H @ W_hh + b_h)
Y_t = H @ W_hq + b_q
```

**输出 `Y_t` 是 logits**，没有经过 softmax。损失函数 `nn.CrossEntropyLoss` 内部会做 softmax。

**注意 `state` 用元组包起来**：书上的写法是为了将来支持有多层隐藏状态的 RNN（LSTM 有 `(H, C)` 两个）。
torch 里直接用张量也行，但按书的接口写，后面能复用。

## 4 预测函数

给定一段前缀，逐字符生成：

```
前缀 "分开"
  第 1 步：输入 "分"，预测下一个字符 → "开"
  第 2 步：输入 "开"，预测 → ?
  ...
```

**实现要点**：

1. 初始隐藏状态 `H = zeros(1, num_hiddens)`——**批量是 1**，一次只生成一条
2. 用前缀的字符喂进去，**但不采信模型的预测**（前缀是强制的）
3. 前缀喂完之后，每步把上一步的**预测结果**作为下一步的输入
4. 每步从 `Y` 里取 `argmax`

**为什么前缀阶段不采信预测**：要让模型从给定的上下文出发，而不是自己乱编开头。

**`argmax` 的选择方式**：书上用贪心（每步取概率最大的）。这样生成的文本容易重复，
更好的做法有温度采样、beam search，这一课不展开。

## 5 BPTT：梯度怎么沿时间传

### 5.1 计算图

RNN 展开之后，每个时间步是计算图里的一层。损失是各步损失之和：

$$L = \frac{1}{T}\sum_{t=1}^{T} \ell(\boldsymbol{o}_t, y_t)$$

**反向传播要沿时间反着走**，这叫**通过时间反向传播**（BPTT，backpropagation through time）。

### 5.2 隐藏状态的梯度是递归的

对最后一步：

$$\frac{\partial L}{\partial \boldsymbol{h}_T} = \boldsymbol{W}_{qh}^\top \frac{\partial L}{\partial \boldsymbol{o}_T}$$

对 $t < T$：

$$\frac{\partial L}{\partial \boldsymbol{h}_t} = \boldsymbol{W}_{hh}^\top \frac{\partial L}{\partial \boldsymbol{h}_{t+1}} + \boldsymbol{W}_{qh}^\top \frac{\partial L}{\partial \boldsymbol{o}_t}$$

**第一项**来自"下一步的隐藏状态依赖这一步"，**第二项**来自"这一步的输出依赖这一步的隐藏状态"。

展开之后，$\partial L/\partial \boldsymbol{h}_t$ 里会出现 $(\boldsymbol{W}_{hh}^\top)^{T-t}$ 这样的**连乘**。

### 5.3 为什么梯度会爆炸或消失

**矩阵的连乘**：如果 $\boldsymbol{W}_{hh}$ 的特征值都小于 1，连乘 $T-t$ 次后趋近 0（梯度消失）；
如果都大于 1，连乘后指数增长（梯度爆炸）。序列越长，两个问题越严重。

**这一条和 B11 讲的前馈网络梯度消失是同一件事**，多出来的是"连乘的次数等于时间步数"。

### 5.4 梯度裁剪

**做法**：把所有参数的梯度拼成一个向量 $\boldsymbol{g}$，如果它的范数超过阈值 $\theta$，
就整体按比例缩小：

$$\boldsymbol{g} \leftarrow \min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right) \boldsymbol{g}$$

**效果**：裁剪后 $\|\boldsymbol{g}\| \le \theta$，方向不变。

**注意"整体缩放"**：不是逐元素裁剪（那会改变梯度的方向），是把整个梯度向量按比例缩。

**实现**（torch 自带，但要自己写一遍理解）：

```python
def grad_clipping(params, theta):
    norm = torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None))
    if norm > theta:
        for p in params:
            if p.grad is not None:
                p.grad.mul_(theta / norm)
```

**torch 的现成实现**：`torch.nn.utils.clip_grad_norm_(params, theta)`，就是上面这几十行。
实际写代码用它，但要知道它在做什么。

## 6 困惑度

**困惑度**（perplexity）是语言模型的标准评价指标：

$$\text{perplexity} = \exp(\text{cross-entropy loss})$$

| 情况 | 交叉熵 | 困惑度 |
|---|---|---|
| 完美预测（概率 1）| 0 | 1 |
| 均匀瞎猜（$1/V$）| $\ln V$ | $V$ |
| 最差（概率 0）| $\infty$ | $\infty$ |

**基线是词表大小**：词表 1000 时，困惑度 1000 等于瞎猜。有效模型的困惑度必须小于词表大小。

**为什么用困惑度而不是交叉熵**：它的单位是"等效的候选词个数"。
困惑度 10 的意思是"模型平均在 10 个候选里犹豫"，比"损失 2.3"直观。

## 7 训练函数

训练循环与 B12 的相同，多三件事：

1. **每个批量用上一步的隐藏状态**（相邻采样时），但要把计算图切断
2. **每次更新前裁剪梯度**
3. **用困惑度报告**

### 7.1 切断计算图

相邻采样时，隐藏状态跨批量传递。如果一直连下去，计算图会无限增长。

**做法**：

```python
state = tuple(s.detach() for s in state)
```

**`detach()`** 把张量从计算图上摘下来——保留数值，但反向传播到它为止。
这样每个批量的梯度只依赖当前批量的序列，不看历史。

**不做会怎样**：显存持续增长，训练越久越慢，最后 OOM。

### 7.2 训练循环的完整结构

```
对每个 epoch：
    初始化隐藏状态
    对每个批量 (X, Y)：
        隐藏状态 detach
        前向，得到各步输出与损失
        反向
        裁剪梯度
        更新参数
    每若干步打印困惑度
```

## 8 实现会踩的坑

| 坑 | 症状 | 原因 |
|---|---|---|
| one-hot 的维度搞反 | 形状不匹配 | `(batch, num_steps)` 按时间步拆开后 one-hot（1）|
| `Y` 忘了过 softmax | 损失不降 | `CrossEntropyLoss` 内部做 softmax，`Y` 是 logits（3）|
| 忘了裁剪梯度 | 损失变 `nan` | 梯度爆炸（5.4）|
| 裁剪写成逐元素 | 训练变慢、方向被改 | 要整体按比例缩（5.4）|
| 相邻采样没 `detach` | 越训越慢、OOM | 计算图无限增长（7.1）|
| 参数没用 `nn.Parameter` | 不报错，但不更新 | B14（2）|
| 预测时批量当成 `num_steps` | 形状不匹配 | 一次只生成一条，批量是 1（4）|
| `W_hh` 初始化成 1 | 几步就 `nan` | 连乘放大（2）|

## 命令速查

| 命令 | 作用 | 讲义哪一节 |
|---|---|---|
| `torch.nn.functional.one_hot(t, num_classes)` | 索引转 one-hot | 1 |
| `torch.randn(shape) * 0.01` | 小初始值 | 2 |
| `torch.tanh(x)` | RNN 激活 | 3 |
| `torch.argmax(y, dim=1)` | 取预测字符 | 4 |
| `torch.sqrt(sum((p.grad**2).sum() for p in params))` | 梯度范数 | 5.4 |
| `p.grad.mul_(scale)` | 按比例缩梯度 | 5.4 |
| `torch.nn.utils.clip_grad_norm_(params, theta)` | 现成的裁剪 | 5.4 |
| `s.detach()` | 切断计算图 | 7.1 |
| `torch.exp(loss)` | 困惑度 | 6 |

## 自测题

1. one-hot 把索引变成什么？为什么不直接输入整数索引？
2. `W_hh` 的初始值为什么要乘 0.01？
3. RNN 的输出 `Y` 经过 softmax 了吗？损失函数需要什么？
4. 预测函数为什么在前缀阶段不采信模型的预测？
5. 梯度裁剪的公式是什么？为什么必须整体按比例缩放而不是逐元素裁剪？
6. 困惑度的定义是什么？基线是多少？
7. 相邻采样训练时为什么要 `detach` 隐藏状态？

## 答案（做完再看）

**1.** 变成长度为词表大小的向量，只有对应索引那一位是 1。
不直接输整数是因为整数有大小关系，而字符之间没有这种语义——one-hot 让所有字符等距。

**2.** `W_hh` 在时间步之间反复相乘。初始值大了，连乘几步就爆炸成 `nan`。
乘 0.01 是让它在前若干步内保持数值稳定。

**3.** 没有。`Y` 是 logits（未归一化的分数）。`nn.CrossEntropyLoss` 内部会做
log-softmax，所以传 logits 进去就可以。

**4.** 前缀是要强制模型接受的上下文。采信预测的话，模型会一开始就偏离给定开头，
前缀就失去了"引导"的作用。

**5.** $\boldsymbol{g} \leftarrow \min(\theta/\|\boldsymbol{g}\|, 1)\cdot\boldsymbol{g}$。
逐元素裁剪会改变梯度的**方向**（不同元素缩的比例不同），
整体缩放只改长度、保留方向。

**6.** 困惑度 = $\exp(\text{交叉熵})$。基线是词表大小——
均匀瞎猜时交叉熵是 $\ln V$，困惑度就是 $V$。

**7.** 相邻采样时隐藏状态跨批量传递，如果保留计算图，梯度会沿着所有历史批量累积，
计算图无限增长。`detach` 只保留数值，让每个批量的梯度只看当前批量。
