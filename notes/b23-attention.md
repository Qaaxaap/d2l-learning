# B23 注意力机制

B22 的 seq2seq 有一个硬瓶颈：整个输入压成一个定长背景变量。
输入 5 个词还是 50 个词，解码器拿到的都是同一形状的向量。

本章讲注意力机制。它让解码器每一步重新看一遍整个输入，按相关程度加权取平均。

本章讲四件事。第一，评分函数的几种形式。第二，掩蔽 softmax。
第三，一个最简单的注意力模型。第四，把机制接回 seq2seq 的 Bahdanau 注意力。

## 与书对应

| 讲义小节 | 纸质书（第一版） | 电子版（第二版） |
|---|---|---|
| 1 为什么需要注意力 | 10.11 的开头 | [10.1](https://zh.d2l.ai/chapter_attention-mechanisms/attention-cues.html) |
| 2 查询、键、值 | 10.11 的开头 | 10.1 |
| 3 Nadaraya-Watson | 无，第一版没有 | [10.2](https://zh.d2l.ai/chapter_attention-mechanisms/nadaraya-waston.html) |
| 4 掩蔽 softmax | 无 | [10.3](https://zh.d2l.ai/chapter_attention-mechanisms/attention-scoring-functions.html) 开头 |
| 5 加性注意力 | 10.11 的 $\boldsymbol{v}^\top\tanh(\cdots)$ | 10.3 的"加性注意力" |
| 6 缩放点积注意力 | 10.11 提到的内积 | 10.3 的"缩放点积注意力" |
| 7 Bahdanau 注意力 | 10.11 的"更新隐藏状态" | [10.4](https://zh.d2l.ai/chapter_attention-mechanisms/bahdanau-attention.html) |
| 8 实现会踩的坑 | 散在正文里，本节汇总 | 同上 |
| 自测题 | 无 | 无 |

**两版覆盖差异很大**。第一版的注意力一节只有 Bahdanau 一个应用。
它没有 Nadaraya-Watson、没有掩蔽 softmax、也没有把评分函数单独抽出来。

这三块只有第二版有。本章以电子版为主。

**解锁点**：v2 的 `masked_softmax`、`AdditiveAttention`、`DotProductAttention`、
`NadarayaWatson` 都标了 `#@save`。本章要自己写。

## 1 为什么需要注意力

### 1.1 B22 的瓶颈

B22 的编码器把整个输入读完，取最后一步的隐藏状态当背景变量：

$$\boldsymbol{c} = \boldsymbol{h}_T$$

这个向量是**定长**的。输入 5 个词与输入 50 个词，$\boldsymbol{c}$ 的形状相同。

50 个词的信息要挤进一个 $h$ 维向量。很多细节丢失。

**具体表现**：翻译长句子时，句末的词还能对得上，句首的词经常翻错或漏掉。
因为句首的信息经过几十步传递，已经被稀释。

### 1.2 注意力的思路

不问「怎么把整个输入压成一个向量」，改问「解码到这一步时，该看输入的哪些部分」。

做法：

- 解码器每一步重新看一遍整个输入的所有位置。
- 给每个位置算一个相关度分数。
- 按分数加权取平均，得到这一步的背景变量。

**关键差别**：

| | B22 | 本章 |
|---|---|---|
| 背景变量 | $\boldsymbol{c} = \boldsymbol{h}_T$，固定 | $\boldsymbol{c}_{t'} = \sum_t \alpha_{t't}\boldsymbol{h}_t$，每步重算 |
| 长输入 | 压成一个向量，丢信息 | 每个位置都保留，按需取 |

注意力不是新的一层网络。它是「加权平均」这个操作加一个学出来的打分函数。

## 2 查询、键、值

### 2.1 三个概念

注意力用三个概念描述：

| 概念 | 含义 | 在 seq2seq 里 |
|---|---|---|
| **查询**（query）| 我现在想要什么 | 解码器当前步的隐藏状态 $\boldsymbol{s}_{t'-1}$ |
| **键**（key）| 每个位置的「标识」，用来与查询比对 | 编码器各步的隐藏状态 $\boldsymbol{h}_t$ |
| **值**（value）| 实际被加权平均的内容 | 同样 $\boldsymbol{h}_t$ |

**用一个具体例子**。翻译「我爱学习」时，解码器生成 "learning" 这一步：

- 查询是解码器在生成 "I love" 之后的状态，它想要「一个表示动作或状态的名词」。
- 键是四个源词各自的隐藏状态。
- 值也是四个源词的隐藏状态。

打分的结果大概是「学习」这个位置的分数最高。加权平均后，
背景变量主要包含「学习」的信息。解码器据此生成 "learning"。

### 2.2 三个步骤

1. 用查询与每个键算一个**分数**：$e_i = a(\boldsymbol{q}, \boldsymbol{k}_i)$。
2. 分数过 softmax 得到**权重**：$\alpha_i$。代码写 `torch.softmax(scores, dim=-1)`。
3. 用权重对值加权求和：$\boldsymbol{c} = \sum_i \alpha_i \boldsymbol{v}_i$。

这就是注意力的全部。后面讲的评分函数不同、掩蔽方式不同，三步不变。

### 2.3 键与值常是同一个东西

上面的 seq2seq 例子里，键与值都是编码器的隐藏状态。

**这不是必须的**。Transformer 的自注意力里，键与值来自不同的投影。
它们的角色不同：

- 键只用来与查询比对，算分数。
- 值是实际被平均的内容。

分开是为了让模型能独立控制「用什么来匹配」与「取回什么信息」。

## 3 Nadaraya-Watson 核回归

这是注意力最小的例子。没有神经网络，纯加权平均，但结构完整。

### 3.1 非参数的版本

给定训练数据 $(\boldsymbol{x}_i, y_i)$，对新的 $\boldsymbol{x}$：

$$f(\boldsymbol{x}) = \sum_{i=1}^{n}
\frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}
{\sum_j \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i$$

对照 2.2 的三步看：

| 注意力的概念 | 在这里是什么 |
|---|---|
| 查询 $\boldsymbol{q}$ | $x$，要预测的位置 |
| 键 $\boldsymbol{k}_i$ | $x_i$，训练数据的位置 |
| 值 $\boldsymbol{v}_i$ | $y_i$，训练数据的标签 |
| 评分函数 | $-\frac{1}{2}(q - k)^2$，负的平方距离 |

**读法**：离 $x$ 近的训练点权重大，远处的权重小。加权平均得到预测。

**一个数值例子**。训练数据三点：$(0, 0)$、$(1, 10)$、$(2, 20)$。预测 $x = 1$。

- 距离平方：$(1-0)^2 = 1$、$(1-1)^2 = 0$、$(1-2)^2 = 1$。
- 分数：$-0.5$、$0$、$-0.5$。
- 权重（softmax 后）：$0.274$、$0.452$、$0.274$。
- 预测：$0.274\times0 + 0.452\times10 + 0.274\times20 = 10.0$。

正好是中间那个点的值。因为查询与它重合。

### 3.2 参数化的版本

给距离乘一个可学习的参数 $w$：

$$f(\boldsymbol{x}) = \sum_i
\text{softmax}\left(-\frac{1}{2}((x - x_i)w)^2\right) y_i$$

$w$ 控制「有多关注附近」：

- $w$ 大：曲线更尖锐，只看很近的点。
- $w$ 小：更平滑。
- $w = 0$：完全均匀，退化成所有 $y_i$ 的平均。

上例中取 $w = 10$，权重变成 $0$、$1$、$0$，预测恰好等于 10。
取 $w = 0$，权重各 $1/3$，预测是 $10$。这个例子太巧，看不出差别。
换个查询点 $x = 0.5$ 就能看出：$w$ 大时主要看前两点，$w$ 小时三点都算。

**训练**：用平方损失拟合 $y$，梯度下降学 $w$。

### 3.3 这个例子的价值

它把注意力拆成三个可以分开想的部分：

1. **打分**：$a(q,k)$ 怎么算。
2. **归一化**：softmax。
3. **加权平均**：$\sum \alpha_i v_i$。

后面所有的注意力都是这个结构。区别只在打分函数、是否并行、叠加几层。

## 4 掩蔽 softmax

### 4.1 为什么需要

批量里序列长度不同。短的序列要补到最长，叫 **padding**。

padding 的位置不是真实内容，不该参与注意力。

**做法**：把无效位置的分数设成很大的负数，softmax 之后它们的权重接近 0。

**具体例子**。一个批量两条序列，有效长度分别是 2 与 3。补齐到 3 之后：

```
样本 0：真 真 补
样本 1：真 真 真
```

样本 0 的第 3 个位置要掩蔽。

### 4.2 三种情况

`valid_lens` 有三种形式：

| `valid_lens` 形状 | 含义 |
|---|---|
| `None` | 不掩蔽，直接 softmax |
| `(batch,)` | 每个样本一个有效长度，该样本的所有查询共用 |
| `(batch, n)` | 每个查询各一个有效长度 |

### 4.3 实现步骤

1. 把 `(batch,)` 的长度扩展到 `(batch * n,)`，每个查询重复一次：
   `torch.repeat_interleave(valid_len, n)`。
2. 把 `X` 拍平成 `(batch * n, m)`。
3. 对每一行，下标大于等于 `valid_len` 的位置填 `-1e6`：
   `X.masked_fill(mask, -1e6)`，`mask` 是布尔张量。
4. 过 softmax。
5. 变回原形状。

拍平是为了让「每个查询」都变成独立的一行，方便逐行掩蔽。

### 4.4 为什么不用 -inf

`-inf` 参与 `exp` 得到 0，前向没问题。

问题在反向。`exp(-inf)` 的梯度是 `0 * inf`，结果是 `nan`。

用有限的 `-1e6`：softmax 输出仍是 0，梯度也是 0，不会出现 `nan`。

**这个区别在训练时才会暴露**。只跑前向看不出问题。

## 5 加性注意力

### 5.1 公式

$$a(\boldsymbol{q}, \boldsymbol{k}) = \boldsymbol{w}_v^\top
\tanh(\boldsymbol{W}_q\boldsymbol{q} + \boldsymbol{W}_k\boldsymbol{k})$$

参数：

- $\boldsymbol{W}_q \in \mathbb{R}^{h \times q}$：查询的投影。
- $\boldsymbol{W}_k \in \mathbb{R}^{h \times k}$：键的投影。
- $\boldsymbol{w}_v \in \mathbb{R}^{h}$：把隐藏维压成一个分数。

**为什么叫加性**：$\boldsymbol{W}_q\boldsymbol{q}$ 与 $\boldsymbol{W}_k\boldsymbol{k}$ 是
**相加**的，再过 tanh 与一个线性层。

### 5.2 适用场景

查询与键的长度不同（$q \neq k$）时用这个。

两个投影把它们映射到同一个隐藏维 $h$，之后才能相加。

### 5.3 形状与广播

设批量 $b$、查询数 $n_q$、键数 $n_k$、隐藏维 $h$：

| 量 | 形状 |
|---|---|
| 查询 | `(b, n_q, q)` |
| 键 | `(b, n_k, k)` |
| 投影后的查询 | `(b, n_q, h)` |
| 投影后的键 | `(b, n_k, h)` |
| 分数 | `(b, n_q, n_k)` |
| 权重 | `(b, n_q, n_k)` |
| 输出 | `(b, n_q, v)` |

**广播怎么做**：两个投影后的张量形状不同，不能直接相加。

- 查询的 `(b, n_q, h)` 要在中间插一维，变成 `(b, n_q, 1, h)`。
- 键的 `(b, n_k, h)` 要在前面插一维，变成 `(b, 1, n_k, h)`。
- 相加得到 `(b, n_q, n_k, h)`，每个查询与每个键都配上了。

代码里用 `q.unsqueeze(2) + k.unsqueeze(1)`。`unsqueeze(dim)` 在指定位置插入一维。

三个投影都用 `nn.Linear(in, out, bias=False)`，再过 `torch.tanh(x)`。

## 6 缩放点积注意力

### 6.1 公式

$$a(\boldsymbol{q}, \boldsymbol{k}) = \frac{\boldsymbol{q}^\top \boldsymbol{k}}{\sqrt{d}}$$

适用场景：查询与键的长度相同，都是 $d$。

**没有额外参数**。分数直接由内积给出。

### 6.2 为什么除以 sqrt(d)

假设 $\boldsymbol{q}$ 与 $\boldsymbol{k}$ 的元素独立、零均值、单位方差。
先算点积的方差。

$$\text{Var}\left(\sum_{i=1}^{d} q_i k_i\right) = \sum_{i=1}^{d} \text{Var}(q_i k_i) = d$$

**方差是 $d$**。$d = 64$ 时点积的标准差是 8，$d = 512$ 时是 22.6。

分数范围大，softmax 会变得极端（接近 one-hot）。
权重集中在一个位置，梯度接近 0，训练不动。

除以 $\sqrt{d}$ 把方差压回 1。

### 6.3 矢量化形式

$n$ 个查询、$m$ 个键值对：

$$\text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^\top}{\sqrt{d}}\right)\boldsymbol{V}$$

代码里写 `torch.bmm(Q, K.transpose(1, 2))`，`bmm` 是批量矩阵乘。

形状：

- $\boldsymbol{Q}$：`(n, d)`
- $\boldsymbol{K}$：`(m, d)`
- $\boldsymbol{K}^\top$：`(d, m)`
- $\boldsymbol{Q}\boldsymbol{K}^\top$：`(n, m)`，每个查询与每个键的分数
- softmax 在**最后一维**（键那一维）
- $\boldsymbol{V}$：`(m, v)`
- 结果：`(n, v)`

**softmax 的维度不能搞错**。要在键那一维归一化，表示「这个查询对所有键的权重和为 1」。

## 7 Bahdanau 注意力

### 7.1 每步重算背景变量

把上面的机制接进 B22 的 seq2seq。

$$\boldsymbol{c}_{t'} = \sum_{t=1}^{T} \alpha_{t't}\boldsymbol{h}_t$$

$$\alpha_{t't} = \text{softmax}_t\left(a(\boldsymbol{s}_{t'-1}, \boldsymbol{h}_t)\right)$$

**关键**：$t'$ 变了，$\boldsymbol{c}$ 就变。这是与 B22 的差别。

### 7.2 解码器的 GRU 也要改

B22 的解码器：$\boldsymbol{s}_{t'} = g(y_{t'-1}, \boldsymbol{s}_{t'-1})$。

加了注意力之后：$\boldsymbol{s}_{t'} = g(y_{t'-1}, \boldsymbol{s}_{t'-1}, \boldsymbol{c}_{t'})$。

具体到 GRU 的三个门，每一项都多一个 $\boldsymbol{W}_{c*}\boldsymbol{c}_{t'}$：

$$\boldsymbol{r}_{t'} = \sigma(\boldsymbol{W}_{yr}\boldsymbol{y}_{t'-1}
+ \boldsymbol{W}_{sr}\boldsymbol{s}_{t'-1} + \boldsymbol{W}_{cr}\boldsymbol{c}_{t'} + \boldsymbol{b}_r)$$

$$\boldsymbol{z}_{t'} = \sigma(\boldsymbol{W}_{yz}\boldsymbol{y}_{t'-1}
+ \boldsymbol{W}_{sz}\boldsymbol{s}_{t'-1} + \boldsymbol{W}_{cz}\boldsymbol{c}_{t'} + \boldsymbol{b}_z)$$

$$\tilde{\boldsymbol{s}}_{t'} = \tanh(\boldsymbol{W}_{ys}\boldsymbol{y}_{t'-1}
+ \boldsymbol{W}_{ss}(\boldsymbol{s}_{t'-1}\odot\boldsymbol{r}_{t'})
+ \boldsymbol{W}_{cs}\boldsymbol{c}_{t'} + \boldsymbol{b}_s)$$

所以在有注意力的解码器里，GRU 的每个门都多一组权重。

### 7.3 与 B22 的对比

| | B22 seq2seq | B22 加 Bahdanau |
|---|---|---|
| 编码器返回 | 只返回 $\boldsymbol{h}_T$ | 返回**所有** $\boldsymbol{h}_t$ |
| 解码器初始状态 | 直接用 $\boldsymbol{h}_T$ | $\boldsymbol{h}_T$ 只作初始，之后每步重算 $\boldsymbol{c}_{t'}$ |
| 长输入 | 丢失信息 | 每个位置都能被回头看 |
| 计算量 | 小 | 大（每步要跟整个输入算一次注意力）|

**代价**：计算量与内存都变大。长序列上的效果提升明显，所以值得。

## 8 实现会踩的坑

| 坑 | 症状 | 原因 |
|---|---|---|
| 掩蔽用了 `-inf` | 梯度出现 `nan` | `0 * inf`（4.4）|
| 掩蔽的位置没设成负的大数 | padding 参与注意力 | 4.1 |
| 加性注意力没做广播 | 形状不匹配 | 查询与键要在不同维度扩展（5.3）|
| 点积注意力忘了除以 $\sqrt{d}$ | 训练不稳 | $d$ 大时 softmax 过尖（6.2）|
| softmax 的维度搞错 | 权重在错误维度归一化 | 最后一维是键（6.3）|
| 注意力用编码器最后一步当键 | 只有一步，没意义 | 键要用**所有**编码器状态（7.1）|
| 解码器忘了把 $\boldsymbol{c}_{t'}$ 送进 GRU | 效果与 B22 相同 | 每个门都要加 $\boldsymbol{W}_{c*}\boldsymbol{c}_{t'}$（7.2）|

## 命令速查

| 命令 | 作用 | 讲义哪一节 |
|---|---|---|
| `torch.softmax(x, dim=-1)` | 归一化 | 2.2 |
| `torch.bmm(Q, K.transpose(1, 2))` | 批量矩阵乘，算所有分数 | 6.3 |
| `torch.repeat_interleave(t, n)` | 把长度扩展成逐 query 一份 | 4.3 |
| `X.masked_fill(mask, -1e6)` | 掩蔽 | 4.3 |
| `q.unsqueeze(2)` | 插入一维以便广播 | 5.3 |
| `nn.Linear(in, out, bias=False)` | 打分函数里的投影 | 5.1 |
| `torch.tanh(x)` | 加性注意力的激活 | 5.1 |

## 自测题

1. 写出 B22 的背景变量公式，说明它的瓶颈，并指出注意力怎么绕过它。
2. 分别写出查询、键、值在 seq2seq 里对应的东西。
3. 写出注意力的三个步骤。
4. 说明键与值可以是不同的东西，并解释为什么要把它们分开。
5. 写出 Nadaraya-Watson 的预测公式，指出打分函数是什么。
6. 说明掩蔽 softmax 的作用，并说明为什么不能用 `-inf`。
7. 写出加性注意力的打分函数与适用场景。
8. 写出缩放点积注意力的打分函数，说明除以 $\sqrt{d}$ 的理由。
9. 写出 Bahdanau 注意力的背景变量公式，说明它与 B22 的差别。
10. 说明 Bahdanau 解码器的 GRU 与 B22 的差在哪。

## 答案（做完再看）

**1.** B22 的 $\boldsymbol{c} = \boldsymbol{h}_T$ 是定长向量，长输入会丢信息。
注意力改成每一步重新看一遍所有 $\boldsymbol{h}_t$，按相关度加权取平均：
$\boldsymbol{c}_{t'} = \sum_t \alpha_{t't}\boldsymbol{h}_t$。

**2.** 查询是解码器当前步的隐藏状态。键与值都是编码器各步的隐藏状态。

**3.** 用查询与每个键算分数；分数过 softmax 得权重；用权重对值加权求和。

**4.** 键只用来与查询比对，值是实际被平均的内容。
分开是为了让模型能独立控制「用什么匹配」与「取回什么」。

**5.** $f(x) = \sum_i \text{softmax}\left(-\frac{1}{2}(x-x_i)^2\right) y_i$。
打分函数是负的平方距离。

**6.** 把 padding 位置的分数设成负的大数，让它们的权重接近 0。

不能用 `-inf`：反向传播时 `exp(-inf)` 的梯度是 `0 * inf`，得到 `nan`。

**7.** $a(\boldsymbol{q},\boldsymbol{k}) = \boldsymbol{w}_v^\top\tanh(
\boldsymbol{W}_q\boldsymbol{q} + \boldsymbol{W}_k\boldsymbol{k})$。
适用于查询与键长度不等的时候。

**8.** $a(\boldsymbol{q},\boldsymbol{k}) = \boldsymbol{q}^\top\boldsymbol{k}/\sqrt{d}$。

点积的方差是 $d$。$d$ 大时分数范围大，softmax 过尖，梯度接近 0。
除以 $\sqrt{d}$ 把方差压回 1。

**9.** $\boldsymbol{c}_{t'} = \sum_t \alpha_{t't}\boldsymbol{h}_t$，
$\alpha_{t't} = \text{softmax}_t(a(\boldsymbol{s}_{t'-1},\boldsymbol{h}_t))$。

与 B22 的差别：$t'$ 变了背景变量就变，不再是固定的 $\boldsymbol{h}_T$。

**10.** B22 的 GRU 只看 $y_{t'-1}$ 与 $\boldsymbol{s}_{t'-1}$。
加了注意力之后每个门多一项 $\boldsymbol{W}_{c*}\boldsymbol{c}_{t'}$，
GRU 的每个门都多一组权重。
