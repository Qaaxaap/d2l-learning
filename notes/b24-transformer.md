# B24 多头注意力、自注意力与 Transformer

B23 讲了注意力的打分与加权：**一个**查询对一组键值。这一章把它变成能搭深网络的组件。

三件事：**多头**（并行做多次注意力）、**自注意力**（查询键值同源）、**Transformer**
（只用注意力和前馈层堆出来的编码器-解码器）。

## 与书对应

| 讲义小节 | 纸质书（第一版） | 电子版（第二版） |
|---|---|---|
| 1 多头注意力 | 无，第一版没有这一节 | [10.5](https://zh.d2l.ai/chapter_attention-mechanisms/multihead-attention.html) |
| 2 形状变换 `transpose_qkv` | 无 | 10.5 的"实现" |
| 3 自注意力 | 无 | [10.6](https://zh.d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html) |
| 4 位置编码 | 无 | 10.6 的"位置编码" |
| 5 Transformer | 无 | [10.7](https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html) |
| 6 层规范化与 AddNorm | 无 | 10.7 的"残差连接和层规范化" |
| 7 实现会踩的坑 | 散在正文里，本节汇总 | 同上 |
| 自测题 | 无 | 无 |

**第一版完全没有这一章**。第一版（2019 年）的注意力只有 B23 讲过的 Bahdanau，
Transformer 那一节在第二版才补上。这一章全部按第二版取材。

**解锁点**：v2 的 `transpose_qkv`、`transpose_output`、`MultiHeadAttention`、
`PositionalEncoding`、`AddNorm`、`LayerNorm` 都标了 `#@save`。这一章要自己写。

## 1 多头注意力

### 1.1 为什么一个头不够

一个注意力头做的事是：**用一组投影把查询键值映射到一个子空间，在那里算加权平均**。

问题在于**投影只有一组**。同一个输入，注意力只能学到一种"相关性的看法"。
但语言里同时存在多种关系：

- 语法关系（主谓、动宾）
- 指代关系（"他"指谁）
- 位置关系（相邻词）

**多头**的做法：并行跑 $h$ 组独立的注意力，每组用自己的投影，最后把结果拼起来。

$$\boldsymbol{h}_i = f(\boldsymbol{W}_i^{(q)}\boldsymbol{q}, \boldsymbol{W}_i^{(k)}\boldsymbol{k}, \boldsymbol{W}_i^{(v)}\boldsymbol{v}) \in \mathbb{R}^{p_v}$$

拼起来再过一次线性变换：

$$\boldsymbol{W}_o \begin{bmatrix}\boldsymbol{h}_1\\ \vdots\\ \boldsymbol{h}_h\end{bmatrix} \in \mathbb{R}^{p_o}$$

**每个头的维度是总维度除以头数**。$h=8$、总维度 512 时，每个头 64 维。
总计算量与"单头 512 维"差不多，但表达能力强得多。

### 1.2 参数量

设查询维 $d_q$、键维 $d_k$、值维 $d_v$、隐藏维 $p$、头数 $h$、输出维 $p_o$：

| 参数 | 形状 | 数量 |
|---|---|---|
| $\boldsymbol{W}_i^{(q)}$（每个头一个）| $(\frac{p}{h}, d_q)$ | $h \cdot \frac{p}{h} \cdot d_q = p \cdot d_q$ |
| $\boldsymbol{W}_i^{(k)}$ | $(\frac{p}{h}, d_k)$ | $p \cdot d_k$ |
| $\boldsymbol{W}_i^{(v)}$ | $(\frac{p}{h}, d_v)$ | $p \cdot d_v$ |
| $\boldsymbol{W}_o$ | $(p_o, p)$ | $p_o \cdot p$ |

**注意 $h$ 约掉了**：多头注意力的参数量与"单头、总维度 $p$"相同。
多头不是靠加参数换效果，是靠同样的参数换**多个子空间**。

### 1.3 实现的结构

四个线性层 + 一次"拆头 → 算注意力 → 拼头"：

```
queries ─→ W_q ─→ transpose_qkv ─┐
keys    ─→ W_k ─→ transpose_qkv ─┼─→ DotProductAttention ─→ transpose_output ─→ W_o ─→ 输出
values  ─→ W_v ─→ transpose_qkv ─┘
```

**三个投影都用无偏置的 `nn.Linear`**（书上 `use_bias=False`），
$\boldsymbol{W}_o$ 也不带偏置。

### 1.4 用现成的 scaled_dot_product_attention

B23 的 `DotProductAttention` 是自己写的：算分数、除以 $\sqrt{d}$、过 masked softmax、
再批矩阵乘。torch 2.0 起有现成实现：

```python
F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0)
```

| 参数 | 形状 | 说明 |
|---|---|---|
| `query` | `(batch, n_q, d)` | 或 `(batch, h, n_q, d)` |
| `key` | `(batch, n_k, d)` | |
| `value` | `(batch, n_k, d_v)` | |
| `attn_mask` | `(n_q, n_k)` 或可广播 | **bool 时 True 表示允许**；float 时加到分数上（用 `-inf` 遮）|
| 返回 | `(batch, n_q, d_v)` | |

**它内部做的就是** `softmax(q @ k.T / sqrt(d) + mask) @ v`，还会自动选更快的实现
（flash attention 之类）。实际项目里用它，不要自己写。

**`attn_mask` 的 bool 语义容易记反**：`True` 表示**允许**这个位置参与，`False` 表示遮住。
与 `masked_softmax` 里"填 `-1e6` 遮住"正好相反。

多头时把 `(batch*h, n, p/h)` 当作批量维更大的普通三维张量传进去。
**掩码的形状要配合这个三维**：传 `(batch*h, 1, n_k)`，让查询那一维广播。
传成四维 `(batch*h, 1, 1, n_k)` 会与内部的分数张量对不上。

## 2 形状变换 `transpose_qkv`

**这是多头注意力的实现难点**。目标是把"一个批量里 $h$ 个头"变成"$h$ 倍的批量"，
好让注意力用批矩阵乘法一次算完。

### 2.1 三步

输入形状 `(batch, n, p)`，`n` 是查询（或键值）个数、`p` 是总隐藏维：

| 步 | 操作 | 得到 |
|---|---|---|
| 1 | `reshape(batch, n, num_heads, -1)` | `(batch, n, h, p/h)` |
| 2 | `transpose(0, 2, 1, 3)` | `(batch, h, n, p/h)` |
| 3 | `reshape(-1, n, p/h)` | `(batch * h, n, p/h)` |

**第 2 步为什么交换这两个轴**：把"头"放到批量的位置，让每个头成为独立的一个批量。
第 3 步把前两维拍平，`(batch, h)` 合成 `batch * h`。

### 2.2 为什么必须先 reshape 再 transpose

`reshape` 会把内存里连续的元素按新形状重新分组。如果直接 `reshape(batch*h, n, p/h)`，
分出来的是"每 h 个连续元素一组"，那是按**位置**切，不是按**头**切。

正确的切法是：先拆成 `(batch, n, h, p/h)`——这时第 3 维才是"头"，
每个头对应连续的 `p/h` 个元素——再把这个维换到批量旁边。

**记法**：`reshape` 负责"把连续的一段分给一个头"，`transpose` 负责"把头挪到批量维"。

### 2.3 逆变换 `transpose_output`

注意力算完，形状是 `(batch * h, n, p/h)`，要拼回 `(batch, n, p)`：

| 步 | 操作 | 得到 |
|---|---|---|
| 1 | `reshape(-1, h, n, p/h)` | `(batch, h, n, p/h)` |
| 2 | `transpose(0, 2, 1, 3)` | `(batch, n, h, p/h)` |
| 3 | `reshape(batch, n, -1)` | `(batch, n, p)` |

**顺序反过来做**。两个函数互为逆操作。

## 3 自注意力

### 3.1 定义

**查询、键、值都来自同一个输入**：

$$\text{输出} = \text{Attention}(\boldsymbol{X}, \boldsymbol{X}, \boldsymbol{X})$$

每个位置对序列里**所有位置**算注意力，包括自己。

**与 B23 的差别**：B23 里查询来自解码器、键值来自编码器（两个不同来源）。
自注意力里三者同源，用来让序列内部互相"看"。

### 3.2 与 CNN、RNN 的比较

设序列长度 $n$，这是 Transformer 取代 RNN 的三条理由：

| | RNN | CNN | 自注意力 |
|---|---|---|---|
| 并行度 | 差（必须逐步）| 好 | 好 |
| 任意两位置的最大路径长度 | $O(n)$ | $O(\log_k n)$ | $O(1)$ |
| 每层计算量 | $O(n d^2)$ | $O(k n d^2)$ | $O(n^2 d)$ |

**路径长度**决定梯度传播与"看到远处"的难度。自注意力里任意两个位置**直接相连**，
路径长度 1。

**代价是 $O(n^2)$**：$n$ 个位置两两都要算。长序列上这是 Transformer 的主要开销。

### 3.3 掩蔽自注意力

解码器里的自注意力要**遮住未来的位置**：生成第 $t$ 个词时，不能看到 $t$ 之后的词。

**做法**：把上三角的位置（含对角线？不含）设成 $-\infty$，再过 softmax 就变成 0。

书上的 `masked_softmax`（B23 的 3.2 节）正好用得上：
`valid_lens` 里传 `1, 2, 3, ..., n`，就得到逐步增长的可见范围。

## 4 位置编码

### 4.1 为什么需要

自注意力**没有顺序概念**：交换序列里两个位置，输出只是跟着交换，
模型本身区分不出"谁在前"。

RNN 靠逐步递推天然带顺序，CNN 靠卷积核的位置，自注意力什么都没有。
所以要**显式把位置信息加进去**：

$$\boldsymbol{X}' = \boldsymbol{X} + \boldsymbol{P}$$

$\boldsymbol{P}$ 与 $\boldsymbol{X}$ 同形状 `(n, d)`。

### 4.2 正弦余弦编码

$$\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right) \\ p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right) \end{aligned}$$

$i$ 是位置（第几行），$j$ 是维度编号（第几列的一半）。

**偶数维用 sin、奇数维用 cos**，每对维度共用一个频率。
频率从 $1$ 一路降到 $1/10000$：靠前的维度变化快（编码细节），
靠后的维度变化慢（编码大尺度位置）。

### 4.3 实现

```python
P = torch.zeros(1, max_len, d)
X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) \
    / torch.pow(10000, torch.arange(0, d, 2, dtype=torch.float32) / d)
P[:, :, 0::2] = torch.sin(X)
P[:, :, 1::2] = torch.cos(X)
```

**`0::2` 与 `1::2` 是切片步长**：前者取偶数下标，后者取奇数下标。

**`P` 不用训练**：它由公式算出来，用 `register_buffer` 存（B13 讲过），
不进优化器。

### 4.4 为什么用三角函数

书上给了两条理由：

1. **任意位置都能算出**。$P$ 不依赖序列长度，第 10000 个位置的编码可以直接算出来
2. **相对位置可以用线性变换表示**。$\sin(i + \delta)$ 可以写成 $\sin i$ 与 $\cos i$
   的线性组合。模型学得到"相隔 $\delta$ 个位置"这种关系

**第 2 条是可学的位置编码没有的性质**。这也是它至今还被用的原因。

## 5 Transformer

### 5.1 整体结构

Transformer 是 B22 的编码器-解码器架构，但两个 RNN 换成了注意力堆叠：

```
源序列 → 嵌入 + 位置编码 → 编码器（N 个块） → 编码器输出 ─┐
                                                      ↓
目标序列 → 嵌入 + 位置编码 → 解码器（N 个块） → 线性层 → 输出
                              ↑
                        编码器输出（交叉注意力）
```

### 5.2 编码器块

每个块两个子层：

1. **多头自注意力**：查询键值都来自前一个块的输出
2. **基于位置的前馈网络**：对每个位置**独立**做一次两层全连接

**每个子层都包在 `AddNorm` 里**：

$$\boldsymbol{X} \leftarrow \text{LayerNorm}(\boldsymbol{X} + \text{sublayer}(\boldsymbol{X}))$$

**残差连接要求输入输出同维**：所以 Transformer 里所有子层的输入输出都是 $d$ 维。

### 5.3 基于位置的前馈网络

对序列里每个位置**单独**过一个两层 MLP：

$$\text{FFN}(\boldsymbol{x}) = \max(0, \boldsymbol{x}\boldsymbol{W}_1 + \boldsymbol{b}_1)\boldsymbol{W}_2 + \boldsymbol{b}_2$$

$\boldsymbol{W}_1$ 把 $d$ 维升到 $d_{ff}$（一般 $d_{ff} = 4d$），$\boldsymbol{W}_2$ 降回来。

**"基于位置"的意思是同一个 MLP 作用在每个位置上**，位置之间不交流——
交流的事交给注意力层。

**实现要点**：输入是 `(batch, n, d)`，要先展平成 `(batch*n, d)` 过 MLP，
再 reshape 回 `(batch, n, d)`。因为 `nn.Linear` 只处理最后一维，形状上本来就能直接用，
展平只是为了在批处理上更整齐。

### 5.4 解码器块

三个子层：

1. **掩蔽多头自注意力**：查询键值来自前一个解码器块的输出，**遮住未来位置**
2. **编码器-解码器注意力**（交叉注意力）：查询来自解码器，**键值来自编码器输出**
3. **前馈网络**

**三个子层都有 AddNorm**。

**交叉注意力是编码器与解码器唯一的信息通道**：解码器从这里"看"整个源序列。
其余部分两个模块互不知道。

### 5.5 训练时的掩蔽

训练用强制教学（B22 的 2.2 节）：解码器输入是 `labels[:, :-1]`。

**掩蔽自注意力保证第 $t$ 个位置只能看到 $1..t$ 的位置**，
与必须"只依赖已生成词元"的自回归要求一致。

掩蔽的实现：`valid_lens = [1, 2, 3, ..., n]`。第 0 个位置只能看 1 个，
第 1 个位置能看 2 个，以此类推。

## 6 层规范化与 AddNorm

### 6.1 层规范化 vs 批量规范化

两者目标相同：把数据重新中心化、重新缩放。**区别在归一化的维度**：

| | 批量规范化（B18）| 层规范化 |
|---|---|---|
| 归一化哪些元素 | 同一个特征，跨**批量** | 同一个样本，跨**特征** |
| 依赖批量大小 | 是 | 否 |
| NLP 里用哪个 | 不好用 | **用这个** |

**为什么 NLP 里批量规范化不好用**：序列长度不定，padding 会混进统计量；
而且批量小时统计量不可靠。层规范化只看单个样本自己的特征维，没有这些问题。

### 6.2 公式

对形状 `(batch, n, d)` 的输入，**在最后一维（$d$）上**算均值与方差：

$$\mu = \frac{1}{d}\sum_{k=1}^{d} x_k, \qquad \sigma^2 = \frac{1}{d}\sum_{k=1}^{d}(x_k - \mu)^2$$

$$\text{LN}(\boldsymbol{x}) = \frac{\boldsymbol{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$

$\boldsymbol{\gamma}$、$\boldsymbol{\beta}$ 都是长度为 $d$ 的可学习参数。

**形状里的 `(batch, n)` 两维都算独立样本**：每个 `(batch, n)` 位置各自归一化自己的 $d$ 个特征。

### 6.3 AddNorm

$$\text{AddNorm}(\boldsymbol{X}) = \text{LayerNorm}(\boldsymbol{X} + \text{Dropout}(\text{sublayer}(\boldsymbol{X})))$$

**先加法、再 dropout、最后归一化**。书上在 `AddNorm` 里放了 dropout。

## 7 实现会踩的坑

| 坑 | 症状 | 原因 |
|---|---|---|
| `reshape` 与 `transpose` 顺序反了 | 形状对但结果错 | 先 reshape 拆头，再 transpose（2.2）|
| 头维度算错 | 形状不匹配 | 每头维度 = 总维 / 头数（1.1）|
| 位置编码当参数训练 | 参数量多出 `max_len*d` | 用 `register_buffer`（4.3）|
| 位置编码加在错误的位置 | 效果差 | 加在**嵌入之后、进编码器之前**（5.1）|
| LayerNorm 的维搞错 | 形状对但语义错 | 在**最后一维**归一化（6.2）|
| 解码器忘了掩蔽 | 训练正常、生成乱 | 自注意力要看 `valid_lens`（3.3）|
| 交叉注意力的查询搞反 | 效果差 | 查询来自解码器，键值来自编码器（5.4）|

## 命令速查

| 命令 | 作用 | 讲义哪一节 |
|---|---|---|
| `x.reshape(b, n, h, -1)` | 拆头 | 2.1 |
| `x.transpose(0, 2, 1, 3)` | 把头挪到批量维 | 2.1 |
| `x.reshape(-1, n, p // h)` | 拍平批量与头 | 2.1 |
| `torch.arange(0, d, 2)` | 取偶数维下标 | 4.3 |
| `P[:, :, 0::2] = ...` | 步长切片赋值 | 4.3 |
| `torch.pow(10000, ...)` | 频率 | 4.2 |
| `register_buffer(name, tensor)` | 存不训练的张量 | 4.3 |
| `x.mean(-1, keepdim=True)` | 层规范化的均值 | 6.2 |
| `x.var(-1, keepdim=True, unbiased=False)` | 层规范化的方差 | 6.2 |
| `nn.LayerNorm(d)` | 现成的层规范化 | 6.2 |

## 自测题

1. 多头注意力为什么有用？为什么头数增加不增加参数量？
2. `transpose_qkv` 的三步各做什么？为什么不能直接 reshape 成 `(batch*h, n, p/h)`？
3. 自注意力相比 RNN 的优势是什么？代价是什么？
4. 位置编码为什么必要？为什么用正弦余弦而不是可学习参数？
5. Transformer 的编码器块有几个子层？解码器块呢？
6. 交叉注意力的查询、键、值分别来自哪里？
7. LayerNorm 与 BatchNorm 的区别？为什么 NLP 用 LayerNorm？

## 答案（做完再看）

**1.** 一个头只有一组投影，只能学一种相关性；多个头并行学多个子空间（语法、指代、位置）。
参数量不随头数增加，因为每个头的维度是总维除以头数，$h \cdot (p/h) = p$ 约掉了 $h$。

**2.** reshape 拆成 `(batch, n, h, p/h)`、transpose 成 `(batch, h, n, p/h)`、
reshape 拍平成 `(batch*h, n, p/h)`。
不能直接 reshape，因为那样是按位置切分而不是按头切分（2.2）。

**3.** 优势：并行、任意两位置路径长度 $O(1)$。
代价：$O(n^2)$ 的计算量与内存，长序列上很贵。

**4.** 自注意力没有顺序概念，交换位置只交换输出。用正弦余弦是因为固定公式不占参数、
任意长度都能算，而且相对位置可以线性表示。

**5.** 编码器块 2 个子层（多头自注意力、前馈网络）。解码器块 3 个（掩蔽自注意力、
交叉注意力、前馈网络）。每个子层都包 AddNorm。

**6.** 查询来自解码器前一个块的输出；键和值来自编码器的输出。

**7.** BatchNorm 在批量维归一化（同一特征跨样本），LayerNorm 在特征维归一化（同样本跨特征）。
NLP 里序列变长、padding 会污染批量统计量，而且批量小的时候统计量不可靠，
所以用 LayerNorm。
