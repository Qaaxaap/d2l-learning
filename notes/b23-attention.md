# B23 注意力机制

B22 的 seq2seq 有一个硬瓶颈：**整个输入压成一个定长背景变量**。输入 5 个词还是 50 个词，
解码器拿到的都是同一个形状的向量。长句子必然丢信息。

**注意力机制**解决它：解码时**每一小步都重新看一遍整个输入**，按相关程度加权取平均。
相关程度高的位置权重大。

这一章讲三四件事：评分函数的几种形式、掩蔽 softmax、以及一个最简单的注意力模型
（Nadaraya-Watson）。Bahdanau 注意力是把这些接回 seq2seq 的完整例子。

## 与书对应

| 讲义小节 | 纸质书（第一版） | 电子版（第二版） |
|---|---|---|
| 1 查询、键、值 | 10.11 的开头 | [10.1](https://zh.d2l.ai/chapter_attention-mechanisms/attention-cues.html) |
| 2 Nadaraya-Watson 核回归 | 无，第一版没有这一节 | [10.2](https://zh.d2l.ai/chapter_attention-mechanisms/nadaraya-waston.html) |
| 3 掩蔽 softmax | 无，第一版跳过 | [10.3](https://zh.d2l.ai/chapter_attention-mechanisms/attention-scoring-functions.html) 开头 |
| 4 加性注意力 | 10.11 的 $\boldsymbol{v}^\top\tanh(\boldsymbol{W}_s\boldsymbol{s} + \boldsymbol{W}_h\boldsymbol{h})$ | 10.3 的"加性注意力" |
| 5 缩放点积注意力 | 10.11 提到的内积 | 10.3 的"缩放点积注意力" |
| 6 Bahdanau 注意力 | 10.11 的"更新隐藏状态" | [10.4](https://zh.d2l.ai/chapter_attention-mechanisms/bahdanau-attention.html) |
| 7 实现会踩的坑 | 散在正文里，本节汇总 | 同上 |
| 自测题 | 无 | 无 |

**注意两版的覆盖差异**：第一版的注意力一节只有 Bahdanau 那一个应用，
没有 Nadaraya-Watson、没有掩蔽 softmax、没有把评分函数单独抽出来讲。
这三块**只有第二版有**，所以这一章讲义在第二版上取材更多。

**解锁点**：v2 的 `masked_softmax`、`AdditiveAttention`、`DotProductAttention`、
`NadarayaWatson` 都标了 `#@save`，属于包内函数。这一章要自己写。

## 1 查询、键、值

注意力机制用三个概念描述：

| 概念 | 含义 | 在 seq2seq 里 |
|---|---|---|
| **查询**（query）| 我现在想要什么 | 解码器当前步的隐藏状态 $\boldsymbol{s}_{t'-1}$ |
| **键**（key）| 每个位置的"标识"，用来与查询比对 | 编码器各步的隐藏状态 $\boldsymbol{h}_t$ |
| **值**（value）| 实际被加权平均的内容 | 同样 $\boldsymbol{h}_t$ |

**流程**：

1. 用查询与每个键算一个**分数** $e_i = a(\boldsymbol{q}, \boldsymbol{k}_i)$
2. 分数过 softmax 得到**权重** $\alpha_i$
3. 用权重对值加权求和：$\boldsymbol{c} = \sum_i \alpha_i \boldsymbol{v}_i$

**键与值常常是同一个东西**（seq2seq 里都是编码器隐藏状态），但不一定。
自注意力里键与值来自输入的不同投影。

### 1.1 与 B22 的对比

| | B22 的 seq2seq | B23 的注意力 |
|---|---|---|
| 背景变量 | $\boldsymbol{c} = \boldsymbol{h}_T$（固定） | $\boldsymbol{c}_{t'} = \sum_t \alpha_{t't}\boldsymbol{h}_t$（每步重新算）|
| 长输入 | 压成一个向量，丢信息 | 每个位置都保留，按需取 |

**注意力不是新的一层网络**，它是"加权平均"这个操作 + 一个学出来的打分函数。

## 2 Nadaraya-Watson 核回归

这是注意力的**最小例子**——没有神经网络，纯加权平均，但已经是完整的注意力结构。

### 2.1 非参数的注意力汇聚

给定训练数据 $(\boldsymbol{x}_i, y_i)$，对新的 $\boldsymbol{x}$：

$$f(\boldsymbol{x}) = \sum_{i=1}^{n} \text{softmax}\left(-\frac{1}{2}(x - x_i)^2\right) y_i
= \sum_{i=1}^{n} \frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}{\sum_j \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i$$

**读法**：

- 查询 $\boldsymbol{q} = x$（要预测的位置）
- 键 $\boldsymbol{k}_i = x_i$（训练数据位置）
- 值 $\boldsymbol{v}_i = y_i$（训练数据标签）
- 评分函数 $a(q, k) = -\frac{1}{2}(q - k)^2$（负的平方距离：越近分数越高）

**这个式子就是注意力**：离 $x$ 近的训练点权重大，远处的权重小，加权平均得到预测。

### 2.2 参数化的版本

给距离乘一个可学习的参数 $w$：

$$f(\boldsymbol{x}) = \sum_i \text{softmax}\left(-\frac{1}{2}((x - x_i)w)^2\right) y_i$$

**$w$ 的作用**：控制"有多关注附近"。$w$ 大时曲线更尖锐（只看很近的点），
$w$ 小时更平滑。

**训练**：用平方损失拟合 $y$，梯度下降学 $w$。这就是"用注意力做回归"。

### 2.3 为什么这个例子有价值

它把注意力拆成三个可以分开想的部分：

1. **打分**：$a(q,k)$ 怎么算
2. **归一化**：softmax
3. **加权平均**：$\sum \alpha_i v_i$

后面所有的注意力（包括 Transformer 的多头注意力）都是这个结构，
只是换了打分函数、加了并行、加了多层。

## 3 掩蔽 softmax

### 3.1 为什么需要

批量里序列长度不同，短的序列要补到最长（padding）。
**padding 的位置不应该参与注意力**——它们不是真实内容。

**做法**：把无效位置的分数设成一个很大的负数（如 `-1e6`），
softmax 之后它们的权重接近 0。

### 3.2 实现

```python
def masked_softmax(X, valid_lens):
    """X 形状 (batch, n, m)，在最后一维做 softmax。

    valid_lens：每个样本（每个 query）的有效长度。
    None 时退化成普通 softmax。
    """
```

**三种情况**：

| `valid_lens` 形状 | 含义 |
|---|---|
| `None` | 不掩蔽，直接 softmax |
| `(batch,)` | 每个样本一个有效长度，该样本的所有 query 共用 |
| `(batch, n)` | 每个 query 各一个有效长度 |

**实现要点**：

1. 把 `(batch,)` 的长度扩展到 `(batch * n,)`（每个 query 重复一次）
2. 把 `X` 拍平成 `(batch * n, m)`
3. 对每一行，下标 `>= valid_len` 的位置填 `-1e6`
4. softmax
5. 变回原形状

**为什么用 `-1e6` 而不是 `-inf`**：`-inf` 在 `exp` 之后是 0，但梯度里会出现 `nan`
（`0 * inf`）。用有限的大负数，softmax 输出仍是 0，梯度是 0，不会 `nan`。

## 4 加性注意力

**适用场景**：查询与键的长度不同（`q` 维与 `k` 维不等）。

$$a(\boldsymbol{q}, \boldsymbol{k}) = \boldsymbol{w}_v^\top \tanh(\boldsymbol{W}_q\boldsymbol{q} + \boldsymbol{W}_k\boldsymbol{k})$$

**参数**：$\boldsymbol{W}_q\in\mathbb{R}^{h\times q}$、$\boldsymbol{W}_k\in\mathbb{R}^{h\times k}$、
$\boldsymbol{w}_v\in\mathbb{R}^{h}$。$h$ 是自己定的隐藏维。

**为什么叫"加性"**：$\boldsymbol{W}_q\boldsymbol{q}$ 与 $\boldsymbol{W}_k\boldsymbol{k}$ 是**相加**的，
再加 tanh 与一个线性层。

**形状**：

| 量 | 形状 |
|---|---|
| 查询 | `(batch, n_queries, q)` |
| 键 | `(batch, n_keys, k)` |
| 分数 | `(batch, n_queries, n_keys)` |
| 权重 | `(batch, n_queries, n_keys)` |
| 输出 | `(batch, n_queries, v)` |

**实现要注意广播**：查询与键要在"查询数"与"键数"这两个维度上扩成三维才能相加。

## 5 缩放点积注意力

**适用场景**：查询与键长度相同（都是 $d$）。

$$a(\boldsymbol{q}, \boldsymbol{k}) = \frac{\boldsymbol{q}^\top \boldsymbol{k}}{\sqrt{d}}$$

**为什么除以 $\sqrt{d}$**：假设 $\boldsymbol{q}$ 与 $\boldsymbol{k}$ 的元素是独立的零均值单位方差随机变量，
那么点积的**均值是 0、方差是 $d$**。$d$ 大时点积的数值范围很大，
softmax 会变得极端（几乎 one-hot），梯度接近 0。

除以 $\sqrt{d}$ 把方差压回 1。

**矢量化形式**（$n$ 个查询、$m$ 个键值对）：

$$\text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^\top}{\sqrt{d}}\right)\boldsymbol{V}$$

$\boldsymbol{Q}$ 形状 `(n, d)`、$\boldsymbol{K}$ 形状 `(m, d)`、$\boldsymbol{V}$ 形状 `(m, v)`。

**注意是 `softmax` 在最后一维**（键那一维）。

## 6 Bahdanau 注意力

把上面的机制接进 B22 的 seq2seq。

### 6.1 解码器每一步重新算背景变量

$$\boldsymbol{c}_{t'} = \sum_{t=1}^{T} \alpha_{t't}\boldsymbol{h}_t, \qquad
\alpha_{t't} = \text{softmax}_t\left(a(\boldsymbol{s}_{t'-1}, \boldsymbol{h}_t)\right)$$

**关键**：$t'$ 变了，$\boldsymbol{c}$ 就变。这就是与 B22 的差别。

### 6.2 解码器的 GRU 也要改

原来的 GRU：$\boldsymbol{s}_{t'} = g(y_{t'-1}, \boldsymbol{s}_{t'-1})$

**加了注意力之后**：$\boldsymbol{s}_{t'} = g(y_{t'-1}, \boldsymbol{s}_{t'-1}, \boldsymbol{c}_{t'})$

具体到三个门，每一项都多一个 $\boldsymbol{W}_{c*}\boldsymbol{c}_{t'}$：

$$\boldsymbol{r}_{t'} = \sigma(\boldsymbol{W}_{yr}\boldsymbol{y}_{t'-1} + \boldsymbol{W}_{sr}\boldsymbol{s}_{t'-1} + \boldsymbol{W}_{cr}\boldsymbol{c}_{t'} + \boldsymbol{b}_r)$$

$$\boldsymbol{z}_{t'} = \sigma(\boldsymbol{W}_{yz}\boldsymbol{y}_{t'-1} + \boldsymbol{W}_{sz}\boldsymbol{s}_{t'-1} + \boldsymbol{W}_{cz}\boldsymbol{c}_{t'} + \boldsymbol{b}_z)$$

$$\tilde{\boldsymbol{s}}_{t'} = \tanh(\boldsymbol{W}_{ys}\boldsymbol{y}_{t'-1} + \boldsymbol{W}_{ss}(\boldsymbol{s}_{t'-1}\odot\boldsymbol{r}_{t'}) + \boldsymbol{W}_{cs}\boldsymbol{c}_{t'} + \boldsymbol{b}_s)$$

**所以在有注意力的解码器里，GRU 的每个门都多一组权重**。

### 6.3 与 B22 的对比

| | B22 seq2seq | B22 + Bahdanau |
|---|---|---|
| 编码器 | 返回 $\boldsymbol{h}_T$ | 返回**所有** $\boldsymbol{h}_t$ |
| 解码器初始状态 | 直接用 $\boldsymbol{h}_T$ | $\boldsymbol{h}_T$ 只作初始，之后每步重算 $\boldsymbol{c}$ |
| 长输入 | 丢失信息 | 每个位置都能被"回头看" |

**代价**：计算量与内存都变大（每步要跟整个输入做一次注意力）。
但长序列上的效果提升明显。

## 7 实现会踩的坑

| 坑 | 症状 | 原因 |
|---|---|---|
| 掩蔽用了 `-inf` | 梯度出现 `nan` | `0 * inf`（3.2）|
| 掩蔽的位置没设成负无穷 | padding 参与注意力 | 3.1 |
| 加性注意力没做广播 | 形状不匹配 | 查询与键要在不同维度扩展（4）|
| 点积注意力忘了除以 $\sqrt{d}$ | 训练不稳 | $d$ 大时 softmax 过尖（5）|
| softmax 维搞错 | 权重在错误维度归一化 | 最后一维是键（5）|
| 注意力用编码器最后一步当键 | 只有一步，没意义 | 键要用**所有**编码器状态（6.1）|

## 命令速查

| 命令 | 作用 | 讲义哪一节 |
|---|---|---|
| `torch.softmax(x, dim=-1)` | 归一化 | 1 |
| `torch.bmm(Q, K.transpose(1, 2))` | 批量矩阵乘，算所有分数 | 5 |
| `torch.repeat_interleave(t, n)` | 把长度扩展成逐 query 一份 | 3.2 |
| `x.masked_fill(mask, -1e6)` | 掩蔽 | 3.2 |
| `nn.Linear(in, out)` | 打分函数里的投影 | 4 |
| `torch.tanh(x)` | 加性注意力的激活 | 4 |

## 自测题

1. 查询、键、值分别是什么？在 seq2seq 里各对应什么？
2. 注意力与 B22 的背景变量相比，解决了什么问题？
3. Nadaraya-Watson 里的打分函数是什么？它表达了什么直觉？
4. 掩蔽 softmax 为什么要用 `-1e6` 而不是 `-inf`？
5. 加性注意力与点积注意力各适用于什么情况？
6. 点积注意力为什么除以 $\sqrt{d}$？
7. Bahdanau 注意力的解码器与普通 seq2seq 的解码器差在哪？

## 答案（做完再看）

**1.** 查询是"我现在要什么"，键是"每个位置的标识"，值是"被加权的内容"。
seq2seq 里查询是解码器当前步的隐藏状态，键与值都是编码器各步的隐藏状态。

**2.** B22 把整个输入压成一个定长向量，长句子丢信息。
注意力每步重新看一遍整个输入、按相关性加权，长度不再受限于固定向量。

**3.** $a(q,k) = -\frac{1}{2}(q-k)^2$，负的平方距离。
直觉：离查询位置越近的训练点权重越大，预测值越接近那几个点。

**4.** `-inf` 参与 `exp` 得 0 没问题，但反向传播里会出现 `0 * inf = nan`。
用有限的 `-1e6`，softmax 输出是 0、梯度也是 0，不会 `nan`。

**5.** 加性注意力用于**查询与键长度不同**的情况（它把两者各自线性投影到同一个维度再相加）。
点积注意力要求**两者长度相同**，计算更快，是 Transformer 用的那种。

**6.** 假设元素独立、零均值单位方差，那么点积的方差是 $d$。$d$ 大时分数范围大、
softmax 过尖（接近 one-hot）、梯度接近 0。除以 $\sqrt{d}$ 把方差压回 1。

**7.** 普通解码器只用固定的背景变量；Bahdanau 的每一小步先算 $\boldsymbol{c}_{t'}$，
再把它送进 GRU 的每个门（多一组 $\boldsymbol{W}_{c*}$ 参数）。
