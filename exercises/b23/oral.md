# B23 口试题

关书作答。答不上来先想，说"不知道"我就讲。

**答题要求**：中文与代码描述即可，公式不必写 LaTeX。

题目分两档：

- **必会**（Q1、Q2、Q3、Q4）：理解与实现注意力的核心
- **了解**（Q5、Q6、Q7）：知道设计动机即可

## 必会

**Q1** 查询、键、值各是什么？注意力怎么用它们算出背景变量？

<details><summary>参考答点</summary>

| 概念 | 含义 | seq2seq 里 |
|---|---|---|
| 查询 | 我现在想要什么 | 解码器当前步的隐藏状态 |
| 键 | 每个位置的"标识" | 编码器各步隐藏状态 |
| 值 | 被加权平均的内容 | 同样是编码器各步隐藏状态 |

三步：

1. 查询与每个键算分数 $e_i = a(\boldsymbol{q}, \boldsymbol{k}_i)$
2. 分数过 softmax 得权重 $\alpha_i$
3. 加权求和 $\boldsymbol{c} = \sum_i \alpha_i \boldsymbol{v}_i$

</details>

**Q2** 注意力相比 B22 的固定背景变量，解决了什么问题？

<details><summary>参考答点</summary>

B22 把整个输入压成一个**定长**向量 $\boldsymbol{h}_T$，输入再长也是同一个形状，
长句子必然丢信息。

注意力让解码器**每一步重新看一遍整个输入**，按相关性加权。
每个位置的信息都保留着，需要哪个就取哪个，不受固定向量的容量限制。

代价：计算量与显存都变大（每步要跟整个输入算一次注意力）。

</details>

**Q3** 掩蔽 softmax 是干什么的？为什么用 `-1e6` 而不是 `-inf`？

<details><summary>参考答点</summary>

批量里序列长度不同，短的要 padding。**padding 位置不该参与注意力**，
所以把它们的分数设成极大负数，softmax 后权重接近 0。

用 `-1e6` 而不是 `-inf` 的原因：`-inf` 参与 `exp` 得到 0 没问题，
但反向传播时会出现 `0 * inf`，梯度变成 `nan`。有限大负数让 softmax 输出 0、
梯度也是 0，不会 `nan`。

</details>

**Q4** 点积注意力为什么要除以 $\sqrt{d}$？

<details><summary>参考答点</summary>

假设 $\boldsymbol{q}$ 与 $\boldsymbol{k}$ 的元素独立、零均值、单位方差，
那么点积的均值是 0、**方差是 $d$**。

$d$ 大时分数范围很大，softmax 会变得极端（接近 one-hot），
梯度接近 0，训练不动。除以 $\sqrt{d}$ 把方差压回 1。

</details>

## 了解

**Q5** 加性注意力与点积注意力各适用于什么情况？

<details><summary>参考答点</summary>

**加性**：$a(\boldsymbol{q},\boldsymbol{k}) = \boldsymbol{w}_v^\top\tanh(\boldsymbol{W}_q\boldsymbol{q} + \boldsymbol{W}_k\boldsymbol{k})$。
两者长度可以不同，各投影到同一个隐藏维再相加。参数多、慢。

**点积**：$a = \boldsymbol{q}^\top\boldsymbol{k}/\sqrt{d}$。
要求两者长度相同。没有额外参数，可以用矩阵乘一次算完所有分数，快。
Transformer 用的是缩放点积。

</details>

**Q6** Nadaraya-Watson 里的打分函数是什么？它表达了什么直觉？

<details><summary>参考答点</summary>

$a(q, k) = -\frac{1}{2}(q - k)^2$，负的平方距离。

直觉：离查询位置越近的训练点权重越大，预测值越接近那几个点的标签值。
$w$ 控制"关注范围"——$w$ 大时只看很近的点（曲线尖锐），$w=0$ 时完全均匀（直线）。

这个模型没有神经网络，但结构就是完整的注意力。

</details>

**Q7** Bahdanau 注意力的解码器与 B22 的解码器差在哪？

<details><summary>参考答点</summary>

**B22**：背景变量固定，解码器从 $\boldsymbol{h}_T$ 出发，GRU 每步只看 $\boldsymbol{y}_{t'-1}$ 与
$\boldsymbol{s}_{t'-1}$。

**Bahdanau**：

1. 编码器返回**所有** $\boldsymbol{h}_t$，不再只返回 $\boldsymbol{h}_T$
2. 每一步先用 $\boldsymbol{s}_{t'-1}$ 与所有 $\boldsymbol{h}_t$ 算 $\boldsymbol{c}_{t'}$
3. GRU 的每个门都多一组 $\boldsymbol{W}_{c*}\boldsymbol{c}_{t'}$ 项

代价是计算量与显存变大，但长序列上效果好得多。

</details>

## 评分

- **必会**四题全对才算通过
- Q3 与 Q4 是最容易实现错的两处
