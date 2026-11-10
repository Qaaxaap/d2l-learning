# B24 口试题

关书作答。答不上来先想，说"不知道"我就讲。

**答题要求**：中文与代码描述即可，公式不必写 LaTeX。

题目分两档：

- **必会**（Q1、Q2、Q3、Q4）：写多头与 Transformer 时直接用
- **了解**（Q5、Q6、Q7）：知道设计动机即可

## 必会

**Q1** 多头注意力为什么有用？为什么增加头数不增加参数量？

<details><summary>参考答点</summary>

一个头只有一组投影，只能学**一种**相关性；多个头并行学**多个子空间**
（语法、指代、位置各不相同）。

参数量不变：每个头的维度是总维除以头数，$h \cdot (p/h) = p$，$h$ 约掉了。
多头不是靠加参数换效果，是靠同样的参数换多个视角。

</details>

**Q2** `transpose_qkv` 的三步各做什么？为什么不能直接 `reshape(-1, n, p // h)`？

<details><summary>参考答点</summary>

| 步 | 操作 | 形状 |
|---|---|---|
| 1 | `reshape(batch, n, h, -1)` | `(batch, n, h, p/h)` |
| 2 | `transpose(0, 2, 1, 3)` | `(batch, h, n, p/h)` |
| 3 | `reshape(-1, n, p/h)` | `(batch*h, n, p/h)` |

**不能直接 reshape**：reshape 按内存顺序切分，直接切成 `(batch*h, n, p/h)` 是
"每 p/h 个连续元素一组"，那是按**位置**切，不是按**头**切。
必须先拆出"头"这一维再挪位置。

</details>

**Q3** 自注意力相比 RNN 的优势是什么？代价是什么？

<details><summary>参考答点</summary>

**优势**：

- 并行：所有位置同时算，不用逐步递推
- 任意两位置的路径长度 $O(1)$（RNN 是 $O(n)$），远处依赖更容易学到

**代价**：$O(n^2)$ 的计算与内存。$n$ 个位置两两要算分数，长序列上很贵。

</details>

**Q4** LayerNorm 与 BatchNorm 的区别？为什么 NLP 用 LayerNorm？

<details><summary>参考答点</summary>

| | BatchNorm | LayerNorm |
|---|---|---|
| 归一化范围 | 同一特征跨**批量** | 同样本跨**特征** |
| 依赖批量大小 | 是 | 否 |

**NLP 用 LayerNorm**：序列长度不定，padding 会混进批量统计量；批量小时统计量不可靠。
LayerNorm 只看单个样本自己的特征维，不受这些影响。

</details>

## 了解

**Q5** 位置编码为什么必要？为什么用正弦余弦而不是可学习参数？

<details><summary>参考答点</summary>

自注意力**没有顺序概念**：交换两个位置，输出只是跟着交换。所以要显式加位置信息。

用正弦余弦的理由：

1. 固定公式，不占参数，任意长度都能直接算
2. **相对位置可以线性表示**：$\sin(i+\delta)$ 能写成 $\sin i$ 与 $\cos i$ 的组合，
   模型学得到"相隔 $\delta$ 个位置"

第 2 条是可学的位置编码没有的性质。

</details>

**Q6** 编码器块与解码器块各有几个子层？交叉注意力的查询键值从哪来？

<details><summary>参考答点</summary>

**编码器块 2 个**：多头自注意力、前馈网络。
**解码器块 3 个**：掩蔽多头自注意力、交叉注意力、前馈网络。

每个子层都包在 AddNorm 里：`LayerNorm(X + sublayer(X))`。

**交叉注意力**：查询来自解码器前一个块的输出，键和值来自编码器输出。
这是编码器与解码器之间唯一的信息通道。

</details>

**Q7** `F.scaled_dot_product_attention` 的 `attn_mask` 用 bool 时，`True` 表示什么？

<details><summary>参考答点</summary>

**`True` 表示允许这个位置参与**，`False` 表示遮住。

这与 B23 的 `masked_softmax` 相反——那里是"填 `-1e6` 表示遮住"。

记法：SDPA 的 mask 是"保留哪些"，masked_softmax 填的是"盖住哪些"。

</details>

## 评分

- **必会**四题全对才算通过
- Q2 是这一章最容易写错的地方
