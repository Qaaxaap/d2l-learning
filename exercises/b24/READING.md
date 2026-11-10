# B24 阅读任务书

## 读什么

| 材料 | 位置 | 预估 |
|---|---|---|
| 讲义 | `notes/b24-transformer.md` | 45 分钟 |
| 纸质书 | **无**，第一版没有注意力之后的内容 | — |
| 电子版 | [10.5](https://zh.d2l.ai/chapter_attention-mechanisms/multihead-attention.html)、[10.6](https://zh.d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html)、[10.7](https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html) | 60 分钟 |

**纸质书完全没有这一章**。第一版（2019 年）的注意力只讲到 Bahdanau，Transformer 是
第二版才补上的。这一章以电子版为主。

## 读的时候要能回答

1. 多头注意力为什么有用？为什么头数增加不增加参数量？
2. `transpose_qkv` 的三步各做什么？为什么不能直接 reshape？
3. 自注意力相比 RNN 的优势与代价？
4. 位置编码为什么必要？为什么用正弦余弦？
5. 编码器块与解码器块各有几个子层？
6. 交叉注意力的查询、键、值分别来自哪里？
7. LayerNorm 与 BatchNorm 的区别？

## 顺序

先读讲义 1–2 节（多头与形状变换），`transpose_qkv` 是这一章唯一容易写错的地方。

再读 3–4 节（自注意力与位置编码）。位置编码的公式要能自己算出一两个值。

最后读 5–6 节（Transformer 与层规范化）。结构不难，难在记清每个子层的位置。

## 三处要留意

**`F.scaled_dot_product_attention` 的掩码语义**：bool 时 `True` 表示**允许**，
与 B23 的 `masked_softmax` 里"填 `-1e6` 遮住"正好相反。

**位置编码用 `register_buffer`**，不是 `nn.Parameter`。它由公式算出，不该训练。

**LayerNorm 在最后一维归一化**，与 BatchNorm 在批量维归一化不同。
