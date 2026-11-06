# B23 阅读任务书

## 读什么

| 材料 | 位置 | 预估 |
|---|---|---|
| 讲义 | `notes/b23-attention.md` | 35 分钟 |
| 纸质书 | 10.11 注意力机制（只有 Bahdanau 一节，其余没有）| 15 分钟 |
| 电子版 | [10.1](https://zh.d2l.ai/chapter_attention-mechanisms/attention-cues.html)、[10.2](https://zh.d2l.ai/chapter_attention-mechanisms/nadaraya-waston.html)、[10.3](https://zh.d2l.ai/chapter_attention-mechanisms/attention-scoring-functions.html)、[10.4](https://zh.d2l.ai/chapter_attention-mechanisms/bahdanau-attention.html) | 40 分钟 |

**两版覆盖差异很大**：第一版的"注意力机制"只有 Bahdanau 那一个应用（10.11），
**没有** Nadaraya-Watson、没有掩蔽 softmax、也没有把评分函数单独抽出来讲。
这三块只有第二版有，所以这一章以电子版为主。

## 读的时候要能回答

1. 查询、键、值各是什么？seq2seq 里分别对应什么？
2. 注意力相比 B22 的固定背景变量，解决了什么问题？
3. Nadaraya-Watson 的打分函数与直觉？
4. 掩蔽 softmax 为什么用 `-1e6` 而不是 `-inf`？
5. 加性注意力与点积注意力各适用什么情况？
6. 点积注意力为什么除以 $\sqrt{d}$？

## 顺序

先读 10.1 与 10.2，把"查询—键—值"与"加权平均"这个基本结构弄清。
Nadaraya-Watson 是最简的注意力，没有神经网络，但结构完整。

再读 10.3 的三种评分函数与掩蔽 softmax。这些是后面 Transformer 的直接零件。

最后读 10.4 的 Bahdanau，看它怎么接回 B22 的 seq2seq。

## 两处要留意

**键与值常常是同一个东西**，但不是必须——自注意力里它们来自不同的投影。

**掩蔽的值必须是有限的大负数**。`-inf` 在前向没问题，反向会出 `nan`。
