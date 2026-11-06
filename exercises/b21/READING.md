# B21 阅读任务书

## 读什么

| 材料 | 位置 | 预估 |
|---|---|---|
| 讲义 | `notes/b21-gru-lstm.md` | 35 分钟 |
| 纸质书 | 6.7 门控循环单元、6.8 长短期记忆、6.9 深度循环神经网络、6.10 双向循环神经网络 | 40 分钟 |
| 电子版 | [9.1](https://zh.d2l.ai/chapter_recurrent-modern/gru.html)、[9.2](https://zh.d2l.ai/chapter_recurrent-modern/lstm.html)、[9.3](https://zh.d2l.ai/chapter_recurrent-modern/deep-rnn.html)、[9.4](https://zh.d2l.ai/chapter_recurrent-modern/bi-rnn.html) | 40 分钟 |

## 读的时候要能回答

1. 门控解决什么问题？门的值域为什么是 [0,1]？
2. GRU 的重置门与更新门各控制什么？
3. GRU 的 `nn.GRU` 与书上公式有两处什么差异？
4. LSTM 的记忆细胞为什么能缓解梯度消失？
5. LSTM 与 GRU 的状态数量差在哪？
6. 双向 RNN 为什么不能用于生成？

## 顺序

先读 GRU（6.7）。三个公式：两个门 + 一个候选状态，再加一个组合式。

再读 LSTM（6.8）。比 GRU 多一个"记忆细胞"，理解它为什么能缓解梯度消失。

6.9 与 6.10 各只有一页，讲的是"怎么用"——多层堆叠与双向。

## 三处要留意

**`nn.GRU` 与书上约定不同**（讲义第 3 节）。手写时先确定用哪一套，
混合使用不会报错，只会训练结果不同。

**LSTM 的状态是 `(H, C)` 两个**，GRU 只有一个 `H`。

**双向 RNN 不能用于生成**——后向部分需要看到序列末尾。
