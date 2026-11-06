# B21 GRU、LSTM、深层与双向 RNN

B20 的原始 RNN 有个问题：**长序列上的依赖学不到**。梯度沿时间连乘，要么消失要么爆炸。

这一章的两个门控单元——GRU 与 LSTM——用"门"控制信息的流动，让梯度有更通畅的传播路径。

## 与书对应

| 讲义小节 | 纸质书（第一版） | 电子版（第二版） |
|---|---|---|
| 1 门控解决什么 | 6.7 开头 | [9.1](https://zh.d2l.ai/chapter_recurrent-modern/gru.html) 开头 |
| 2 GRU | 6.7 全节 | 9.1 |
| 3 GRU 的两套约定（torch 与书上相反）| 无，本节补充 | 无，本节补充 |
| 4 LSTM | 6.8 全节 | [9.2](https://zh.d2l.ai/chapter_recurrent-modern/lstm.html) |
| 5 深层 RNN | 6.9 全节 | [9.3](https://zh.d2l.ai/chapter_recurrent-modern/deep-rnn.html) |
| 6 双向 RNN | 6.10 全节 | [9.4](https://zh.d2l.ai/chapter_recurrent-modern/bi-rnn.html) |
| 7 实现会踩的坑 | 散在正文里，本节汇总 | 同上 |
| 自测题 | 无 | 无 |

**解锁点**：v1 的 6.7 里 `gru` 函数、6.8 的 `lstm` 函数都标了"已保存在 d2lzh 中"，
6.9 与 6.10 的 `train_and_predict_rnn_nn` 也是。这一章要自己写门控单元的前向。

## 1 门控解决什么问题

原始 RNN 的隐藏状态每一步都**整体重写**：

$$\boldsymbol{H}_t = \tanh(\boldsymbol{X}_t\boldsymbol{W}_{xh} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{hh} + \boldsymbol{b}_h)$$

信息传了 50 步之后基本被冲掉。梯度沿时间连乘 $\boldsymbol{W}_{hh}^\top$，
特征值小于 1 就衰减到 0。

**门控的思路**：不要整体重写，而是**有选择地保留旧信息、有选择地接受新信息**。
门是一个 0 到 1 之间的向量，用 sigmoid 算出来，与信息做按元素乘。

- 门接近 1：让信息通过
- 门接近 0：挡住

## 2 GRU

### 2.1 三个公式

**重置门**（reset）与**更新门**（update）：

$$\boldsymbol{R}_t = \sigma(\boldsymbol{X}_t\boldsymbol{W}_{xr} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{hr} + \boldsymbol{b}_r)$$

$$\boldsymbol{Z}_t = \sigma(\boldsymbol{X}_t\boldsymbol{W}_{xz} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{hz} + \boldsymbol{b}_z)$$

**候选隐藏状态**：

$$\tilde{\boldsymbol{H}}_t = \tanh(\boldsymbol{X}_t\boldsymbol{W}_{xh} + (\boldsymbol{R}_t \odot \boldsymbol{H}_{t-1})\boldsymbol{W}_{hh} + \boldsymbol{b}_h)$$

**隐藏状态**：

$$\boldsymbol{H}_t = \boldsymbol{Z}_t \odot \boldsymbol{H}_{t-1} + (1 - \boldsymbol{Z}_t) \odot \tilde{\boldsymbol{H}}_t$$

**形状**：所有中间量都是 $\mathbb{R}^{n \times h}$（批量 × 隐藏维）。
权重：$\boldsymbol{W}_{x*}\in\mathbb{R}^{d\times h}$、$\boldsymbol{W}_{h*}\in\mathbb{R}^{h\times h}$。

### 2.2 两个门各管什么

**重置门 $\boldsymbol{R}_t$**：控制上一时刻隐藏状态**有多少进入候选状态**。
它先与 $\boldsymbol{H}_{t-1}$ 做按元素乘，再过矩阵乘。
$\boldsymbol{R}_t$ 接近 0 时，候选状态只看当前输入，历史被丢掉——管**短期依赖**。

**更新门 $\boldsymbol{Z}_t$**：控制**旧状态保留多少**。
$\boldsymbol{Z}_t = 1$ 时 $\boldsymbol{H}_t = \boldsymbol{H}_{t-1}$，这一时刻什么都不更新，
旧信息原样传下去——管**长期依赖**。

书上总结成两句：重置门捕捉短期依赖，更新门捕捉长期依赖。

**为什么能缓解梯度消失**：更新门接近 1 时，$\boldsymbol{H}_t = \boldsymbol{H}_{t-1}$，
梯度可以沿这条"直通路径"传回去，不需要每步乘 $\boldsymbol{W}_{hh}^\top$。

## 3 GRU 的两套约定

**这是这一章最容易踩的地方**：torch 的 `nn.GRU` 与书上（以及多数论文）的写法不同。

| | 书上 | torch `nn.GRU` |
|---|---|---|
| 更新门语义 | $\boldsymbol{Z}$ 是**保留旧状态**的比例 | $z$ 是**接受新状态**的比例 |
| 隐藏状态 | $\boldsymbol{H}_t = \boldsymbol{Z}\odot\boldsymbol{H}_{t-1} + (1-\boldsymbol{Z})\odot\tilde{\boldsymbol{H}}$ | $h_t = (1-z)\odot n_t + z\odot h_{t-1}$ |
| 候选状态的门控位置 | $(\boldsymbol{R}\odot\boldsymbol{H})\boldsymbol{W}_{hh}$：先门控再矩阵乘 | $r_t \odot (\boldsymbol{W}_{hn}h_{t-1} + b_{hn})$：先矩阵乘再门控 |

**两条差异叠加**，手写的 GRU 与 `nn.GRU` 直接对比一定对不上。

**这一章的做法**：
- **T1 按书上的公式实现**（这是理解门控机理的标准写法）
- **T2 按 torch 的约定实现**，与 `nn.GRU` 逐元素对比，确认差异真的在哪

**实践含义**：读别人的 GRU 实现时，先看它是哪一套。
`nn.GRU` 的文档里写得很清楚，但两个约定混用时不会报错，只会训练结果不同。

## 4 LSTM

### 4.1 三个门加一个候选

$$\boldsymbol{I}_t = \sigma(\boldsymbol{X}_t\boldsymbol{W}_{xi} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{hi} + \boldsymbol{b}_i)$$

$$\boldsymbol{F}_t = \sigma(\boldsymbol{X}_t\boldsymbol{W}_{xf} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{hf} + \boldsymbol{b}_f)$$

$$\boldsymbol{O}_t = \sigma(\boldsymbol{X}_t\boldsymbol{W}_{xo} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{ho} + \boldsymbol{b}_o)$$

$$\tilde{\boldsymbol{C}}_t = \tanh(\boldsymbol{X}_t\boldsymbol{W}_{xc} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{hc} + \boldsymbol{b}_c)$$

### 4.2 记忆细胞与隐藏状态

$$\boldsymbol{C}_t = \boldsymbol{F}_t \odot \boldsymbol{C}_{t-1} + \boldsymbol{I}_t \odot \tilde{\boldsymbol{C}}_t$$

$$\boldsymbol{H}_t = \boldsymbol{O}_t \odot \tanh(\boldsymbol{C}_t)$$

**LSTM 比 GRU 多一个状态**：$\boldsymbol{C}_t$（记忆细胞）与 $\boldsymbol{H}_t$（隐藏状态）分开。

| 量 | 作用 |
|---|---|
| $\boldsymbol{C}_t$ | 长期记忆，只经过加法和按元素乘，**没有矩阵乘** |
| $\boldsymbol{H}_t$ | 输出给下一层/下一步，是 $\boldsymbol{C}_t$ 的"门控版" |

**为什么这样设计**：$\boldsymbol{C}_t = \boldsymbol{F}_t\odot\boldsymbol{C}_{t-1} + \boldsymbol{I}_t\odot\tilde{\boldsymbol{C}}_t$
是**加法**关系。梯度沿 $\boldsymbol{C}$ 传回去时，乘的是 $\boldsymbol{F}_t$（一个 0 到 1 的向量），
不像原始 RNN 那样每步乘一个矩阵。$\boldsymbol{F}_t$ 接近 1 时梯度可以几乎无损地传很远。

这条路径常被叫作**恒定误差传送带**（constant error carousel）。

### 4.3 输入门与遗忘门的分工

$$\boldsymbol{C}_t = \underbrace{\boldsymbol{F}_t \odot \boldsymbol{C}_{t-1}}_{\text{保留多少旧记忆}} + \underbrace{\boldsymbol{I}_t \odot \tilde{\boldsymbol{C}}_t}_{\text{写入多少新内容}}$$

**两个门互补**：遗忘门管"删"，输入门管"写"。输出门 $\boldsymbol{O}_t$ 单独控制
"这一步给外面看多少"。

**torch 的 `nn.LSTM` 与书上约定一致**（`i`、`f`、`g`、`o` 与 $\boldsymbol{I}$、$\boldsymbol{F}$、$\tilde{\boldsymbol{C}}$、$\boldsymbol{O}$ 对应），
手写实现可以直接与它对比。

## 5 深层 RNN

**单层 RNN 的输出直接喂给下一层**，堆几层就是深层 RNN：

```python
nn.RNN(input_size=d, hidden_size=h, num_layers=3, batch_first=True)
```

**形状**：`num_layers` 层时，隐藏状态是 `(num_layers, batch, h)`。

**每一层都要维护自己的隐藏状态**，层与层之间用输出串起来。

**为什么要堆**：一层 RNN 的表达能力有限，多层能学到更高层的抽象（类似 CNN 堆卷积层）。

## 6 双向 RNN

**单向 RNN 只能看过去**。但做序列标注（词性、命名实体）时，后面的词也能帮助判断当前词。

**双向 RNN**：两个 RNN 分别从前往后、从后往前跑，输出拼接：

$$\overrightarrow{\boldsymbol{H}}_t = f(\boldsymbol{X}_t, \overrightarrow{\boldsymbol{H}}_{t-1}), \quad
\overleftarrow{\boldsymbol{H}}_t = f(\boldsymbol{X}_t, \overleftarrow{\boldsymbol{H}}_{t+1})$$

$$\boldsymbol{H}_t = [\overrightarrow{\boldsymbol{H}}_t; \overleftarrow{\boldsymbol{H}}_t]$$

**输出维度翻倍**（`2 * hidden_size`）。

**限制**：双向 RNN 需要看到整个序列，所以**不能用于在线预测或自回归生成**——
生成时未来的字符还不存在。

```python
nn.RNN(input_size=d, hidden_size=h, bidirectional=True, batch_first=True)
```

## 7 实现会踩的坑

| 坑 | 症状 | 原因 |
|---|---|---|
| 手写 GRU 与 `nn.GRU` 对不上 | 训练结果差很多 | 两套约定不同（3）|
| LSTM 少返回 `C` | 形状或数量不对 | 状态是 `(H, C)` 两个（4.2）|
| 深层 RNN 的初始状态形状 | 形状报错 | `(num_layers, batch, h)`（5）|
| 双向 RNN 的输出维度没翻倍 | 后续层形状错 | 拼接后是 `2h`（6）|
| 双向 RNN 用于生成 | 逻辑错 | 生成时看不到未来（6）|
| 门忘了 `sigmoid` | 值域不对、训练不稳 | 门必须在 [0,1]（2.1、4.1）|

## 命令速查

| 命令 | 作用 | 讲义哪一节 |
|---|---|---|
| `torch.sigmoid(x)` | 算门 | 2.1 |
| `torch.tanh(x)` | 候选状态 | 2.1、4.1 |
| `a * b`（同形状） | 按元素乘 | 2.1 |
| `nn.GRU(input_size, hidden_size, batch_first=True)` | 现成 GRU | 3 |
| `nn.LSTM(input_size, hidden_size, batch_first=True)` | 现成 LSTM | 4.3 |
| `nn.RNN(..., num_layers=3)` | 深层 RNN | 5 |
| `nn.RNN(..., bidirectional=True)` | 双向 RNN | 6 |

## 自测题

1. 门控解决什么问题？门的值域为什么是 [0,1]？
2. GRU 的重置门与更新门各控制什么？
3. GRU 的 `nn.GRU` 与书上公式有两处什么差异？
4. LSTM 的记忆细胞为什么能缓解梯度消失？
5. LSTM 与 GRU 的状态数量差在哪？
6. 深层 RNN 的隐藏状态是什么形状？
7. 双向 RNN 为什么不能用于生成？

## 答案（做完再看）

**1.** 解决长序列上的梯度消失——原始 RNN 每步整体重写隐藏状态，历史信息传不远。
门用 sigmoid 算，值域 [0,1]，与信息按元素乘起到"通过/挡住"的开关作用。

**2.** 重置门控制上一时刻隐藏状态有多少进入**候选状态**（管短期依赖）；
更新门控制旧状态保留多少（管长期依赖）。

**3.** （a）更新门语义相反：书上的 $\boldsymbol{Z}$ 是保留旧状态的比例，
torch 的 $z$ 是接受新状态的比例；
（b）候选状态里门控的位置不同：书上先门控再矩阵乘 $(\boldsymbol{R}\odot\boldsymbol{H})\boldsymbol{W}_{hh}$，
torch 先矩阵乘再门控 $r\odot(\boldsymbol{W}_{hn}h_{t-1}+b_{hn})$。

**4.** 记忆细胞之间是**加法**关系 $\boldsymbol{C}_t = \boldsymbol{F}_t\odot\boldsymbol{C}_{t-1} + \cdots$，
梯度沿它回传时乘的是遗忘门（0 到 1），不像原始 RNN 每步乘矩阵。
遗忘门接近 1 时梯度几乎无损。

**5.** GRU 只有隐藏状态 $\boldsymbol{H}$；LSTM 有记忆细胞 $\boldsymbol{C}$ 与隐藏状态 $\boldsymbol{H}$ 两份，
所以 `nn.LSTM` 返回的状态是元组 `(h_n, c_n)`。

**6.** `(num_layers, batch, hidden_size)`。双向时是 `(num_layers * 2, batch, hidden_size)`。

**7.** 双向 RNN 的后向部分要从序列末尾往前跑，生成时未来的字符还不存在。
