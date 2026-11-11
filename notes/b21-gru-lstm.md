# B21 GRU、LSTM、深层与双向 RNN

B20 的原始 RNN 有一个问题：长序列上的依赖学不到。梯度沿时间连乘，结果要么消失要么爆炸。

本章讲两个门控单元。GRU 与 LSTM 用「门」控制信息的流动，
让梯度有一条更通畅的传播路径。

## 与书对应

| 讲义小节 | 纸质书（第一版） | 电子版（第二版） |
|---|---|---|
| 1 门控解决什么问题 | 6.7 开头 | [9.1](https://zh.d2l.ai/chapter_recurrent-modern/gru.html) 开头 |
| 2 GRU | 6.7 全节 | 9.1 |
| 3 GRU 的两套约定 | 无，本节补充 | 无，本节补充 |
| 4 LSTM | 6.8 全节 | [9.2](https://zh.d2l.ai/chapter_recurrent-modern/lstm.html) |
| 5 深层 RNN | 6.9 全节 | [9.3](https://zh.d2l.ai/chapter_recurrent-modern/deep-rnn.html) |
| 6 双向 RNN | 6.10 全节 | [9.4](https://zh.d2l.ai/chapter_recurrent-modern/bi-rnn.html) |
| 7 实现会踩的坑 | 散在正文里，本节汇总 | 同上 |
| 自测题 | 无 | 无 |

**解锁点**：v1 的 6.7 与 6.8 里 `gru`、`lstm` 两个函数都标了"已保存在 d2lzh 中"。
本章要自己写门控单元的前向。

## 1 门控解决什么问题

### 1.1 原始 RNN 的问题

原始 RNN 每一步整体重写隐藏状态：

$$\boldsymbol{H}_t = \tanh(\boldsymbol{X}_t\boldsymbol{W}_{xh} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{hh} + \boldsymbol{b}_h)$$

三个后果：

- **信息被冲掉**。传 50 步之后，第 1 步的信息基本没了。
- **梯度沿时间连乘**。B20 的 5.4 节推导过，$\boldsymbol{W}_{hh}^\top$ 连乘会让梯度指数衰减或爆炸。
- **无法选择性保留**。每一步都整体覆盖，模型没法决定「这一步该记住什么」。

### 1.2 门的机制

门控的思路是不整体重写，改为有选择地保留与接受：

- 用 0 到 1 之间的向量控制「保留多少旧信息」。
- 用另一个向量控制「接受多少新信息」。

这个 0 到 1 的向量就是**门**。

门的值接近 1 时信息通过。接近 0 时信息被挡住。

### 1.3 门怎么算

门用 sigmoid 算：

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

sigmoid 的值域是 $(0, 1)$，正好表示「通过的比例」。

然后用按元素乘（$\odot$）把门作用到信息上：

- 门接近 1：原值保留。
- 门接近 0：原值被压成 0。

代码里写 `torch.sigmoid(z)`。

**为什么不能用 tanh**：tanh 的值域是 $[-1, 1]$，会出现负的「通过比例」，
语义上说不通。

## 2 GRU

GRU（gated recurrent unit，门控循环单元）用两个门。

### 2.1 两个门

**重置门**（reset gate）：

$$\boldsymbol{R}_t = \sigma(\boldsymbol{X}_t\boldsymbol{W}_{xr} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{hr} + \boldsymbol{b}_r)$$

**更新门**（update gate）：

$$\boldsymbol{Z}_t = \sigma(\boldsymbol{X}_t\boldsymbol{W}_{xz} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{hz} + \boldsymbol{b}_z)$$

两个门的输入相同：当前步输入与上一步隐藏状态。参数各自独立。

形状都是 $\mathbb{R}^{n \times h}$。

### 2.2 候选隐藏状态

$$\tilde{\boldsymbol{H}}_t = \tanh(\boldsymbol{X}_t\boldsymbol{W}_{xh} + (\boldsymbol{R}_t \odot \boldsymbol{H}_{t-1})\boldsymbol{W}_{hh} + \boldsymbol{b}_h)$$

与原始 RNN 的差别在第二项：$\boldsymbol{H}_{t-1}$ 先乘上重置门，再进矩阵乘。

代码里 tanh 写 `torch.tanh(x)`。

**重置门的作用**：控制上一步的隐藏状态有多少进入候选状态。

- 门接近 1：保留完整历史，候选状态看当前输入加完整历史。
- 门接近 0：历史被清零，候选状态只看当前输入。

所以重置门管**短期依赖**。需要丢掉无关历史时，把门压到 0。

### 2.3 隐藏状态

$$\boldsymbol{H}_t = \boldsymbol{Z}_t \odot \boldsymbol{H}_{t-1} + (1 - \boldsymbol{Z}_t) \odot \tilde{\boldsymbol{H}}_t$$

门与信息之间用按元素乘，代码里就是同形状张量直接写 `a * b`。

更新门 $\boldsymbol{Z}_t$ 在旧状态与候选状态之间做插值：

- $\boldsymbol{Z}_t = 1$：$\boldsymbol{H}_t = \boldsymbol{H}_{t-1}$，这一步什么都不更新。
- $\boldsymbol{Z}_t = 0$：$\boldsymbol{H}_t = \tilde{\boldsymbol{H}}_t$，完全换成候选状态。

**注意 $\boldsymbol{Z}_t$ 是保留旧状态的比例**。书上的约定如此。
第 3 节会讲 torch 的约定相反。

### 2.4 两个门各管什么

书上的总结是两句：

- 重置门有助于捕捉时间序列里**短期**的依赖关系。
- 更新门有助于捕捉时间序列里**长期**的依赖关系。

长期依赖来自更新门的特性。$\boldsymbol{Z}_t$ 接近 1 时，
较早时刻的隐藏状态可以通过时间原样传下来，中间的信息几乎不流入。

### 2.5 为什么能缓解梯度消失

看 $\boldsymbol{H}_t = \boldsymbol{Z}_t \odot \boldsymbol{H}_{t-1} + \cdots$ 这一项。

对 $\boldsymbol{H}_{t-1}$ 求导，得到的是 $\boldsymbol{Z}_t$，不是矩阵 $\boldsymbol{W}_{hh}^\top$。

于是梯度沿时间回传时，乘的是一串 0 到 1 之间的**向量**，不是矩阵的连乘。
遗忘门接近 1 时，这条路径的梯度几乎不衰减。

**这就是「直通路径」**：原始 RNN 每步都要穿过矩阵乘，GRU 提供了一条只穿过门的路径。

## 3 GRU 的两套约定

### 3.1 差异在哪

torch 的 `nn.GRU` 与书上（以及多数论文）的写法不同。差异有两处。

**差异一：更新门的语义相反。**

| | 书上 | torch |
|---|---|---|
| $\boldsymbol{Z}$ 表示 | 保留旧状态的比例 | 接受新状态的比例 |
| 隐藏状态 | $\boldsymbol{Z}\odot\boldsymbol{H}_{t-1} + (1-\boldsymbol{Z})\odot\tilde{\boldsymbol{H}}_t$ | $(1-z)\odot n_t + z\odot h_{t-1}$ |

书上的 $\boldsymbol{Z} = 1$ 表示不更新。torch 的 $z = 1$ 表示全部换成新状态。

**差异二：候选状态里门控的位置不同。**

- 书上：$(\boldsymbol{R}_t \odot \boldsymbol{H}_{t-1})\boldsymbol{W}_{hh}$。先门控，再矩阵乘。
- torch：$r_t \odot (\boldsymbol{W}_{hn}h_{t-1} + b_{hn})$。先矩阵乘，再门控。

两种写法**不等价**。$\boldsymbol{R}\odot(\boldsymbol{H}\boldsymbol{W})$ 与
$(\boldsymbol{R}\odot\boldsymbol{H})\boldsymbol{W}$ 是两种不同的运算。

### 3.2 torch 的完整公式

```
r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
h_new = (1 - z) * n + z * h
```

### 3.3 实践含义

手写的 GRU 与 `nn.GRU` 直接对比一定对不上。两个差异叠加，结果差很远。

改别人的 GRU 实现时，先看它用哪一套。两个约定混用不会返回错误，
只会训练结果不同，很难排查。

B21 的代码题让两种都写一遍：先按书上的写，再按 torch 的写并与 `nn.GRU` 对比。

## 4 LSTM

### 4.1 三个门与候选

LSTM（long short-term memory）用三个门加一个候选记忆。

$$\boldsymbol{I}_t = \sigma(\boldsymbol{X}_t\boldsymbol{W}_{xi} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{hi} + \boldsymbol{b}_i)$$

$$\boldsymbol{F}_t = \sigma(\boldsymbol{X}_t\boldsymbol{W}_{xf} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{hf} + \boldsymbol{b}_f)$$

$$\boldsymbol{O}_t = \sigma(\boldsymbol{X}_t\boldsymbol{W}_{xo} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{ho} + \boldsymbol{b}_o)$$

$$\tilde{\boldsymbol{C}}_t = \tanh(\boldsymbol{X}_t\boldsymbol{W}_{xc} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{hc} + \boldsymbol{b}_c)$$

四个式子结构相同：当前输入加上一步隐藏状态，过线性层，再激活。

- **输入门** $\boldsymbol{I}_t$：控制写入多少新内容。
- **遗忘门** $\boldsymbol{F}_t$：控制保留多少旧记忆。
- **输出门** $\boldsymbol{O}_t$：控制给外面看多少。
- **候选记忆** $\tilde{\boldsymbol{C}}_t$：这一步想写入的内容。

### 4.2 记忆细胞

$$\boldsymbol{C}_t = \boldsymbol{F}_t \odot \boldsymbol{C}_{t-1} + \boldsymbol{I}_t \odot \tilde{\boldsymbol{C}}_t$$

这是一个**加法**关系，不是覆盖。

两项的含义：

- $\boldsymbol{F}_t \odot \boldsymbol{C}_{t-1}$：保留多少旧记忆。
- $\boldsymbol{I}_t \odot \tilde{\boldsymbol{C}}_t$：写入多少新内容。

**LSTM 比 GRU 多一个状态**。它有记忆细胞 $\boldsymbol{C}_t$ 与隐藏状态 $\boldsymbol{H}_t$ 两个。

### 4.3 隐藏状态

$$\boldsymbol{H}_t = \boldsymbol{O}_t \odot \tanh(\boldsymbol{C}_t)$$

隐藏状态是记忆细胞的「门控版」：

1. 先把记忆细胞过 tanh，值域压到 $[-1, 1]$。
2. 再乘输出门，决定给外面看多少。

所以 $\boldsymbol{C}_t$ 是内部的长期记忆，$\boldsymbol{H}_t$ 是对外的输出。

### 4.4 为什么能缓解梯度消失

关键在于 $\boldsymbol{C}_t = \boldsymbol{F}_t \odot \boldsymbol{C}_{t-1} + \cdots$ 是加法。

对 $\boldsymbol{C}_{t-1}$ 求导，得到的是 $\boldsymbol{F}_t$，一个 0 到 1 之间的向量。

梯度沿 $\boldsymbol{C}$ 回传时，乘的是一串遗忘门，不是矩阵连乘。
遗忘门接近 1 时，梯度几乎无损地传很远。

书上把这条路径叫**恒定误差传送带**（constant error carousel）。

对比原始 RNN：每一步都要乘 $\boldsymbol{W}_{hh}^\top$，特征值小于 1 时很快衰减到 0。

### 4.5 LSTM 与 GRU 的对比

| | GRU | LSTM |
|---|---|---|
| 状态个数 | 1 个（$\boldsymbol{H}$）| 2 个（$\boldsymbol{C}$、$\boldsymbol{H}$）|
| 门个数 | 2 个 | 3 个 |
| 参数量 | 少 | 多（约为 GRU 的 1.3 倍）|
| 长依赖 | 较好 | 通常更好 |
| 计算量 | 小 | 大 |

**实践选择**：序列不太长、资源紧张时用 GRU。
长依赖重要、数据量足够时用 LSTM。两者效果差异在多数任务上不大。

**torch 的 `nn.LSTM` 与书上约定一致**，门顺序是 `i`、`f`、`g`、`o`。
手写实现可以直接与它对比。

## 5 深层 RNN

### 5.1 结构

单层 RNN 的输出直接喂给下一层。堆几层就是深层 RNN。

```python
nn.RNN(input_size=d, hidden_size=h, num_layers=3, batch_first=True)
```

### 5.2 形状

`num_layers` 层时，隐藏状态形状是 `(num_layers, batch, h)`。

每一层维护自己的隐藏状态。层与层之间用输出串起来：
第 $l$ 层的输出是第 $l+1$ 层的输入。

### 5.3 为什么要堆

一层 RNN 的表达能力有限。多层能学到更高层的抽象。

这与 CNN 堆卷积层的思路相同。B16 的 LeNet 是两层卷积，B17 的 VGG 堆到 16 层。

### 5.4 代价

- 参数量随层数增加。
- 梯度要穿过更多层，训练更难。
- 深层 RNN 更容易梯度消失。

所以深层 RNN 一般不超过 3 到 5 层，且要配梯度裁剪或残差连接。

## 6 双向 RNN

### 6.1 结构

单向 RNN 只能看过去。做序列标注（词性、命名实体）时，
后面的词也能帮助判断当前词。

双向 RNN 用两个 RNN，一个从前往后，一个从后往前：

$$\overrightarrow{\boldsymbol{H}}_t = f(\boldsymbol{X}_t, \overrightarrow{\boldsymbol{H}}_{t-1})$$
$$\overleftarrow{\boldsymbol{H}}_t = f(\boldsymbol{X}_t, \overleftarrow{\boldsymbol{H}}_{t+1})$$

两个方向的输出在通道维拼接：

$$\boldsymbol{H}_t = [\overrightarrow{\boldsymbol{H}}_t; \overleftarrow{\boldsymbol{H}}_t]$$

### 6.2 形状与参数

```python
nn.RNN(input_size=d, hidden_size=h, bidirectional=True, batch_first=True)
```

| 量 | 形状 |
|---|---|
| 输出最后一维 | `2 * h` |
| 隐藏状态第一维 | `num_layers * 2` |

隐藏状态的第一维翻倍，因为两个方向各有一份状态。

### 6.3 限制

双向 RNN 的后向部分要从序列末尾往前跑。它需要看到整个序列。

所以双向 RNN **不能用于在线预测或自回归生成**。生成时未来的字符还不存在。

适用场景是可以一次拿到整段输入的任务：

- 序列标注（词性、命名实体）
- 句分类（情感分析）
- 机器阅读理解

不适用场景：语言模型、机器翻译的解码器。

## 7 实现会踩的坑

| 坑 | 症状 | 原因 |
|---|---|---|
| 手写 GRU 与 `nn.GRU` 对不上 | 训练结果差很多 | 两套约定不同（3）|
| LSTM 少返回 `C` | 形状或数量不对 | 状态是 `(H, C)` 两个（4.2）|
| 深层 RNN 的初始状态形状 | 形状不匹配 | `(num_layers, batch, h)`（5.2）|
| 双向 RNN 的输出维度没翻倍 | 后续层形状不匹配 | 拼接后是 `2h`（6.2）|
| 双向 RNN 用于生成 | 逻辑错误 | 生成时看不到未来（6.3）|
| 门忘了 sigmoid | 值域不对、训练不稳 | 门必须在 $(0,1)$（1.3、2.1、4.1）|
| 候选状态忘了 tanh | 值域不对 | 候选状态是 tanh（2.2、4.1）|

## 命令速查

| 命令 | 作用 | 讲义哪一节 |
|---|---|---|
| `torch.sigmoid(x)` | 算门 | 1.3 |
| `torch.tanh(x)` | 候选状态 | 2.2 |
| `a * b`（同形状） | 按元素乘 | 2.3 |
| `nn.GRU(input_size, hidden_size, batch_first=True)` | 现成 GRU | 3.2 |
| `nn.LSTM(input_size, hidden_size, batch_first=True)` | 现成 LSTM | 4.5 |
| `nn.RNN(..., num_layers=3)` | 深层 RNN | 5.1 |
| `nn.RNN(..., bidirectional=True)` | 双向 RNN | 6.2 |

## 自测题

1. 说明原始 RNN 的三个问题，以及门控用哪两步解决它们。
2. 写出 GRU 的重置门与更新门公式。
3. 说明 GRU 重置门与更新门各自控制什么。
4. 写出 GRU 隐藏状态的组合公式，并判断 $\boldsymbol{Z}_t = 1$ 时会发生什么。
5. 写出 torch `nn.GRU` 与书上公式的两处差异。
6. 写出 LSTM 的三个门与候选记忆的公式。
7. 写出 LSTM 记忆细胞与隐藏状态的公式，说明两者的分工。
8. 说明 LSTM 缓解梯度消失的原因，指出关键的运算类型。
9. 对比 LSTM 与 GRU 的状态个数、门个数与适用场景。
10. 说明深层 RNN 的隐藏状态形状，以及为什么不堆太多层。
11. 判断双向 RNN 能否用于生成任务，并说明理由。

## 答案（做完再看）

**1.** 三个问题：信息被冲掉、梯度沿时间连乘、无法选择性保留。
门控用两步：用 sigmoid 算出 0 到 1 的「通过比例」，再用按元素乘把门作用到信息上。

**2.**
$\boldsymbol{R}_t = \sigma(\boldsymbol{X}_t\boldsymbol{W}_{xr} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{hr} + \boldsymbol{b}_r)$、
$\boldsymbol{Z}_t = \sigma(\boldsymbol{X}_t\boldsymbol{W}_{xz} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{hz} + \boldsymbol{b}_z)$。

**3.** 重置门控制上一步隐藏状态有多少进入**候选状态**，管短期依赖。
更新门控制旧状态保留多少，管长期依赖。

**4.** $\boldsymbol{H}_t = \boldsymbol{Z}_t \odot \boldsymbol{H}_{t-1}
+ (1 - \boldsymbol{Z}_t) \odot \tilde{\boldsymbol{H}}_t$。
$\boldsymbol{Z}_t = 1$ 时 $\boldsymbol{H}_t = \boldsymbol{H}_{t-1}$，这一步不更新。

**5.** 两处差异如下。

第一，更新门语义相反。书上 $\boldsymbol{Z}$ 是保留旧状态的比例，
torch 的 $z$ 是接受新状态的比例。

第二，候选状态里门控的位置不同。
书上先门控再矩阵乘：$(\boldsymbol{R}\odot\boldsymbol{H})\boldsymbol{W}_{hh}$。
torch 先矩阵乘再门控：$r\odot(\boldsymbol{W}_{hn}h+b_{hn})$。

**6.** 三个门的公式结构相同：

$\boldsymbol{I}_t$、$\boldsymbol{F}_t$、$\boldsymbol{O}_t$ 各自是
$\sigma(\boldsymbol{X}_t\boldsymbol{W}_{x*} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{h*} + \boldsymbol{b}_*)$。

候选记忆用 tanh：
$\tilde{\boldsymbol{C}}_t = \tanh(\boldsymbol{X}_t\boldsymbol{W}_{xc} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{hc} + \boldsymbol{b}_c)$。

**7.**
$\boldsymbol{C}_t = \boldsymbol{F}_t \odot \boldsymbol{C}_{t-1} + \boldsymbol{I}_t \odot \tilde{\boldsymbol{C}}_t$、
$\boldsymbol{H}_t = \boldsymbol{O}_t \odot \tanh(\boldsymbol{C}_t)$。

$\boldsymbol{C}_t$ 是内部的长期记忆。$\boldsymbol{H}_t$ 是对外的门控输出。

**8.** 记忆细胞之间是**加法**关系。
梯度沿 $\boldsymbol{C}$ 回传时乘的是遗忘门（0 到 1 的向量），不是矩阵连乘。
遗忘门接近 1 时梯度几乎无损，这条路径叫恒定误差传送带。

**9.** GRU 一个状态（$\boldsymbol{H}$）、两个门。LSTM 两个状态（$\boldsymbol{C}$、$\boldsymbol{H}$）、三个门。
资源紧张或序列不长时用 GRU。长依赖重要时用 LSTM。

**10.** 形状 `(num_layers, batch, h)`。
不堆太多层的原因：参数量增加、梯度要穿更多层、深层 RNN 更容易梯度消失。
一般不超过 3 到 5 层。

**11.** 不能。后向部分要从序列末尾往前跑，需要看到整个序列。
生成时未来的字符还不存在。
