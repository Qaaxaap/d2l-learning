# B19 语言模型与循环神经网络

前十八章处理的是**定长输入**：一张图、一行特征。这一章开始处理**序列**——
文本、语音、时间序列，它们的长度不固定，而且**顺序有意义**。

核心是两件事：**怎么把文本变成能训练的数据**（这一章的主体），
以及**一个能处理序列的层**（RNN）。

## 与书对应

| 讲义小节 | 纸质书（第一版） | 电子版（第二版） |
|---|---|---|
| 1 语言模型 | 6.1 全节 | [8.1](https://zh.d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html) |
| 2 n 元语法 | 6.1 的"n元语法" | 8.1 的对应小节 |
| 3 循环神经网络 | 6.2 全节 | [8.2](https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn.html) |
| 4 字符级语言模型的输入与标签 | 6.2 的"应用"一节 | 8.1 的"字符级语言模型" |
| 5 数据管道 | 6.3 全节 | 8.1 的"读取数据集"与"时序数据的采样" |
| 6 实现会踩的坑 | 散在正文里，本节汇总 | 同上 |
| 自测题 | 无 | 无 |

**解锁点**：v1 的 6.3 里 `data_iter_random` 与 `data_iter_consecutive` 都标了
"本函数已保存在 d2lzh 包中方便以后使用"，`load_data_jay_lyrics` 也是。
所以这一章的数据管道要自己写。

**这一章不训练网络**。数据管道单独成题，RNN 的前向单独成题，训练放在 B20。

## 0 这一单元解决什么问题

**语言模型**要回答：一个句子出现的概率是多少？

$$P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, \ldots, w_{t-1})$$

直接算这个连乘有两个困难：

1. **条件越来越长**：第 $T$ 个词要依赖前面 $T-1$ 个词，参数按指数增长
2. **序列长度不固定**：不能像图像那样固定输入尺寸

这一章的两个技术分别对付它们：**n 元语法**砍掉远处依赖，**RNN**用隐藏状态压缩历史。

## 1 n 元语法

### 1.1 马尔可夫假设

假设一个词只与前面 $n-1$ 个词有关，那么

$$P(w_1, \ldots, w_T) \approx \prod_{t=1}^{T} P(w_t \mid w_{t-(n-1)}, \ldots, w_{t-1})$$

这叫作 **$n$ 元语法**（$n$-grams）：

| $n$ | 名称 | 用前面几个词 |
|---|---|---|
| 1 | 一元语法 | 0 个（各词独立）|
| 2 | 二元语法 | 1 个 |
| 3 | 三元语法 | 2 个 |

**代价**：$n$ 小则不准。书上举的例子——一元语法下"你走先"与"你先走"概率相同。
$n$ 大则要存的词频组合爆炸。

**为什么还要用**：$n$ 元语法是**统计方法**，参数是词频表，不需要训练。
它给 RNN 提供了一个对照基线。

### 1.2 用词频估算概率

$$P(w_2 \mid w_1) = \frac{\text{count}(w_1, w_2)}{\text{count}(w_1)}$$

**这不需要神经网络**。理解它有个用处：知道"语言模型"这件事本身衡量什么，
以及为什么统计方法在大语料上会撞上稀疏性问题（很多组合没出现过，频率为 0）。

## 2 循环神经网络

### 2.1 隐藏状态

普通全连接层每步独立处理输入。RNN 多保存一个**隐藏状态** $\boldsymbol{H}_t$：

$$\boldsymbol{H}_t = \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hh} + \boldsymbol{b}_h)$$

$$\boldsymbol{O}_t = \boldsymbol{H}_t \boldsymbol{W}_{hq} + \boldsymbol{b}_q$$

**形状**（设批量 $n$、输入维 $d$、隐藏维 $h$、输出维 $q$）：

| 符号 | 形状 | 说明 |
|---|---|---|
| $\boldsymbol{X}_t$ | $(n, d)$ | 第 $t$ 步的输入 |
| $\boldsymbol{H}_{t-1}$、$\boldsymbol{H}_t$ | $(n, h)$ | 隐藏状态 |
| $\boldsymbol{W}_{xh}$ | $(d, h)$ | 输入到隐藏 |
| $\boldsymbol{W}_{hh}$ | $(h, h)$ | 上一步隐藏到这一步隐藏 |
| $\boldsymbol{b}_h$ | $(h,)$ | |
| $\boldsymbol{W}_{hq}$ | $(h, q)$ | 隐藏到输出 |
| $\boldsymbol{O}_t$ | $(n, q)$ | |

### 2.2 参数共享

**不同时间步用同一组参数**。序列有 100 步，参数也是这五组，不会变成 500 组。

**这是 RNN 与"每步一个全连接层"的根本区别**：参数数量与序列长度无关。
书上写"循环神经网络模型参数的数量不随时间步的增加而增长"。

### 2.3 拼接写法

$$\boldsymbol{X}_t \boldsymbol{W}_{xh} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hh}$$

等价于把 $\boldsymbol{X}_t$ 与 $\boldsymbol{H}_{t-1}$ **按列拼**、把两个权重**按行拼**，再做一次矩阵乘：

```python
torch.cat([X, H], dim=1) @ torch.cat([W_xh, W_hh], dim=0)
```

前者形状 `(n, d+h)`，后者 `(d+h, h)`，乘出来 `(n, h)`。**结果与分开算再相加完全一样。**

这个等价性在实现里有意义：合并成一次矩阵乘，GPU 上更快。

### 2.4 与 `nn.RNN` 的对应

```python
rnn = nn.RNN(input_size=d, hidden_size=h, num_layers=1, batch_first=True)
out, h_n = rnn(x, h_0)
```

| 参数 | 形状 | 对应手工实现的 |
|---|---|---|
| `x` | `(batch, seq_len, d)` | 整个序列的输入 |
| `h_0` | `(num_layers, batch, h)` | 初始隐藏状态 |
| `out` | `(batch, seq_len, h)` | 每一步的 $\boldsymbol{H}_t$ |
| `h_n` | `(num_layers, batch, h)` | 最后一步的隐藏状态 |

**`nn.RNN` 内部用的是 $\tanh$**，不指定激活函数时就是它。

**`batch_first=True` 不能忘**：torch 的 RNN 默认期望 `(seq_len, batch, d)`，
加了 `batch_first=True` 才是 `(batch, seq_len, d)`。

## 3 字符级语言模型的输入与标签

### 3.1 输入与标签错一位

对文本 `"想要有直升机"`，设时间步 5：

| | 内容 |
|---|---|
| **输入** $\boldsymbol{X}$ | 想 要 有 直 升 |
| **标签** $\boldsymbol{Y}$ | 要 有 直 升 机 |

**标签就是输入右移一位**。模型在每个时间步的任务是"根据到目前为止看到的所有字符，
预测下一个字符"。

### 3.2 字符级与词级

| | 字符级 | 词级 |
|---|---|---|
| 词表大小 | 几千（中文常用字）| 几万到几十万 |
| 序列长度 | 长 | 短 |
| 输出层参数 | 少 | 多 |

**这一章用字符级**：词表小，输出层便宜，可以先不管分词。

## 4 数据管道

### 4.1 建立字符索引

文本是字符串，网络只吃整数。所以先建两个方向的映射：

```python
chars = sorted(set(text))          # 去重、排序
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
```

**为什么先 `sorted`**：`set` 的迭代顺序在不同运行之间不保证一致。
排序之后同一份文本每次建出的词表相同，实验可复现。

**然后用索引表示整个文本**：

```python
corpus_indices = [char_to_idx[ch] for ch in text]
```

### 4.2 两种采样

一个样本是"长度为 `num_steps` 的一段连续索引"，标签是它右移一位。

**随机采样**：在整条序列上随便挑起点。

```
序列: 0 1 2 3 4 5 6 7 8 9 ...
样本1:      ↑ 从这里取 3 个
样本2:                ↑ 另一个随机位置
```

**特点**：相邻两个小批量在原始序列上**不相邻**，所以它们的隐藏状态没有连续性——
每个批量开始时要**重新初始化隐藏状态**。

**相邻采样**：把整条序列切成 `batch_size` 条，每条内部按 `num_steps` 顺序取。

```
序列切成 2 条：
条0: 0 1 2 3 4 5 6 7 ...
条1: 8 9 10 11 12 13 ...

批量1 = 条0[0:3] 与 条1[0:3]
批量2 = 条0[3:6] 与 条1[3:6]     ← 与批量1 在时间上连续
```

**特点**：相邻批量的片段在原始序列上是接着的，隐藏状态可以跨批量传递。
代价是梯度会沿着批量之间的连接累积，所以每次读新批量前要把隐藏状态 `detach`（B20 讲）。

### 4.3 两种采样的形状

设 `batch_size = 2`、`num_steps = 6`：

| | `X` 形状 | `Y` 形状 |
|---|---|---|
| 随机采样 | `(2, 6)` | `(2, 6)` |
| 相邻采样 | `(2, 6)` | `(2, 6)` |

两者形状相同，**区别在于相邻批量之间是否连续**。

### 4.4 随机采样的实现要点

```python
num_examples = (len(corpus) - 1) // num_steps     # 能切出多少段
epoch_size = num_examples // batch_size          # 每轮多少个批量
```

**为什么减 1**：标签要取到 `pos + num_steps`，最后一个位置没有下一个字符。

**顺序**：

1. 把所有可能的起点洗牌
2. 每 `batch_size` 个起点组成一批
3. 对每个起点 `p`，`X = corpus[p : p+num_steps]`、`Y = corpus[p+1 : p+num_steps+1]`

### 4.5 相邻采样的实现要点

```python
batch_len = len(corpus) // batch_size            # 每条多长
indices = corpus[: batch_size * batch_len].reshape(batch_size, batch_len)
epoch_size = (batch_len - 1) // num_steps
```

**为什么先 `reshape`**：把一维序列切成 `batch_size` 条平行序列，
之后每个批量从每条里切同样位置的 `num_steps` 个。

**取前 `batch_size * batch_len` 个元素**：丢掉尾巴上不够整除的部分。

## 5 实现会踩的坑

| 坑 | 症状 | 原因 |
|---|---|---|
| 标签忘记右移一位 | 模型学成复制输入 | 标签必须是 `corpus[p+1 : p+num_steps+1]`（3.1）|
| 随机采样时忘了重置隐藏状态 | 训练不正常 | 相邻批量不连续，隐藏状态要归零（4.2）|
| `nn.RNN` 忘了 `batch_first=True` | 形状对不上 | 默认是 `(seq_len, batch, d)`（2.4）|
| 字符表用 `set` 但不排序 | 两次运行的词表映射不同 | `sorted(set(text))`（4.1）|
| 相邻采样没处理尾巴 | 索引越界或长度不对 | 只用 `batch_size * batch_len` 个元素（4.5）|
| 拼接写法维度搞反 | 形状报错 | `cat([X,H], dim=1)` 对 `cat([W_xh,W_hh], dim=0)`（2.3）|

## 命令速查

| 命令 | 作用 | 讲义哪一节 |
|---|---|---|
| `sorted(set(text))` | 建字符表 | 4.1 |
| `enumerate(chars)` | 建索引映射 | 4.1 |
| `torch.tensor(list_of_indices)` | 索引列表转张量 | 4.4 |
| `torch.cat([a, b], dim=1)` | 按列拼 | 2.3 |
| `torch.cat([a, b], dim=0)` | 按行拼 | 2.3 |
| `torch.tanh(x)` | RNN 默认激活 | 2.1 |
| `nn.RNN(input_size, hidden_size, batch_first=True)` | 现成的 RNN 层 | 2.4 |
| `torch.zeros(batch, hidden)` | 初始化隐藏状态 | 2.1 |

## 自测题

1. 语言模型的连乘公式是什么？它有两个什么困难？
2. 二元语法下，`P(w1,w2,w3,w4)` 怎么分解？
3. 写出 RNN 隐藏状态与输出的公式，标出每个矩阵的形状。
4. RNN 有多少个参数？序列从 10 步变成 100 步，参数变吗？
5. `X @ W_xh + H @ W_hh` 的拼接等价写法是什么？
6. 字符级语言模型的输入与标签差什么？
7. 随机采样与相邻采样的区别是什么？各自对隐藏状态有什么要求？

## 答案（做完再看）

**1.** $P(w_1,\ldots,w_T) = \prod_t P(w_t \mid w_1,\ldots,w_{t-1})$。
困难：条件长度随 $T$ 增长导致参数爆炸；序列长度不固定。

**2.** $\prod_t P(w_t \mid w_{t-1})$，即 $P(w_1)P(w_2|w_1)P(w_3|w_2)P(w_4|w_3)$。

**3.**
$\boldsymbol{H}_t = \phi(\boldsymbol{X}_t\boldsymbol{W}_{xh} + \boldsymbol{H}_{t-1}\boldsymbol{W}_{hh} + \boldsymbol{b}_h)$，
$\boldsymbol{O}_t = \boldsymbol{H}_t\boldsymbol{W}_{hq} + \boldsymbol{b}_q$。
形状：$\boldsymbol{X}_t(n,d)$、$\boldsymbol{H}(n,h)$、$\boldsymbol{W}_{xh}(d,h)$、
$\boldsymbol{W}_{hh}(h,h)$、$\boldsymbol{W}_{hq}(h,q)$。

**4.** 参数只有五组（$\boldsymbol{W}_{xh}$、$\boldsymbol{W}_{hh}$、$\boldsymbol{b}_h$、
$\boldsymbol{W}_{hq}$、$\boldsymbol{b}_q$），序列变长参数不变——参数在时间维共享。

**5.** `cat([X, H], dim=1) @ cat([W_xh, W_hh], dim=0)`。

**6.** 标签是输入右移一位（`corpus[p+1 : p+num_steps+1]`）。

**7.** 随机采样每次随机挑起点，相邻批量在序列上不连续，所以每批要重置隐藏状态；
相邻采样把序列切成 `batch_size` 条平行序列、按顺序切批，批量之间连续，
隐藏状态可以传递（但要 `detach`）。
