# B19 代码题

把你的实现写在 `work/b19/b19.py` 里，函数名与签名必须和下面一致。

验收：`just check b19`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章做什么

两件事：**把文本变成能训练的批次**（T1–T3）、**手写 RNN 的单步与多步前向**（T4）。

**这一章不训练**。数据管道与 RNN 前向分开验证，训练在 B20。

验收分三类：**精确**（形状、右移一位、与 `nn.RNN` 一致）、
**结构**（词表互逆）、**性质**（随机采样的随机性、相邻采样的连续性）。

---

## T1 字符索引

```python
def build_vocab(text: str) -> tuple[dict, dict]:
    """返回 (char_to_idx, idx_to_char)，两个映射互逆。"""


def to_indices(text: str, char_to_idx: dict) -> list[int]:
    """把文本转成索引列表。"""
```

| 检查 | 期望 |
|---|---|
| 词表大小 | 不同字符的个数 |
| 两个映射 | 互逆 |
| 索引转回字符 | 得到原文 |
| 两次建词表 | 映射完全相同 |

**提示**：用 `sorted(set(text))`，不要只写 `set(text)`。

## T2 随机采样

```python
def data_iter_random(corpus_indices, batch_size: int, num_steps: int):
    """每次随机截取一段长为 num_steps 的序列作为样本，标签右移一位。

    yield (X, Y)，形状都是 (batch_size, num_steps)。
    """
```

`corpus_indices` 是一个 Python `int` 列表。

| 检查 | 期望 |
|---|---|
| `X`、`Y` 形状 | `(batch_size, num_steps)` |
| `Y` 与 `X` 的关系 | 右移一位 |
| 批量个数 | `((len - 1) // num_steps) // batch_size` |
| 不同随机种子 | 起点顺序不同 |

**为什么 `len - 1`**：标签要取到 `pos + num_steps`，最后一个位置没有下一个字符。

## T3 相邻采样

```python
def data_iter_consecutive(corpus_indices, batch_size: int, num_steps: int):
    """把序列切成 batch_size 条，按顺序切批。

    yield (X, Y)，形状都是 (batch_size, num_steps)。
    """
```

| 检查 | 期望 |
|---|---|
| `X`、`Y` 形状 | `(batch_size, num_steps)` |
| `Y` 与 `X` 的关系 | 右移一位 |
| 第 1 条序列的起点 | `len // batch_size` |
| 相邻批量的片段 | 在时间上相接 |

## T4 手写 RNN 前向

```python
def rnn_step(X, H, W_xh, W_hh, b_h):
    """单步：H_new = tanh(X @ W_xh + H @ W_hh + b_h)。

    X: (batch, d)  H: (batch, h)  W_xh: (d, h)  W_hh: (h, h)  b_h: (h,)
    """


def rnn_forward(Xs, H0, W_xh, W_hh, b_h):
    """多步：Xs 形状 (seq_len, batch, d)，返回 (seq_len, batch, h)。"""
```

| 检查 | 期望 |
|---|---|
| `rnn_step` 与手算 `tanh(X@W_xh + H@W_hh + b_h)` | 一致 |
| 换一个 `H` | 输出应当不同（证明 `H @ W_hh` 生效）|
| `rnn_forward` 形状 | `(seq_len, batch, h)` |
| `rnn_forward` 与逐步循环 | 一致 |
| 与 `nn.RNN`（`nonlinearity="tanh"`）| 一致 |

**`nn.RNN` 的权重布局**：`weight_ih_l0` 形状是 `(h, d)`，与我们的 `W_xh`（`(d, h)`）
是转置关系。断言做对比时会自己转置，你只需保证自己的实现用的是 `(d, h)`。

## T5 改错

下面这段想实现随机采样。**三处**问题。

```python
def data_iter_random(corpus, batch_size, num_steps):
    num_examples = len(corpus) // num_steps
    epoch_size = num_examples // batch_size
    starts = list(range(num_examples))

    for i in range(epoch_size):
        batch = starts[i * batch_size:(i + 1) * batch_size]
        X = [corpus[j * num_steps: j * num_steps + num_steps] for j in batch]
        Y = [corpus[j * num_steps: j * num_steps + num_steps] for j in batch]
        yield torch.tensor(X), torch.tensor(Y)
```

三处分别指：**样本个数的算法**、**标签的切片**、**起点的顺序**。

三段提示：第一处不报错，但最后一段的标签会越界；第二处不报错，损失能降，
但模型学成原样复制；第三处不报错，只是每轮的批次顺序固定不变。
