# B19 代码题

把你的实现写在 `work/b19/b19.py` 里，函数名与签名必须和下面一致。

验收：`just check b19`。断言全过，并且我读过代码之后，这一题才算过。

---

## 需要的导入

题面的签名用到 `Generator` 与 `Any`，从 `typing` 导入：

```python
from typing import Any, Generator
```

两个采样函数是生成器（含 `yield`），返回类型写
`Generator[tuple[torch.Tensor, torch.Tensor], Any, None]`。
三个参数依次是产出类型、`send()` 接受类型、`return` 类型。

另外需要 `torch`。向量与索引用 `torch.Tensor`，索引列表用 `list[int]`。

## 先看一段：这一章做什么

两件事：**把文本变成能训练的批次**（T1–T3）、**手写 RNN 的单步与多步前向**（T4）。

**这一章不训练**。数据管道与 RNN 前向分开验证，训练在 B20。

验收分三类：**精确**（形状、右移一位、与 `nn.RNN` 一致）、
**结构**（词表互逆）、**性质**（随机采样的随机性、相邻采样的连续性）。

---

## T1 字符索引

```python
def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """返回 (char_to_idx, idx_to_char)，两个映射互逆。

    char_to_idx 的键是字符、值是下标；idx_to_char 反过来。
    """


def to_indices(text: str, char_to_idx: dict[str, int]) -> list[int]:
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
def data_iter_random(corpus_indices: list[int], batch_size: int,
                     num_steps: int) -> Generator[tuple[torch.Tensor, torch.Tensor], Any, None]:
    """每次随机截取一段长为 num_steps 的序列作为样本，标签右移一位。

    生成器，每次产出 (X, Y)，形状都是 (batch_size, num_steps)。
    """
```

`corpus_indices` 是一个 Python `int` 列表。

| 检查 | 期望 |
|---|---|
| `X`、`Y` 形状 | `(batch_size, num_steps)` |
| `Y` 与 `X` 的关系 | 右移一位 |
| 批量个数 | `((len - 1) // num_steps) // batch_size` |
| 不同随机种子 | 起点顺序不同 |
| **同一行内部** | 元素在**原始序列上连续** |
| **同一批内的不同行** | 起点互不相邻（先洗牌再切） |
| **相邻两批之间** | 起点无关（各自独立洗牌） |

**为什么 `len - 1`**：标签要取到 `pos + num_steps`，最后一个位置没有下一个字符。

**三种「相邻」要分清**（讲义 5.3）。一个批量是 `(batch_size, num_steps)`：

| 关系 | 指什么 | 这个任务里是否连续 |
|---|---|---|
| 行内 | 一行里相邻的时间步 | **连续**，这是 RNN 处理的时间序列 |
| 行间 | 同一批里不同的行 | 不连续，是独立样本 |
| 批间 | 这一批与下一批的同一行 | 不连续，所以训练时每批要重置隐藏状态 |

只抽一行出来看，`X[i]` 是 `corpus[pos_i : pos_i + num_steps]`，
`Y[i]` 是 `corpus[pos_i + 1 : pos_i + num_steps + 1]`。这一行内前后接着。

行与行之间没有时间关系。第 1 行的起点与第 0 行的起点可能跨越整个语料。

## T3 相邻采样

```python
def data_iter_consecutive(corpus_indices: list[int], batch_size: int,
                          num_steps: int
                          ) -> Generator[tuple[torch.Tensor, torch.Tensor], Any, None]:
    """把序列切成 batch_size 条，按顺序切批。

    生成器，每次产出 (X, Y)，形状都是 (batch_size, num_steps)。
    """
```

| 检查 | 期望 |
|---|---|
| `X`、`Y` 形状 | `(batch_size, num_steps)` |
| `Y` 与 `X` 的关系 | 右移一位 |
| 第 1 条序列的起点 | `len // batch_size` |
| **同一行内部** | 连续 |
| **同一批内的不同行** | 不连续（来自两条不同的平行序列） |
| **相邻两批之间** | **连续**（同一行的片段接着上一批） |

**三种「相邻」的关系与 T2 只有一处不同**（讲义 5.6）：

| 关系 | 随机采样 | 相邻采样 |
|---|---|---|
| 行内 | 连续 | 连续 |
| 行间 | 不连续 | 不连续 |
| 批间 | 不连续 | **连续** |

批间连续是唯一的差别，也是训练时「隐藏状态可以跨批传」的依据。

**举例**：`corpus = [0..15]`、`batch_size = 2`、`num_steps = 3`。

- 第 0 条平行序列是 `[0..7]`，第 1 条是 `[8..15]`。
- 批量 1 = `[[0,1,2], [8,9,10]]`。
- 批量 2 = `[[3,4,5], [11,12,13]]`。
- 批量 1 的第 0 行与批量 2 的第 0 行相接（`2` 之后是 `3`）。
- 批量 1 的两行之间没有关系（`[0,1,2]` 与 `[8,9,10]` 隔着 8 个位置）。

## T4 手写 RNN 前向

```python
def rnn_step(X: torch.Tensor, H: torch.Tensor, W_xh: torch.Tensor,
             W_hh: torch.Tensor, b_h: torch.Tensor) -> torch.Tensor:
    """单步。返回 H_new，形状 (batch, h)。

    H_new = tanh(X @ W_xh + H @ W_hh + b_h)。

    参数形状：X 是 (batch, d)、H 是 (batch, h)、W_xh 是 (d, h)、
    W_hh 是 (h, h)、b_h 是 (h,)。
    """


def rnn_forward(Xs: torch.Tensor, H0: torch.Tensor, W_xh: torch.Tensor,
                W_hh: torch.Tensor, b_h: torch.Tensor) -> torch.Tensor:
    """多步前向。返回 H_all，形状 (seq_len, batch, h)。

    参数形状：Xs 是 (seq_len, batch, d)，H0 是 (batch, h)。

    返回值定义（逐个下标写死，没有别的解读）：

        H_all[0]   = tanh(Xs[0] @ W_xh + H0 @ W_hh + b_h)
        H_all[1]   = tanh(Xs[1] @ W_xh + H_all[0] @ W_hh + b_h)
        H_all[t]   = tanh(Xs[t] @ W_xh + H_all[t-1] @ W_hh + b_h)
        ...
        H_all[T-1] = tanh(Xs[T-1] @ W_xh + H_all[T-2] @ W_hh + b_h)

    三点强调：

    1. **返回全部时间步**，不是只返回最后一步。`H_all[-1]` 才是最后的状态。
    2. **`H0` 只用在第 0 步**。它不参与输出，也不出现在返回值里。
    3. **每个 `H_all[t]` 是「看过 `Xs[t]` 之后」的状态**，不是看之前的。
    """
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

**这个函数与 B20 的同名函数不是一回事。** B20 也会有一个 `rnn_forward`，
但它的输入是 one-hot 列表加参数元组，返回 `(outputs, state)` 两个值。
本章这个只做状态序列，返回一个张量。写 B20 时不要照搬。

## T5 改错

下面这段想实现随机采样。**三处**问题。

```python
def data_iter_random(corpus: list[int], batch_size: int, num_steps: int):
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
