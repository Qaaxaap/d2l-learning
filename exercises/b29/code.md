# B29 代码题

把你的实现写在 `work/b29/b29.py` 里，函数名与签名必须和下面一致。

验收：`just check b29`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章做什么

写自然语言应用的五个基础组件：截断与填充、一维卷积的输出长度、时序最大池化、
文本对的特征拼接、分类头。

**全部是纯数值计算**。不训练模型，序列与向量都是人造的小张量。

---

## T1 截断与填充

```python
def truncate_pad(line: list[int], num_steps: int,
                 padding_token: int) -> list[int]:
    """把一条索引序列变成固定的 num_steps 长度。

    超长时只留前 num_steps 个，不足时末尾补 padding_token。
    """
```

两条规则：
- `len(line) > num_steps`：留前 `num_steps` 个。
- 否则：末尾补 `padding_token` 到 `num_steps` 个。

| 检查 | 期望 | 类型 |
|---|---|---|
| `([1,2,3,4,5], 3, 0)` | `[1,2,3]` | 精确 |
| `([1,2], 5, 0)` | `[1,2,0,0,0]` | 精确 |
| `([1,2,3], 3, 0)` | `[1,2,3]` | 精确 |
| 截断保留的顺序 | 前三个，不是末尾填充 | 精确 |

## T2 一维卷积的输出长度

```python
def conv1d_output_len(n_in: int, k: int, stride: int = 1,
                      padding: int = 0) -> int:
    """一维卷积的输出长度取整。"""
```

公式：

$$L_{out} = \left\lfloor\frac{L_{in} + 2p - k}{s}\right\rfloor + 1$$

| 检查 | 期望 | 类型 |
|---|---|---|
| `(5, 2)` | 4 | 精确 |
| `(16, 3, 2, 1)` | 8 | 精确 |
| `(100, 5, 1, 2)` | 100 | 精确 |

## T3 时序最大池化

```python
def time_max_pool(x: torch.Tensor) -> torch.Tensor:
    """在时间维上取最大值。

    x 形状 (batch, channels, length)，返回 (batch, channels)。
    """
```

一行能写完。注意用 `.values` 取张量，`torch.max` 返回的是 `(值, 下标)` 的元组。

| 检查 | 期望 | 类型 |
|---|---|---|
| `[[[1,5,3],[2,0.5,8]]]` | `[[5, 8]]` | 精确 |
| 形状 | `(batch, channels)` | 精确 |
| 梯度分布 | 每个通道只有一个位置有梯度 | 精确 |

## T4 文本对的特征

```python
def pair_features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """把两句的表示拼成分类特征。

    a、b 形状都是 (n, d)。返回 (n, 4*d)，
    依次是 a、b、|a-b|、a*b 四段拼接。
    """
```

四段（讲义 6.2）：

$$\boldsymbol{f} = [\boldsymbol{a};\ \boldsymbol{b};\ |\boldsymbol{a}-\boldsymbol{b}|;\
\boldsymbol{a}\odot\boldsymbol{b}]$$

**做差要取绝对值**。有符号的差会把「差 4」与「差 −4」当成相反的东西，
但两者都表示「这两个位置不一致」。

| 检查 | 期望 | 类型 |
|---|---|---|
| `a=[[1,2]]`、`b=[[3,1]]` | `[[1,2,3,1,2,1,3,2]]` | 精确 |
| `a=[[1,0]]`、`b=[[5,0]]` | 第三段是 `[4,0]`，不是 `[-4,0]` | 精确 |
| 形状 | `(n, 4d)` | 精确 |

## T5 分类头

```python
def build_classifier(embed: torch.Tensor, num_classes: int) -> torch.nn.Module:
    """从最后一层隐藏状态接一个线性分类头。

    embed 是 BERT 的 [CLS] 表示，形状 (batch, hidden)。
    """
```

返回一个 `nn.Linear`，输入维取 `embed.shape[1]`，输出维是 `num_classes`。

| 检查 | 期望 | 类型 |
|---|---|---|
| 返回类型 | `nn.Linear` | 精确 |
| `in_features` 与 `out_features` | 16 与 5 | 精确 |
| 前向输出形状 | `(3, 5)` | 精确 |

## T6 改错

下面这段想做一个用双向 RNN 的情感分类器。**三处**问题。

输入 `inputs` 是 `(batch, num_steps)` 的整数张量（已填充到同一长度）。
`vocab_size` 是词表大小，`num_classes` 是情感类别数。
`nn.GRU` 用了 `batch_first=True` 与 `bidirectional=True`，所以 `h` 的形状是
`(num_layers * 2, batch, num_hiddens)`。

```python
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, 1,
                          bidirectional=True, batch_first=True)
        self.fc = nn.Linear(num_hiddens, num_classes)

    def forward(self, inputs):
        x = self.embedding(inputs)
        out, h = self.rnn(x)
        enc = h[-1]
        logits = self.fc(enc)
        return torch.softmax(logits, dim=1)
```

三处分别指：**嵌入层的填充符号**、**双向 RNN 取哪个状态**、**softmax 与损失函数**。

三段提示：

1. 第一处让填充符号也被训练成有意义的向量。
2. 第二处只取到一个方向。
3. 第三处与 `CrossEntropyLoss` 重复了。