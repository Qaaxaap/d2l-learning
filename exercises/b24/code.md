# B24 代码题

把你的实现写在 `work/b24/b24.py` 里，函数名与签名必须和下面一致。

验收：`just check b24`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章做什么

把 B23 的注意力变成能搭深网络的组件：多头、位置编码、残差加层规范化、编码器块。

这一章**不训练**。验证的是形状变换的正确性、多头是否真的分了头、位置编码的数值。

---

## T1 形状变换

```python
def transpose_qkv(X: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(batch, n, p) -> (batch * num_heads, n, p / num_heads)。"""


def transpose_output(X: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(batch * num_heads, n, p / num_heads) -> (batch, n, p)。"""
```

**三步**（讲义 2.1）：reshape 拆头、transpose 把头挪到批量旁、reshape 拍平。
逆变换反过来做。

| 检查 | 期望 |
|---|---|
| `transpose_qkv` 形状 | `(batch * heads, n, p / heads)` |
| 第 i 个头的内容 | 等于 `X[:, :, i*(p/h) : (i+1)*(p/h)]` |
| 两个函数互逆 | 转一圈回到原值 |

**不能直接 `reshape(-1, n, p // h)`**：那样是按位置切分，不是按头切分。

## T2 多头注意力

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size: int, query_size: int, value_size: int,
                 num_hiddens: int, num_heads: int, dropout: float = 0.0,
                 bias: bool = False):
        """四个无偏置的 nn.Linear：W_q、W_k、W_v、W_o。"""

    def forward(self, queries, keys, values, valid_lens=None):
        """三个张量形状 (batch, n, d)。返回 (batch, n_q, num_hiddens)。

        valid_lens：形状 (batch,) 的整数张量，每个样本的键值有效长度。
        """
```

**内部怎么算注意力**：用 torch 自带的

```python
F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=...)
```

它的用法见讲义 1.4 节。`mask` 用 bool，`True` 表示允许。

| 检查 | 期望 |
|---|---|
| 输出形状 | `(batch, n_q, num_hiddens)` |
| `num_heads=1` 时 | 等价于单头缩放点积注意力 |
| 各头的结果 | 互不相同（说明真的分了头）|
| `valid_lens` | 超出长度的键不参与注意力 |

## T3 位置编码

```python
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens: int, dropout: float = 0.0,
                 max_len: int = 1000):
        """P 用公式算出来，注册成 buffer，不训练。"""

    def forward(self, X):
        """X 形状 (batch, n, num_hiddens)。返回同形状。"""
```

公式（讲义 4.2）：

$$p_{i, 2j} = \sin\left(\frac{i}{10000^{2j/d}}\right), \qquad
p_{i, 2j+1} = \cos\left(\frac{i}{10000^{2j/d}}\right)$$

| 检查 | 期望 |
|---|---|
| `P` 形状 | `(1, max_len, num_hiddens)` |
| `P` 的具体值 | 与公式对上 |
| 可训练参数个数 | **0** |
| 前向 | `X + P[:, :n, :]` |

## T4 编码器块

```python
class AddNorm(nn.Module):
    def __init__(self, normalized_shape: int, dropout: float = 0.0):
        """LayerNorm + Dropout。"""

    def forward(self, X, Y):
        """返回 LayerNorm(dropout(Y) + X)。"""


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input: int, ffn_num_hiddens: int,
                 ffn_num_outputs: int):
        """两层全连接，中间 ReLU。"""

    def forward(self, X):
        """X 形状 (batch, n, ffn_num_input)。"""


class EncoderBlock(nn.Module):
    def __init__(self, key_size: int, query_size: int, value_size: int,
                 num_hiddens: int, norm_shape: int, ffn_num_input: int,
                 ffn_num_hiddens: int, num_heads: int, dropout: float,
                 use_bias: bool = False):
        """多头自注意力 → AddNorm → 前馈 → AddNorm。"""

    def forward(self, X, valid_lens=None):
        """X 形状 (batch, n, num_hiddens)。返回同形状。"""
```

**`AddNorm` 的顺序**：`LayerNorm(dropout(Y) + X)`。dropout 作用于**子层输出**，
不是作用于加上残差之后。

| 检查 | 期望 |
|---|---|
| `AddNorm` 输出 | `LayerNorm(dropout(Y) + X)` |
| 编码器块输出形状 | `(batch, n, num_hiddens)` |
| 编码器块里的 AddNorm 个数 | 2 |

## T5 改错

下面这段想实现多头注意力与位置编码。**三处**问题。

```python
def transpose_qkv(X, num_heads):
    # X: (batch, n, p)
    return X.reshape(-1, X.shape[1], X.shape[2] // num_heads)

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        P = torch.zeros(1, max_len, num_hiddens)
        X = torch.arange(max_len).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2) / num_hiddens)
        P[:, :, 0::2] = torch.sin(X)
        P[:, :, 1::2] = torch.cos(X)
        self.P = nn.Parameter(P)

    def forward(self, X):
        return self.dropout(X + self.P[:, :X.shape[1], :])

class EncoderBlock(nn.Module):
    def forward(self, X, valid_lens=None):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class AddNorm(nn.Module):
    def forward(self, X, Y):
        return self.ln(self.dropout(Y + X))
```

三处分别指：**`transpose_qkv` 的切法**、**`P` 的存放方式**、**`AddNorm` 里 dropout 的位置**。

三段提示：第一处会把每个位置的前几段通道当成一个头；第二处会让位置编码被训练；
第三处会把残差路径也随机丢掉。
