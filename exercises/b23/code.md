# B23 代码题

把你的实现写在 `work/b23/b23.py` 里，函数名与签名必须和下面一致。

验收：`just check b23`。断言全过，并且我读过代码之后，这一题才算过。

---

## 需要的导入

```python
import torch
import torch.nn as nn
```

`valid_lens` 可能为空，签名里写成 `torch.Tensor | None`。

## 先看一段：这一章做什么

四个零件：掩蔽 softmax、加性注意力、缩放点积注意力、Nadaraya-Watson 核回归。

前一章（B22）把输入压成一个定长背景变量；这一章让"解码时每步重新看一遍输入"。
这四个零件合起来就是注意力的全部机制。

---

## T1 掩蔽 softmax

```python
def sequence_mask(X: torch.Tensor, valid_len: torch.Tensor,
                  value: float = 0.0) -> torch.Tensor:
    """把 X 每行第 valid_len[i] 位之后填成 value。X 形状 (n, m)。"""


def masked_softmax(X: torch.Tensor, valid_lens: torch.Tensor | None
                   ) -> torch.Tensor:
    """在最后一维做 softmax，按 valid_lens 掩蔽。

    X 形状 (batch, n, m)。
    valid_lens：None / (batch,) / (batch, n)。
    """
```

| 检查 | 期望 |
|---|---|
| `valid_lens=None` | 等价于普通 softmax |
| `valid_lens` 是一维 | 每个样本一个长度，该样本所有 query 共用 |
| `valid_lens` 是二维 | 每个 query 各一个长度 |
| 掩蔽位置 | 权重为 0 |
| 有效位置 | 权重和为 1 |
| 梯度 | **不含 `nan`** |

**提示**：掩蔽值用 `-1e6`，不要用 `-float("inf")`。

## T2 加性注意力

```python
class AdditiveAttention(nn.Module):
    def __init__(self, key_size: int, query_size: int, num_hiddens: int) -> None:
        """三个无偏置的 Linear：W_q、W_k、w_v。"""

    def forward(self, queries: torch.Tensor, keys: torch.Tensor,
                values: torch.Tensor, valid_lens: torch.Tensor | None = None
                ) -> torch.Tensor:
        """queries: (batch, n_q, q)  keys: (batch, n_k, k)  values: (batch, n_k, v)

        返回 (batch, n_q, v)，并把权重存进 self.attention_weights。
        """
```

**打分函数**：$a(\boldsymbol{q},\boldsymbol{k}) = \boldsymbol{w}_v^\top\tanh(\boldsymbol{W}_q\boldsymbol{q} + \boldsymbol{W}_k\boldsymbol{k})$

| 检查 | 期望 |
|---|---|
| 输出形状 | `(batch, n_q, v)` |
| 权重形状 | `(batch, n_q, n_k)` |
| 每个 query 的权重和 | 1 |
| 与手算 | 一致 |

**广播**：`q` 的 `(b, n_q, h)` 与 `k` 的 `(b, n_k, h)` 要在中间插一维，
扩成 `(b, n_q, 1, h)` 与 `(b, 1, n_k, h)` 才能相加。

## T3 缩放点积注意力

```python
class DotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None: ...

    def forward(self, queries: torch.Tensor, keys: torch.Tensor,
                values: torch.Tensor, valid_lens: torch.Tensor | None = None
                ) -> torch.Tensor:
        """queries: (batch, n_q, d)  keys: (batch, n_k, d)  values: (batch, n_k, v)

        返回 (batch, n_q, v)，并把权重存进 self.attention_weights。
        """
```

| 检查 | 期望 |
|---|---|
| 输出形状 | `(batch, n_q, v)` |
| 与手算 | 一致 |
| `d=64` 时权重的最大值 | **小于 0.99**（不接近 one-hot）|

## T4 Nadaraya-Watson

```python
def nadaraya_watson(queries: torch.Tensor, keys: torch.Tensor,
                    values: torch.Tensor, w: float = 1.0
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """queries: (batch, n_q)  keys: (batch, n_k)  values: (batch, n_k)

    返回 (输出 (batch, n_q), 注意力权重 (batch, n_q, n_k))。
    """
```

**打分**：$-\frac{1}{2}((q - k)w)^2$，过 softmax 后加权平均 `values`。

| 检查 | 期望 |
|---|---|
| 输出形状 | `(batch, n_q)` |
| 与手算 | 一致 |
| `w=10` | 权重集中到与查询重合的位置 |
| `w=0` | 权重完全均匀 |

## T5 改错

下面这段想实现注意力。**三处**问题。

`masked_softmax`：`X` 是 `(batch, n, m)`，`valid_lens` 是 `None` 或
`(batch,)` 或 `(batch, n)`；返回与 `X` 同形状。

`AdditiveAttention`：`queries` 是 `(batch, n_q, q)`，`keys` 是 `(batch, n_k, k)`，
`values` 是 `(batch, n_k, v)`，其中 `q ≠ k` 是允许的；返回 `(batch, n_q, v)`。

`DotProductAttention`：`queries` 是 `(batch, n_q, d)`，`keys` 是 `(batch, n_k, d)`，
两者最后一维相同，都是 `d`；返回 `(batch, n_q, v)`。

```python
def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)
    mask = (torch.arange(shape[-1])[None, :] < valid_lens[:, None])
    X = X.reshape(-1, shape[-1]).masked_fill(~mask, float("-inf"))
    return F.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens):
        super().__init__()
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)

    def forward(self, queries, keys, values):
        q = self.W_q(queries)
        k = self.W_k(keys)
        features = torch.tanh(q + k)
        scores = self.w_v(features).squeeze(-1)
        return torch.bmm(F.softmax(scores, dim=-1), values)


class DotProductAttention(nn.Module):
    def forward(self, queries, keys, values):
        scores = torch.bmm(queries, keys.transpose(1, 2))
        return torch.bmm(F.softmax(scores, dim=-1), values)
```

三处分别指：**掩蔽值**、**加性注意力的广播**、**点积注意力的缩放**。

三段提示：

1. 第一处训练几步梯度就变 `nan`。
2. 第二处要求 `n_q == n_k` 才能跑。
3. 第三处 `d` 大时权重接近 one-hot。