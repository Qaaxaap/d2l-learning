"""B23 参考答案。做完再看。

跑一遍：
    python3 solutions/b23/b23.py
或者拿它验断言：
    python3 tools/check_b23.py solutions/b23/b23.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- T1 掩蔽 softmax ----------

def sequence_mask(X: torch.Tensor, valid_len: torch.Tensor,
                  value: float = 0.0) -> torch.Tensor:
    """把 X 每行第 valid_len[i] 位之后填成 value。X 形状 (n, m)。"""
    maxlen = X.size(1)
    mask = torch.arange(maxlen, device=X.device)[None, :] < valid_len[:, None]
    return X.masked_fill(~mask, value)


def masked_softmax(X: torch.Tensor, valid_lens):
    """在最后一维做 softmax，按 valid_lens 掩蔽。

    X 形状 (batch, n, m)。
    valid_lens：None / (batch,) / (batch, n)。
    """
    if valid_lens is None:
        return F.softmax(X, dim=-1)

    shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)

    X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return F.softmax(X.reshape(shape), dim=-1)


# ---------- T2 加性注意力 ----------

class AdditiveAttention(nn.Module):
    def __init__(self, key_size: int, query_size: int, num_hiddens: int):
        super().__init__()
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.attention_weights = None

    def forward(self, queries, keys, values, valid_lens=None):
        """queries: (batch, n_q, q)  keys: (batch, n_k, k)  values: (batch, n_k, v)

        返回 (batch, n_q, v)。
        """
        q = self.W_q(queries)                      # (b, n_q, h)
        k = self.W_k(keys)                         # (b, n_k, h)
        features = torch.tanh(q.unsqueeze(2) + k.unsqueeze(1))   # (b, n_q, n_k, h)
        scores = self.w_v(features).squeeze(-1)    # (b, n_q, n_k)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.attention_weights, values)


# ---------- T3 缩放点积注意力 ----------

class DotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, queries, keys, values, valid_lens=None):
        """queries: (batch, n_q, d)  keys: (batch, n_k, d)  values: (batch, n_k, v)"""
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


# ---------- T4 Nadaraya-Watson ----------

def nadaraya_watson(queries: torch.Tensor, keys: torch.Tensor,
                    values: torch.Tensor, w: float = 1.0) -> torch.Tensor:
    """非参数（或固定 w）的核回归。

    queries: (batch, n_q)  keys: (batch, n_k)  values: (batch, n_k)
    返回 (batch, n_q)。
    """
    dist = (queries.unsqueeze(2) - keys.unsqueeze(1)) ** 2      # (b, n_q, n_k)
    weights = F.softmax(-0.5 * (w ** 2) * dist, dim=-1)
    return (weights * values.unsqueeze(1)).sum(dim=-1), weights


"""T5 改错

第 1 处：掩蔽用了 `-float("inf")`
    `-inf` 参与 `exp` 得到 0 没问题，但反向传播时 `exp(-inf)` 的梯度是 `0 * inf`，
    得到 nan，训练几步就崩。
    改法：用有限的 `-1e6`。

第 2 处：加性注意力没做广播
    `q` 是 (b, n_q, h)、`k` 是 (b, n_k, h)。直接相加会要求 n_q == n_k。
    要分别扩展出一个维度：(b, n_q, 1, h) + (b, 1, n_k, h) → (b, n_q, n_k, h)。
    改法：`q.unsqueeze(2) + k.unsqueeze(1)`。

第 3 处：点积注意力忘了除以 sqrt(d)
    d 大时点积的方差是 d，softmax 过尖（接近 one-hot），梯度接近 0。
    改法：`scores / math.sqrt(d)`。
"""


if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.rand(2, 2, 4)
    print("T1 无掩蔽:", masked_softmax(X, None).sum(-1).tolist())
    print("T1 掩蔽 [2,3]:", masked_softmax(X, torch.tensor([2, 3])).sum(-1).tolist())

    q = torch.randn(2, 3, 5)
    k = torch.randn(2, 4, 6)
    v = torch.randn(2, 4, 7)
    attn = AdditiveAttention(key_size=6, query_size=5, num_hiddens=8)
    print("T2 加性:", tuple(attn(q, k, v).shape), "权重和:", attn.attention_weights.sum(-1).tolist())

    dpa = DotProductAttention()
    print("T3 点积:", tuple(dpa(q[..., :5], k[..., :5], v).shape))

    qs = torch.tensor([[0.0, 1.0, 2.0]])
    ks = torch.tensor([[0.0, 1.0, 2.0]])
    vs = torch.tensor([[0.0, 10.0, 20.0]])
    out_w1, w1 = nadaraya_watson(qs, ks, vs, w=1.0)
    out_w10, w10 = nadaraya_watson(qs, ks, vs, w=10.0)
    print("T4 w=1:", out_w1.tolist(), "权重:", [round(x, 4) for x in w1[0, 1].tolist()])
    print("T4 w=10:", out_w10.tolist(), "权重:", [round(x, 4) for x in w10[0, 1].tolist()])
