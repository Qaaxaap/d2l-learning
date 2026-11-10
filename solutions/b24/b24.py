"""B24 参考答案。做完再看。

跑一遍：
    python3 solutions/b24/b24.py
或者拿它验断言：
    python3 tools/check_b24.py solutions/b24/b24.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- T1 形状变换 ----------

def transpose_qkv(X: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(batch, n, p) -> (batch * num_heads, n, p / num_heads)。"""
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(batch * num_heads, n, p / num_heads) -> (batch, n, p)。"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


# ---------- T2 多头注意力 ----------

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size: int, query_size: int, value_size: int,
                 num_hiddens: int, num_heads: int, dropout: float = 0.0,
                 bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        """三个张量形状 (batch, n, d)。返回 (batch, n_q, num_hiddens)。"""
        q = transpose_qkv(self.W_q(queries), self.num_heads)
        k = transpose_qkv(self.W_k(keys), self.num_heads)
        v = transpose_qkv(self.W_v(values), self.num_heads)

        mask = None
        if valid_lens is not None:
            # 每个头各一份有效长度
            lens = torch.repeat_interleave(valid_lens, self.num_heads, dim=0)
            n_k = k.shape[1]
            # (batch*heads, n_k)，True 表示允许
            mask = torch.arange(n_k, device=k.device)[None, :] < lens[:, None]
            # (b*h, n_k) -> (b*h, 1, n_k)，让查询那一维广播
            mask = mask[:, None, :]

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = transpose_output(out, self.num_heads)
        return self.W_o(out)


# ---------- T3 位置编码 ----------

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens: int, dropout: float = 0.0,
                 max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        P = torch.zeros(1, max_len, num_hiddens)
        pos = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        freq = torch.pow(10000.0,
                         torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        X = pos / freq
        P[:, :, 0::2] = torch.sin(X)
        P[:, :, 1::2] = torch.cos(X)
        self.register_buffer("P", P)

    def forward(self, X):
        """X 形状 (batch, n, num_hiddens)。返回同形状。"""
        n = X.shape[1]
        return self.dropout(X + self.P[:, :n, :].to(X.device))


# ---------- T4 编码器块 ----------

class AddNorm(nn.Module):
    def __init__(self, normalized_shape: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        """返回 LayerNorm(dropout(Y) + X)。"""
        return self.ln(self.dropout(Y) + X)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input: int, ffn_num_hiddens: int,
                 ffn_num_outputs: int):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class EncoderBlock(nn.Module):
    def __init__(self, key_size: int, query_size: int, value_size: int,
                 num_hiddens: int, norm_shape: int, ffn_num_input: int,
                 ffn_num_hiddens: int, num_heads: int, dropout: float,
                 use_bias: bool = False):
        super().__init__()
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            bias=use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens=None):
        """X 形状 (batch, n, num_hiddens)。返回同形状。"""
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


"""T5 改错

第 1 处：`transpose_qkv` 里直接 `reshape(-1, n, p // h)`
    那样是按位置切分，不是按头切分。必须先 reshape 成 `(batch, n, h, p/h)`
    再 transpose，最后拍平。
    改法：三步走（讲义 2.1）。

第 2 处：`PositionalEncoding` 的 P 用 `nn.Parameter` 包了
    位置编码由公式算出来，不该被训练。用 `register_buffer` 存，不进优化器。
    改法：`self.register_buffer("P", P)`。

第 3 处：`EncderBlock.forward` 里先加法再 dropout 的顺序反了
    AddNorm 是 `LayerNorm(dropout(sublayer(X)) + X)`——dropout 作用于子层输出，
    不是作用于加上残差之后。顺序反了会把残差路径也随机丢弃。
    改法：先对子层输出 dropout，再加 X，最后归一化。
"""


if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(2, 5, 8)
    print("T1 transpose_qkv:", tuple(transpose_qkv(X, 4).shape))
    print("T1 transpose_output:", tuple(transpose_output(transpose_qkv(X, 4), 4).shape))

    mha = MultiHeadAttention(8, 8, 8, 16, 4, dropout=0.0)
    print("T2 MHA:", tuple(mha(X, X, X).shape))

    pe = PositionalEncoding(16, dropout=0.0)
    print("T3 PE:", tuple(pe(torch.randn(2, 5, 16)).shape))

    blk = EncoderBlock(key_size=16, query_size=16, value_size=16,
                       num_hiddens=16, norm_shape=16, ffn_num_input=16,
                       ffn_num_hiddens=32, num_heads=4, dropout=0.0)
    print("T4 EncoderBlock:", tuple(blk(torch.randn(2, 5, 16)).shape))
