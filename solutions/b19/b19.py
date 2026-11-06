"""B19 参考答案。做完再看。

跑一遍：
    python3 solutions/b19/b19.py
或者拿它验断言：
    python3 tools/check_b19.py solutions/b19/b19.py
"""

import random

import torch


# ---------- T1 字符索引 ----------

def build_vocab(text: str) -> tuple[dict, dict]:
    """返回 (char_to_idx, idx_to_char)，两个映射互逆。"""
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char


def to_indices(text: str, char_to_idx: dict) -> list[int]:
    """把文本转成索引列表。"""
    return [char_to_idx[ch] for ch in text]


# ---------- T2 随机采样 ----------

def data_iter_random(corpus_indices, batch_size: int, num_steps: int):
    """每次随机截取一段长为 num_steps 的序列作为样本，标签右移一位。

    yield (X, Y)，形状都是 (batch_size, num_steps)。
    """
    n = len(corpus_indices)
    num_examples = (n - 1) // num_steps        # 减 1 是因为标签要取到下一位
    epoch_size = num_examples // batch_size
    starts = list(range(num_examples))
    random.shuffle(starts)

    for i in range(epoch_size):
        batch = starts[i * batch_size: (i + 1) * batch_size]
        X = [corpus_indices[j * num_steps: j * num_steps + num_steps] for j in batch]
        Y = [corpus_indices[j * num_steps + 1: j * num_steps + 1 + num_steps] for j in batch]
        yield torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


# ---------- T3 相邻采样 ----------

def data_iter_consecutive(corpus_indices, batch_size: int, num_steps: int):
    """把序列切成 batch_size 条，按顺序切批。相邻批量的片段在时间上相接。

    yield (X, Y)，形状都是 (batch_size, num_steps)。
    """
    data = torch.tensor(corpus_indices, dtype=torch.long)
    batch_len = len(data) // batch_size
    indices = data[: batch_size * batch_len].reshape(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps

    for i in range(epoch_size):
        start = i * num_steps
        X = indices[:, start: start + num_steps]
        Y = indices[:, start + 1: start + 1 + num_steps]
        yield X, Y


# ---------- T4 手写 RNN 前向 ----------

def rnn_step(X, H, W_xh, W_hh, b_h):
    """单步：H_new = tanh(X @ W_xh + H @ W_hh + b_h)。

    X: (batch, d)  H: (batch, h)  W_xh: (d, h)  W_hh: (h, h)  b_h: (h,)
    返回新的 H，形状 (batch, h)。
    """
    return torch.tanh(X @ W_xh + H @ W_hh + b_h)


def rnn_forward(Xs, H0, W_xh, W_hh, b_h):
    """多步：Xs 形状 (seq_len, batch, d)，返回 (seq_len, batch, h) 的所有隐藏状态。"""
    hs = []
    H = H0
    for x in Xs:
        H = rnn_step(x, H, W_xh, W_hh, b_h)
        hs.append(H)
    return torch.stack(hs)


"""T5 改错

第 1 处：`num_examples = len(corpus) // num_steps`
    漏了减 1。标签要取 `pos+1` 到 `pos+num_steps+1`，最后一段没有下一个字符，
    取标签时会越界（或静默截短）。
    改法：`(len(corpus) - 1) // num_steps`。

第 2 处：`Y` 的切片与 `X` 一样
    标签没有右移一位，模型会学成"原样复制输入"。这是最隐蔽的一处——不报错，
    损失能降，但模型学到的是恒等映射，生成不出任何东西。
    改法：`Y = corpus[pos+1 : pos+1+num_steps]`。

第 3 处：起点没有洗牌
    每轮遍历顺序完全相同，等于每轮喂同样的批次顺序。不是致命错误，
    但训练时批次的随机性没了。
    改法：`random.shuffle(starts)`。
"""


if __name__ == "__main__":
    text = "想要有直升机，想要和你飞到宇宙去"
    c2i, i2c = build_vocab(text)
    print("T1 词表大小:", len(c2i))
    idx = to_indices(text, c2i)
    print("T1 索引前 10:", idx[:10])
    print("T1 还原:", "".join(i2c[i] for i in idx[:10]))

    seq = list(range(30))
    for X, Y in data_iter_random(seq, 2, 6):
        print("T2 随机:", X[0].tolist(), "->", Y[0].tolist())
    for X, Y in data_iter_consecutive(seq, 2, 6):
        print("T3 相邻:", X[0].tolist(), "->", Y[0].tolist())

    torch.manual_seed(0)
    d, h, n = 3, 4, 2
    X = torch.randn(n, d)
    H = torch.zeros(n, h)
    W_xh = torch.randn(d, h)
    W_hh = torch.randn(h, h)
    b_h = torch.zeros(h)
    print("T4 单步:", tuple(rnn_step(X, H, W_xh, W_hh, b_h).shape))
    Xs = torch.randn(5, n, d)
    print("T4 多步:", tuple(rnn_forward(Xs, H, W_xh, W_hh, b_h).shape))
