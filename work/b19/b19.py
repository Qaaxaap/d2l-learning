"""
def data_iter_random(corpus: list[int], batch_size: int, num_steps: int):
    num_examples = len(corpus) // num_steps 没有-1。
    epoch_size = num_examples // batch_size
    starts = list(range(num_examples)) 没有随机打乱

    for i in range(epoch_size):
        batch = starts[i * batch_size:(i + 1) * batch_size]
        X = [corpus[j * num_steps: j * num_steps + num_steps] for j in batch]
        Y = [corpus[j * num_steps: j * num_steps + num_steps] for j in batch] XY完全一样，Y要+1
        yield torch.tensor(X), torch.tensor(Y)
"""
from typing import Any, Generator
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def build_vocab(text: str) -> tuple[dict, dict]:
    """返回 (char_to_idx, idx_to_char)，两个映射互逆。"""
    chars = sorted(set(text))
    return (
        {ch: i for i, ch in enumerate(chars)},
        {i: ch for i, ch in enumerate(chars)},
    )


def to_indices(text: str, char_to_idx: dict) -> list[int]:
    """把文本转成索引列表。"""
    ret = []
    for i in text:
        ret.append(char_to_idx[i])
    return ret


def data_iter_random(
    corpus_indices, batch_size: int, num_steps: int
) -> Generator[tuple[Tensor, Tensor], Any, None]:
    """每次随机截取一段长为 num_steps 的序列作为样本，标签右移一位。

    yield (X, Y)，形状都是 (batch_size, num_steps)。
    """
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos: int) -> list[int]:
        """从 pos 开始截取长度为 num_steps 的连续片段。"""
        return corpus_indices[pos : pos + num_steps]

    for i in range(0, epoch_size * batch_size, batch_size):
        batch_indices = example_indices[i : i + batch_size]

        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]

        yield (torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long))


def data_iter_consecutive(
    corpus_indices: list[int], batch_size: int, num_steps: int
) -> Generator[tuple[Tensor, Tensor], Any, None]:
    """把序列切成 batch_size 条，按顺序切批。

    生成器，每次产出 (X, Y)，形状都是 (batch_size, num_steps)。
    """
    data = torch.tensor(corpus_indices, dtype=torch.long)
    batch_len = len(data) // batch_size
    indices = data[: batch_size * batch_len].reshape(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        start = i * num_steps
        X = indices[:, start : start + num_steps]
        Y = indices[:, start + 1 : start + num_steps + 1]
        yield X, Y


def rnn_step(
    X: torch.Tensor,
    H: torch.Tensor,
    W_xh: torch.Tensor,
    W_hh: torch.Tensor,
    b_h: torch.Tensor,
) -> torch.Tensor:
    """单步：H_new = tanh(X @ W_xh + H @ W_hh + b_h)。

    X: (batch, d)  H: (batch, h)  W_xh: (d, h)  W_hh: (h, h)  b_h: (h,)
    """
    return torch.tanh(X @ W_xh + H @ W_hh + b_h)


def rnn_forward(
    Xs: torch.Tensor,
    H0: torch.Tensor,
    W_xh: torch.Tensor,
    W_hh: torch.Tensor,
    b_h: torch.Tensor,
) -> torch.Tensor:
    """多步：Xs 形状 (seq_len, batch, d)，返回 (seq_len, batch, h)。"""
    Hl = H0
    Y = []
    for slen in range(Xs.shape[0]):
        Hl = rnn_step(Xs[slen], Hl, W_xh, W_hh, b_h)
        Y.append(Hl)
    return torch.stack(Y)
