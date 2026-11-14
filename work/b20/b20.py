# def to_onehot(X: torch.Tensor, vocab_size: int) -> list[torch.Tensor]:
#     """X: (batch, num_steps) 的整数张量。

#     返回长度 num_steps 的列表，第 t 个元素是第 t 步的输入。
#     """
#     return [torch.nn.functional.one_hot(x, vocab_size).float() for x in X] 需要转置，否则形状不对


# def rnn_forward(inputs: list[torch.Tensor], state: tuple[torch.Tensor],
#                 params: tuple[torch.Tensor, ...]) -> tuple[list[torch.Tensor],
#                                                           tuple[torch.Tensor]]:
#     """inputs: num_steps 个 (batch, vocab)
#     state: (H,)，H 形状 (batch, h)
#     params: (W_xh, W_hh, b_h, W_hq, b_q)
#     返回 (outputs, (H_new,))，outputs 里每个元素是 (batch, vocab) 的 logits。
#     """
#     W_xh, W_hh, b_h, W_hq, b_q = params
#     H, = state
#     outputs = []
#     for X in inputs:                                  # X: (batch, vocab)
#         H = torch.tanh(X @ W_xh + H @ W_hh + b_h)     # H: (batch, h)
#         outputs.append(torch.softmax(H @ W_hq + b_q, dim=1)) 不需要自己softmax
#     return outputs, (H,)


# def grad_clipping(params: list[torch.Tensor], theta: float) -> float:
#     """params 里每个元素的 .grad 与它本身同形状。

#     返回裁剪前的总范数。
#     """
#     norm = torch.sqrt(sum((p.grad ** 2).sum() for p in params))
#     if norm > theta:
#         for p in params:
#             p.data *= theta / norm 缩放错东西了 要缩放梯度
#     return norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch import Tensor
from typing import Any, Generator


def to_onehot(X: torch.Tensor, vocab_size: int) -> list[torch.Tensor]:
    """X 形状 (batch, num_steps) 的整数张量。

    返回长度为 num_steps 的列表，每个元素形状 (batch, vocab_size)。
    """
    return list(F.one_hot(X.long(), vocab_size).float().unbind(dim=1))


def init_rnn_state(
    batch_size: int, num_hiddens: int, device: str
) -> tuple[torch.Tensor]:
    """返回 (H,)，H 形状 (batch_size, num_hiddens)，全 0。"""
    return (torch.zeros((batch_size, num_hiddens)).to(device),)


def rnn_forward(
    inputs: list[torch.Tensor],
    state: tuple[torch.Tensor],
    params: tuple[torch.Tensor, ...],
) -> tuple[list[torch.Tensor], tuple[torch.Tensor]]:
    """多步前向。返回 (outputs, (H_new,))。

    inputs: 长度 num_steps 的列表，每个元素形状 (batch, vocab)，是 one-hot。
    state: (H,)，H 形状 (batch, h)，是进入第 0 步之前的状态。
    params: (W_xh, W_hh, b_h, W_hq, b_q)。

    每步：H = tanh(X_t @ W_xh + H @ W_hh + b_h)
          Y_t = H @ W_hq + b_q        ← logits，不做 softmax

    outputs: 长度 num_steps 的列表，outputs[t] 形状 (batch, vocab)，
             是第 t 步看过 X_t 之后的 logits。
    H_new:   (H,)，最后一步的状态，与 outputs[-1] 是同一时刻。
    """
    H = state[0]
    W_xh, W_hh, b_h, W_hq, b_q = params
    out = []
    for X in inputs:
        H = torch.tanh(X @ W_xh + H @ W_hh + b_h)
        Y = H @ W_hq + b_q
        out.append(Y)
    return (out, (H,))


def grad_clipping(params: list[torch.Tensor], theta: float) -> float:
    """把所有参数的梯度当成一个向量，范数超过 theta 时整体按比例缩。

    分两步：

    1. 算总范数 norm = sqrt(所有参数的梯度元素的平方和)。
    2. norm > theta 时，每个梯度乘上 theta / norm；否则不动。

    返回**第 1 步算出的那个 norm**，也就是裁剪之前的值。
    它不管最后裁没裁都照原样返回。

    举例：梯度是 [3, 4]，总范数 5，theta = 1。
    裁剪后梯度变成 [0.6, 0.8]，范数为 1。**函数返回 5，不是 1**。

    再举例：梯度是 [0.3, 0.4]，总范数 0.5，theta = 1。
    没触发裁剪，梯度不变。函数返回 0.5。

    用途是观察训练。频繁超过 theta 说明梯度太大，常见原因是
    学习率过高、初始化不好或序列太长。
    """
    norm = torch.tensor(0, dtype=torch.float32, device=params[0].device)
    for t in params:
        norm += (t.grad**2).sum()
    norm = norm.item() ** 0.5
    if norm > theta:
        for t in params:
            t.grad *= theta / norm
    return norm


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


def train_rnn(
    corpus_indices: list[int],
    vocab_size: int,
    num_hiddens: int,
    num_epochs: int,
    lr: float,
    batch_size: int,
    num_steps: int,
    device: str,
    use_random: bool = True,
) -> list[float]:
    """训练字符级 RNN，返回每个 epoch 的平均困惑度列表。

    use_random=True 用随机采样（每批重置隐藏状态），
    False 用相邻采样（跨批传递，每次读新批前 detach）。
    """

    def _rand(*shape: int, std: float = 0.01) -> torch.Tensor:
        return (torch.randn(*shape, device=device) * std).requires_grad_(True)

    W_xh = _rand(vocab_size, num_hiddens)
    W_hh = _rand(num_hiddens, num_hiddens)
    b_h = torch.zeros(num_hiddens, device=device, requires_grad=True)
    W_hq = _rand(num_hiddens, vocab_size)
    b_q = torch.zeros(vocab_size, device=device, requires_grad=True)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    optimizer = torch.optim.SGD(params, lr=lr)
    theta = 1.0
    ppls: list[float] = []
    for _ in range(num_epochs):
        data_iter = (
            data_iter_random(corpus_indices, batch_size, num_steps)
            if use_random
            else data_iter_consecutive(corpus_indices, batch_size, num_steps)
        )
        state = init_rnn_state(batch_size, num_hiddens, device)
        total_loss = 0.0
        n_batches = 0
        for X, Y in data_iter:
            X = X.to(device)
            Y = Y.to(device)
            if use_random:
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
                state = (state[0].detach(),)
            inputs = to_onehot(X, vocab_size)
            outputs, state = rnn_forward(inputs, state, tuple(params))
            logits = torch.cat(outputs, dim=0)
            target = Y.t().reshape(-1)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            grad_clipping(params, theta)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        ppls.append(torch.exp(torch.tensor(avg_loss)).item())
    return ppls
