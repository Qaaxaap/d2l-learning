"""B20 参考答案。做完再看。

跑一遍：
    python3 solutions/b20/b20.py
或者拿它验断言：
    python3 tools/check_b20.py solutions/b20/b20.py
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- T1 one-hot ----------

def to_onehot(X: torch.Tensor, vocab_size: int) -> list[torch.Tensor]:
    """X 形状 (batch, num_steps) 的整数张量。

    返回长度为 num_steps 的列表，每个元素形状 (batch, vocab_size)。
    """
    return [
        F.one_hot(X[:, t], num_classes=vocab_size).float()
        for t in range(X.shape[1])
    ]


# ---------- T2 隐藏状态与前向 ----------

def init_rnn_state(batch_size: int, num_hiddens: int, device):
    return (torch.zeros(batch_size, num_hiddens, device=device),)


def rnn_forward(inputs, state, params):
    """inputs 是 num_steps 个 (batch, vocab)；
    params 是 (W_xh, W_hh, b_h, W_hq, b_q)；state 是 (H,)。

    返回 (outputs, (H_new,))，outputs 是 num_steps 个 (batch, vocab) 的 logits。
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(X @ W_xh + H @ W_hh + b_h)
        outputs.append(H @ W_hq + b_q)
    return outputs, (H,)


# ---------- T3 梯度裁剪 ----------

def grad_clipping(params, theta: float) -> float:
    """把所有参数的梯度当成一个向量，范数超过 theta 时整体按比例缩。

    返回裁剪前的范数。
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
    norm_value = norm.item()
    if norm_value > theta:
        scale = theta / norm_value
        for g in grads:
            g.mul_(scale)
    return norm_value


# ---------- T4 训练 ----------

def make_params(vocab_size: int, num_hiddens: int, device):
    def one(*shape):
        return nn.Parameter(torch.randn(*shape, device=device) * 0.01)

    return (
        one(vocab_size, num_hiddens),      # W_xh
        one(num_hiddens, num_hiddens),     # W_hh
        nn.Parameter(torch.zeros(num_hiddens, device=device)),
        one(num_hiddens, vocab_size),      # W_hq
        nn.Parameter(torch.zeros(vocab_size, device=device)),
    )


def data_iter_random(corpus_indices, batch_size, num_steps, device):
    n = len(corpus_indices)
    num_examples = (n - 1) // num_steps
    epoch_size = num_examples // batch_size
    starts = list(range(num_examples))
    random.shuffle(starts)
    for i in range(epoch_size):
        batch = starts[i * batch_size: (i + 1) * batch_size]
        X = [corpus_indices[j * num_steps: j * num_steps + num_steps] for j in batch]
        Y = [corpus_indices[j * num_steps + 1: j * num_steps + 1 + num_steps] for j in batch]
        yield (torch.tensor(X, device=device), torch.tensor(Y, device=device))


def data_iter_consecutive(corpus_indices, batch_size, num_steps, device):
    data = torch.tensor(corpus_indices, device=device)
    batch_len = len(data) // batch_size
    indices = data[: batch_size * batch_len].reshape(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        s = i * num_steps
        yield indices[:, s: s + num_steps], indices[:, s + 1: s + 1 + num_steps]


def train_rnn(corpus_indices, vocab_size, num_hiddens, num_epochs, lr,
              batch_size, num_steps, device, use_random=True):
    """返回每个 epoch 的平均困惑度列表。"""
    params = make_params(vocab_size, num_hiddens, device)
    optimizer = torch.optim.SGD(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    iter_fn = data_iter_random if use_random else data_iter_consecutive

    perplexities = []
    for _ in range(num_epochs):
        state = None
        total_loss = 0.0
        n_batches = 0
        for X, Y in iter_fn(corpus_indices, batch_size, num_steps, device):
            if state is None or use_random:
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
                state = tuple(s.detach() for s in state)

            inputs = to_onehot(X, vocab_size)
            outputs, state = rnn_forward(inputs, state, params)

            # 每个时间步的 logits (batch, vocab) 与标签 (batch,) 算交叉熵
            loss = sum(loss_fn(outputs[t], Y[:, t]) for t in range(num_steps)) / num_steps

            optimizer.zero_grad()
            loss.backward()
            grad_clipping(params, 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        perplexities.append(math.exp(total_loss / n_batches))
    return perplexities


"""T5 改错

第 1 处：`to_onehot` 里按列取、再转置
    题目给的 `X` 是 `(batch, num_steps)`。写成 `X.T` 再遍历会把批量维当时间步，
    每步拿到的是 `(num_steps,)` 而不是 `(batch,)`，one-hot 之后形状全错。
    改法：按 `X[:, t]` 取时间步。

第 2 处：输出 `Y` 上又做了一次 softmax
    `CrossEntropyLoss` 内部已经做 log-softmax 了。再 softmax 一次等于把概率又压一遍，
    损失能算但数值不对（梯度被严重衰减），训练几乎不动。
    改法：`Y_t` 直接给 `H @ W_hq + b_q` 的 logits。

第 3 处：用 `p.data *= theta / norm` 而不是改 `p.grad`
    这会把**模型参数本身**缩小，不是裁剪梯度。参数越训越小，模型废掉。
    改法：`p.grad.mul_(theta / norm)`。
"""


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.tensor([[0, 1, 2], [2, 1, 0]])
    one = to_onehot(X, 3)
    print("T1 one-hot:", len(one), tuple(one[0].shape))

    corpus = "abcd" * 50
    c2i = {ch: i for i, ch in enumerate(sorted(set(corpus)))}
    idx = [c2i[ch] for ch in corpus]
    ppl = train_rnn(idx, 4, 32, 3, 0.5, 16, 8, device, use_random=True)
    print("T4 随机采样困惑度:", [round(p, 3) for p in ppl])
    ppl2 = train_rnn(idx, 4, 32, 3, 0.5, 16, 8, device, use_random=False)
    print("T4 相邻采样困惑度:", [round(p, 3) for p in ppl2])
