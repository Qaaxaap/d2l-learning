"""B22 参考答案。做完再看。

跑一遍：
    python3 solutions/b22/b22.py
或者拿它验断言：
    python3 tools/check_b22.py solutions/b22/b22.py
"""

import math

import torch
import torch.nn as nn


# ---------- T1 编码器 ----------

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, num_hiddens: int,
                 num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers)

    def forward(self, X, state=None):
        """X 形状 (seq_len, batch)。返回 (outputs, state)。

        outputs: (seq_len, batch, num_hiddens)
        state:   (num_layers, batch, num_hiddens)
        """
        X = self.embedding(X)          # (seq_len, batch, embed)
        return self.rnn(X, state)


# ---------- T2 解码器 ----------

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, num_hiddens: int,
                 num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_state):
        """把编码器的最终状态拿来做解码器的初始状态。"""
        return enc_state.detach()

    def forward(self, X, state):
        """X 形状 (seq_len, batch)。返回 (logits, state)。

        logits: (seq_len, batch, vocab_size)
        """
        X = self.embedding(X)
        out, state = self.rnn(X, state)
        return self.dense(out), state


# ---------- T3 seq2seq ----------

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, enc_state=None):
        """返回 (logits, dec_state)，logits 形状 (dec_len, batch, vocab_size)。"""
        _, state = self.encoder(enc_X, enc_state)
        state = self.decoder.init_state(state)
        return self.decoder(dec_X, state)


# ---------- T4 束搜索 ----------

def beam_search(step_fn, init_state, beam_size: int, max_len: int,
                bos_token: int, eos_token: int, alpha: float = 0.75):
    """通用束搜索。

    step_fn(token, state) -> (log_probs, next_state)
      token 是一个整数，state 是任意对象
      log_probs 形状 (vocab_size,)，是对数概率

    返回 (best_tokens, best_score)，best_tokens 不含 <eos>。
    """
    candidates = [([], 0.0, init_state)]     # (tokens, 对数概率和, state)
    finished = []                            # (tokens, 对数概率和)

    for _ in range(max_len):
        next_candidates = []
        for tokens, score, state in candidates:
            prev = tokens[-1] if tokens else bos_token
            log_probs, new_state = step_fn(prev, state)
            # 每个候选要扩展全部词表，再从 k*|V| 个里取前 k
            for idx in range(log_probs.size(0)):
                v = log_probs[idx].item()
                new_tokens = tokens + [idx]
                new_score = score + v
                if idx == eos_token:
                    finished.append((new_tokens[:-1], new_score))
                else:
                    next_candidates.append((new_tokens, new_score, new_state))
        if not next_candidates:
            break
        next_candidates.sort(key=lambda c: c[1], reverse=True)
        candidates = next_candidates[:beam_size]

    # 长度惩罚后比较
    def norm(tokens, score):
        L = max(len(tokens), 1)
        return score / (L ** alpha)

    pool = finished if finished else [(t, s) for t, s, _ in candidates]
    if not pool:
        return [], float("-inf")
    best_tokens, best_score = max(pool, key=lambda p: norm(*p))
    return best_tokens, norm(best_tokens, best_score)


"""T5 改错

第 1 处：`Encoder.forward` 直接返回了 `self.rnn(X)` 的第一个元素
    `nn.GRU` 返回 `(outputs, h_n)`。背景变量要的是 `h_n`（最后一步），
    不是 `outputs`（每一步）。用 `outputs` 当初始化状态会让解码器拿到形状不对的张量。
    改法：返回整个 `(outputs, h_n)`，或者在 Seq2Seq 里取第二个元素。

第 2 处：`Decoder.init_state` 没有 `detach`
    编码器的计算图会一直连到解码器的训练里，显存翻倍、梯度多算一遍。
    改法：`enc_state.detach()`。

第 3 处：`beam_search` 的分数用了概率连乘
    概率连乘到十几步就下溢成 0，所有候选分数都是 0，无法比较。
    改法：全程用对数概率相加。
"""


if __name__ == "__main__":
    torch.manual_seed(0)
    enc = Encoder(vocab_size=10, embed_size=8, num_hiddens=16)
    X = torch.randint(0, 10, (5, 3))
    out, state = enc(X)
    print("T1 编码器:", tuple(out.shape), tuple(state.shape))

    dec = Decoder(vocab_size=12, embed_size=8, num_hiddens=16)
    logits, dstate = dec(torch.randint(0, 12, (4, 3)), dec.init_state(state))
    print("T2 解码器:", tuple(logits.shape), tuple(dstate.shape))

    model = Seq2Seq(enc, dec)
    out, _ = model(X, torch.randint(0, 12, (4, 3)))
    print("T3 seq2seq:", tuple(out.shape))
