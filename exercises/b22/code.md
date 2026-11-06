# B22 代码题

把你的实现写在 `work/b22/b22.py` 里，函数名与签名必须和下面一致。

验收：`just check b22`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章做什么

把两个 RNN 拼成 seq2seq，再写一个通用的束搜索。

**这一章不训练**。验证的是形状、拼接关系、搜索算法的行为。

---

## T1 编码器

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, num_hiddens: int,
                 num_layers: int = 1):
        """Embedding + nn.GRU。"""

    def forward(self, X, state=None):
        """X 形状 (seq_len, batch)。返回 (outputs, state)。

        outputs: (seq_len, batch, num_hiddens)
        state:   (num_layers, batch, num_hiddens)
        """
```

**注意 torch 的 RNN 默认 seq_len 在前**，不加 `batch_first=True`。

| 检查 | 期望 |
|---|---|
| `outputs` 形状 | `(seq_len, batch, h)` |
| `state` 形状 | `(num_layers, batch, h)` |
| 单层时 `state[-1]` | 等于 `outputs[-1]` |

## T2 解码器

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, num_hiddens: int,
                 num_layers: int = 1):
        """Embedding + nn.GRU + Linear(num_hiddens, vocab_size)。"""

    def init_state(self, enc_state):
        """把编码器的最终状态拿来做解码器的初始状态。"""

    def forward(self, X, state):
        """X 形状 (seq_len, batch)。返回 (logits, state)。

        logits: (seq_len, batch, vocab_size)
        """
```

| 检查 | 期望 |
|---|---|
| `logits` 形状 | `(seq_len, batch, vocab_size)` |
| `init_state` 返回的张量 | **`requires_grad` 为 False（detach 过）** |
| `init_state` 的数值 | 与输入相同（detach 只切梯度） |

## T3 seq2seq

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder): ...

    def forward(self, enc_X, dec_X, enc_state=None):
        """返回 (logits, dec_state)。"""
```

| 检查 | 期望 |
|---|---|
| `logits` 形状 | `(dec_len, batch, vocab_size)` |
| 与手动串一遍 | 一致 |

## T4 束搜索

```python
def beam_search(step_fn, init_state, beam_size: int, max_len: int,
                bos_token: int, eos_token: int, alpha: float = 0.75):
    """通用束搜索。

    step_fn(token, state) -> (log_probs, next_state)
      token 是整数；log_probs 形状 (vocab_size,)，是对数概率

    返回 (best_tokens, best_score)。best_tokens 不含 <eos>。
    """
```

| 检查 | 期望 |
|---|---|
| 束宽 1 | 等价于贪心 |
| 束宽 2（讲义第 3 节的例子）| 找到比贪心更好的序列 |
| 第一步就是 `<eos>` | 返回空序列 |
| 分数 | 用**对数概率相加**，最后除以 $L^\alpha$ |

**打分的顺序**：先累积对数概率，最后统一除以 `L ** alpha`。
`alpha=0.75`。

## T5 改错

下面这段想实现 seq2seq 与束搜索。**三处**问题。

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens)

    def forward(self, X):
        return self.rnn(self.embedding(X))[0]

class Decoder(nn.Module):
    def init_state(self, enc_state):
        return enc_state

def beam_search(step_fn, init_state, beam_size, max_len, bos, eos, alpha=0.75):
    candidates = [([], 1.0, init_state)]
    for _ in range(max_len):
        new_candidates = []
        for tokens, score, state in candidates:
            prev = tokens[-1] if tokens else bos
            probs, new_state = step_fn(prev, state)
            for idx in range(probs.size(0)):
                new_tokens = tokens + [idx]
                new_score = score * probs[idx].item()
                new_candidates.append((new_tokens, new_score, new_state))
        new_candidates.sort(key=lambda c: c[1], reverse=True)
        candidates = new_candidates[:beam_size]
    best = candidates[0]
    return best[0], best[1] / (len(best[0]) ** alpha)
```

三处分别指：**编码器的返回值**、**`init_state`**、**束搜索的分数表示**。

三段提示：第一处让背景变量拿到了形状不对的张量；第二处让显存翻倍；
第三处跑到十几步时所有分数都变成 0。
