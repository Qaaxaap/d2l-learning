# B20 代码题

把你的实现写在 `work/b20/b20.py` 里，函数名与签名必须和下面一致。

验收：`just check b20`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章做什么

把 B19 的数据管道与 RNN 前向接起来，**训练一个能记住文本模式的字符级语言模型**。

四个零件：one-hot 输入、前向（含隐藏状态）、梯度裁剪、训练循环（含困惑度）。

验收分三类：**精确**（形状、与手算一致、裁剪后范数）、**结构**（状态是元组）、
**趋势**（困惑度明显下降）。

---

## T1 one-hot

```python
def to_onehot(X: torch.Tensor, vocab_size: int) -> list[torch.Tensor]:
    """X 形状 (batch, num_steps) 的整数张量。

    返回长度为 num_steps 的列表，每个元素形状 (batch, vocab_size)。
    """
```

| 检查 | 期望 |
|---|---|
| 返回元素的个数 | `num_steps` |
| 每个元素形状 | `(batch, vocab_size)` |
| 每个行向量 | 元素和 1，只有对应索引那一位是 1 |

**提示**：`torch.nn.functional.one_hot(t, num_classes)` 接受一维张量，
返回 `(len(t), num_classes)`。

## T2 隐藏状态与前向

```python
def init_rnn_state(batch_size: int, num_hiddens: int, device):
    """返回 (H,)，H 形状 (batch_size, num_hiddens)，全 0。"""


def rnn_forward(inputs, state, params):
    """inputs 是 num_steps 个 (batch, vocab)；
    params 是 (W_xh, W_hh, b_h, W_hq, b_q)；state 是 (H,)。

    每步：H = tanh(X_t @ W_xh + H @ W_hh + b_h)
          Y_t = H @ W_hq + b_q        ← logits，不做 softmax
    返回 (outputs, (H_new,))。
    """
```

| 检查 | 期望 |
|---|---|
| `init_rnn_state` | 返回元组，内含一个全 0 张量，形状 `(batch, h)` |
| 每步输出形状 | `(batch, vocab)` |
| 输出与手算 | 一致 |
| 返回的最终状态 | 等于最后一个时间步的 `H` |

**`Y_t` 是 logits**，不要加 softmax——`CrossEntropyLoss` 内部会做。

## T3 梯度裁剪

```python
def grad_clipping(params, theta: float) -> float:
    """把所有参数的梯度当成一个向量，范数超过 theta 时整体按比例缩。

    返回裁剪前的范数。
    """
```

| 检查 | 期望 |
|---|---|
| 范数超过 `theta` | 裁剪后范数等于 `theta`，方向不变 |
| 范数不超过 `theta` | 梯度不变 |
| 参数本身 | **不能被修改**（改的是梯度）|
| 返回值 | 裁剪前的范数 |

## T4 训练

```python
def train_rnn(corpus_indices, vocab_size, num_hiddens, num_epochs, lr,
              batch_size, num_steps, device, use_random=True):
    """训练字符级 RNN，返回每个 epoch 的平均困惑度列表。

    use_random=True 用随机采样（每批重置隐藏状态），
    False 用相邻采样（跨批传递，每次读新批前 detach）。
    """
```

| 检查 | 期望 |
|---|---|
| 返回 | 长度等于 `num_epochs` 的困惑度列表 |
| 初始困惑度 | 接近词表大小（瞎猜水平）|
| 10 epoch 后 | 明显低于词表大小 |
| 两种采样 | 都能训练成功 |

**测试语料** `"abcd" * 200`（周期性），词表 4。断言用的超参：
`num_hiddens=32, lr=0.5, batch_size=8, num_steps=8, num_epochs=10`。
参考实现跑出来困惑度从 3.98 降到 1.00。

**训练函数里要自己建参数与优化器**（用 `nn.Parameter` 与 `torch.optim.SGD`），
采样可以复用你 B19 写的两个函数。

## T5 改错

下面这段想训练字符级 RNN。**三处**问题。

```python
def to_onehot(X, vocab_size):
    return [torch.nn.functional.one_hot(x, vocab_size).float() for x in X.T]

def rnn_forward(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(X @ W_xh + H @ W_hh + b_h)
        outputs.append(torch.softmax(H @ W_hq + b_q, dim=1))
    return outputs, (H,)

def grad_clipping(params, theta):
    norm = torch.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for p in params:
            p.data *= theta / norm
```

三处分别指：**`to_onehot` 的拆法**、**输出的激活**、**裁剪的对象**。

三段提示：第一处会让每个时间步的输入形状变成 `(num_steps,)` 而不是 `(batch,)`；
第二处不报错，但训练几乎不动；第三处不报错，但模型会越训越小。
