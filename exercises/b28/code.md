# B28 代码题

把你的实现写在 `work/b28/b28.py` 里，函数名与签名必须和下面一致。

验收：`just check b28`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章做什么

写词向量的五个核心计算：跳字模型的损失、负采样的损失、共现矩阵、余弦相似度、词的类比。

**全部是纯数值计算**，语料与向量都是人造的小张量，断言一秒内跑完。

---

## T1 跳字模型的条件概率与损失

```python
def skipgram_log_prob(center_vec: torch.Tensor, context_idx: int,
                      all_vecs: torch.Tensor) -> float:
    """返回 log P(w_o | w_c)。

    center_vec 形状 (d,)。context_idx 是背景词在词表里的下标。
    all_vecs 形状 (V, d)，所有词当背景词时的向量。
    """


def skipgram_loss(center_vecs: torch.Tensor, context_idx: torch.Tensor,
                  all_vecs: torch.Tensor) -> float:
    """一批 (中心词, 背景词) 对的平均损失。

    center_vecs 形状 (n, d)，context_idx 形状 (n,) 是背景词下标。
    """
```

公式（讲义 2.2、2.3）：

$$P(w_o \mid w_c) = \frac{\exp(\boldsymbol{u}_o^\top \boldsymbol{v}_c)}
{\sum_{i \in \mathcal{V}} \exp(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}$$

$$\text{loss} = -\frac{1}{n}\sum_{k=1}^{n} \log P(w_o^{(k)} \mid w_c^{(k)})$$

**对数形式的写法**：$\log P = \boldsymbol{u}_o^\top\boldsymbol{v}_c
- \log\sum_i \exp(\boldsymbol{u}_i^\top\boldsymbol{v}_c)$。
后一项用 `torch.logsumexp` 算，不要先取 exp 再求和。

| 检查 | 期望 | 类型 |
|---|---|---|
| 单对 log 概率 | 与手算一致（误差 1e-5）| 精确 |
| 任意词的 log 概率 | 都是负数或 0 | 精确 |
| 一批损失 | 等于逐个平均 | 精确 |

## T2 负采样的损失

```python
def negative_sampling_loss(center_vecs: torch.Tensor,
                           pos_vecs: torch.Tensor,
                           neg_vecs: torch.Tensor) -> float:
    """负采样的损失。

    center_vecs 形状 (n, d)：每对的核心词向量。
    pos_vecs 形状 (n, d)：正样本背景词向量。
    neg_vecs 形状 (n, K, d)：每对的 K 个噪声词向量。
    """
```

公式（讲义 5.2）：

$$\ell = -\frac{1}{n}\sum_{k=1}^{n}
\left[\log\sigma(\boldsymbol{u}_o^\top\boldsymbol{v}_c)
+ \sum_{i=1}^{K}\log\left(1 - \sigma(\boldsymbol{u}_{w_i}^\top\boldsymbol{v}_c)\right)\right]$$

**注意负样本那一项**：$1 - \sigma(x) = \sigma(-x)$。
用 `F.logsigmoid(-neg_score)` 算，不要先算 sigmoid 再取 `1 - p`（数值不稳）。

| 检查 | 期望 | 类型 |
|---|---|---|
| 正样本同向、负样本反向 | 损失比反过来小 | 趋势 |
| 手算对比 | 误差 1e-4 | 精确 |

## T3 共现矩阵

```python
def cooccurrence(corpus: list[int], vocab_size: int,
                 window: int = 2) -> torch.Tensor:
    """统计共现次数。

    x[i, j] 是词 j 出现在以词 i 为中心词的窗口里的次数。
    返回形状 (vocab_size, vocab_size) 的 float 张量。
    """
```

**要对称**。中心词 $i$ 的窗口里有 $j$，等价于 $j$ 的窗口里有 $i$ 吗？
不是——要看窗口是不是双向的。本节与 GloVe 都用双向窗口，所以 $x_{ij} = x_{ji}$。

**验证方法**：写完检查 `torch.allclose(X, X.T)`。

| 检查 | 期望 | 类型 |
|---|---|---|
| 语料 `[0,1,2,1,0]` 窗口 1 | 矩阵每项与手算一致 | 精确 |
| 对称性 | `X == X.T` | 精确 |

**手算提示**：逐个位置数。位置 0 的词 0 只见位置 1 的词 1，所以 `x[0,1]` 加 1。

## T4 余弦相似度

```python
def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """逐行余弦相似度。a 与 b 形状都是 (n, d)，返回 (n,)。"""
```

$$\cos(\boldsymbol{x}, \boldsymbol{y}) = \frac{\boldsymbol{x}^\top \boldsymbol{y}}
{\|\boldsymbol{x}\| \|\boldsymbol{y}\|}$$

**注意除零**。零向量的范数是 0。用 `clamp_min(1e-8)` 保护。

| 检查 | 期望 | 类型 |
|---|---|---|
| 同向与垂直的三组 | `[1, 0, 1]` | 精确 |
| 方向相同、长度差 100 倍 | 都是 1 | 精确 |

## T5 词的类比

```python
def analogy(a: int, b: int, c: int, vecs: torch.Tensor,
            exclude: list[int] | None = None) -> int:
    """解 a - b + c，返回最相似的词索引，排除 a、b、c 与 exclude 里的。"""
```

**三步**：

1. 算目标向量 $\boldsymbol{v}_a - \boldsymbol{v}_b + \boldsymbol{v}_c$。
2. 与词表里每个词算余弦相似度。
3. 取相似度最高且不在排除名单里的下标。

**为什么必须排除输入词**：$\boldsymbol{v}_a$ 与目标向量高度相似，不排除的话结果总是 $a$。

| 检查 | 期望 | 类型 |
|---|---|---|
| 结果范围 | 在 `[0, V)` 内 | 精确 |
| 结果 | 不是 a、b、c 之一 | 精确 |

## T6 改错

下面这段是跳字损失与共现矩阵的写法。**三处**问题。

```python
def skipgram_loss(center_vecs, context_idx, all_vecs):
    scores = center_vecs @ all_vecs.T
    n = scores.shape[0]
    target = scores[torch.arange(n), context_idx]
    log_probs = F.log_softmax(scores, dim=0)
    picked = log_probs[torch.arange(n), context_idx]
    return float(-picked.mean())


def cooccurrence(corpus, vocab_size, window=2):
    X = torch.zeros(vocab_size, vocab_size)
    n = len(corpus)
    for t in range(n):
        c = corpus[t]
        for j in range(max(0, t - window), min(n, t + window + 1)):
            if j == t:
                continue
            X[c, corpus[j]] += 1
    return X
```

三处分别指：**`log_softmax` 的维度**、**共现矩阵的对称性**、**`target` 那行没用上**。

三段提示：第一处会在样本之间归一化；第二处让 `X` 不等于 `X.T`；
第三处说明写了两个版本的 `picked`，其中一个多余。
