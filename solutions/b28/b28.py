"""B28 参考答案。做完再看。"""

import torch
import torch.nn.functional as F


# ---------- T1 跳字模型的条件概率与损失 ----------

def skipgram_log_prob(center_vec: torch.Tensor, context_idx: int,
                      all_vecs: torch.Tensor) -> float:
    """返回 log P(w_o | w_c)。

    center_vec 形状 (d,)。context_idx 是背景词在词表里的下标。
    all_vecs 形状 (V, d)，所有词当背景词时的向量。
    """
    scores = all_vecs @ center_vec                       # (V,)
    return float(scores[context_idx] - torch.logsumexp(scores, dim=0))


def skipgram_loss(center_vecs: torch.Tensor, context_idx: torch.Tensor,
                  all_vecs: torch.Tensor) -> float:
    """一批 (中心词, 背景词) 对的平均损失。

    center_vecs 形状 (n, d)，context_idx 形状 (n,) 是背景词下标。
    损失是 -mean(log P(w_o | w_c))。
    """
    scores = center_vecs @ all_vecs.T                    # (n, V)
    n = scores.shape[0]
    target = scores[torch.arange(n), context_idx]         # (n,)
    return float(-(target - torch.logsumexp(scores, dim=1)).mean())


# ---------- T2 负采样的损失 ----------

def negative_sampling_loss(center_vecs: torch.Tensor,
                           pos_vecs: torch.Tensor,
                           neg_vecs: torch.Tensor,
                           neg_weights: torch.Tensor | None = None) -> float:
    """负采样的损失。

    center_vecs 形状 (n, d)：每对的正样本中心词向量。
    pos_vecs 形状 (n, d)：正样本背景词向量。
    neg_vecs 形状 (n, K, d)：每对的 K 个噪声词向量。
    返回 -mean( log sigmoid(pos) + sum_k log(1 - sigmoid(neg)) )。
    """
    pos_score = (center_vecs * pos_vecs).sum(dim=1)                 # (n,)
    neg_score = torch.einsum("nd,nkd->nk", center_vecs, neg_vecs)   # (n, K)
    loss = -F.logsigmoid(pos_score) - F.logsigmoid(-neg_score).sum(dim=1)
    return float(loss.mean())


# ---------- T3 共现矩阵 ----------

def cooccurrence(corpus: list[int], vocab_size: int,
                 window: int = 2) -> torch.Tensor:
    """统计共现次数。

    corpus 是索引列表。x[i, j] 是词 j 出现在以词 i 为中心词的窗口里的次数。
    返回形状 (vocab_size, vocab_size) 的 float 张量。
    """
    X = torch.zeros(vocab_size, vocab_size)
    n = len(corpus)
    for t in range(n):
        c = corpus[t]
        for j in range(max(0, t - window), min(n, t + window + 1)):
            if j == t:
                continue
            X[c, corpus[j]] += 1
    return X


# ---------- T4 余弦相似度 ----------

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """逐行余弦相似度。a 与 b 形状都是 (n, d)，返回 (n,)。"""
    a_n = a / a.norm(dim=1, keepdim=True).clamp_min(1e-8)
    b_n = b / b.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (a_n * b_n).sum(dim=1)


# ---------- T5 词的类比 ----------

def analogy(a: int, b: int, c: int, vecs: torch.Tensor,
            exclude: list[int] | None = None) -> int:
    """解 a - b + c，返回最相似的词索引，排除 a、b、c 与 exclude 里的。"""
    target = vecs[a] - vecs[b] + vecs[c]
    sims = cosine_similarity(target.unsqueeze(0).expand_as(vecs), vecs)
    ban = set(exclude or []) | {a, b, c}
    for i in sims.argsort(descending=True).tolist():
        if i not in ban:
            return i
    return -1


"""T6 改错

第 1 处：`log_softmax` 的维度用了 `dim=0`
    `scores` 形状是 `(n, V)`，每个样本对词表的分布在第 1 维。
    `dim=0` 会在样本之间归一化。
    改法：`dim=1`。

第 2 处：负采样的负样本用了 `logsigmoid(neg_score)`
    负样本要的是 `log(1 - sigmoid(x))`，等于 `logsigmoid(-x)`。
    写成 `logsigmoid(x)` 让负样本的分数越大损失越小，方向反了。
    改法：`-F.logsigmoid(-neg_score)`。

第 3 处：共现矩阵只累加了 `X[c, corpus[j]] += 1`，没处理对称
    当窗口是双向的时，j 出现在 c 的窗口里等价于 c 出现在 j 的窗口里。
    要同时加 `X[corpus[j], c] += 1`，GloVe 才能得到对称的矩阵。
    改法：两处都加。
"""


if __name__ == "__main__":
    torch.manual_seed(0)
    V, d = 6, 4
    vecs = torch.randn(V, d)
    c = torch.randn(d)
    o = torch.randn(d)
    print("T1 单对 log 概率:", round(skipgram_log_prob(c, 1, vecs), 4))
    cs = torch.randn(3, d)
    cidx = torch.tensor([0, 2, 4])
    print("T1 一批损失:", round(skipgram_loss(cs, cidx, vecs), 4))

    n, K = 3, 2
    ce = torch.randn(n, d)
    po = torch.randn(n, d)
    ne = torch.randn(n, K, d)
    print("T2 负采样损失:", round(negative_sampling_loss(ce, po, ne), 4))

    X = cooccurrence([0, 1, 2, 1, 0], vocab_size=3, window=1)
    print("T3 共现矩阵:\n", X)

    a = torch.randn(2, d)
    b = torch.randn(2, d)
    print("T4 余弦:", cosine_similarity(a, b).tolist())

    print("T5 类比:", analogy(0, 1, 2, vecs))
