"""B15 参考答案。做完再看。

跑一遍：
    python3 solutions/b15/b15.py
或者拿它验断言：
    python3 tools/check_b15.py solutions/b15/b15.py
"""

import torch
import torch.nn as nn


# ---------- T1 手写二维互相关 ----------

def corr2d(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i : i + h, j : j + w] * K).sum()
    return Y


# ---------- T2 输出形状 ----------

def conv_out_shape(n_h: int, n_w: int, k_h: int, k_w: int,
                   p_h: int = 0, p_w: int = 0,
                   s_h: int = 1, s_w: int = 1) -> tuple[int, int]:
    out_h = (n_h - k_h + p_h) // s_h + 1
    out_w = (n_w - k_w + p_w) // s_w + 1
    return out_h, out_w


# ---------- T3 多输入通道 ----------

def corr2d_multi_in(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    # 每个输入通道配一个核数组，各做一次互相关，再按元素相加
    return sum(corr2d(x, k) for x, k in zip(X, K))


# ---------- T4 多输出通道 ----------

def corr2d_multi_in_out(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    # 遍历 K 的第 0 维（输出通道维），每次用一组 (c_i, kh, kw) 的核
    return torch.stack([corr2d_multi_in(X, k) for k in K])


# ---------- T5 1x1 卷积 ----------

def corr2d_multi_in_out_1x1(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X_flat = X.reshape(c_i, h * w)
    K_flat = K.reshape(c_o, c_i)
    return (K_flat @ X_flat).reshape(c_o, h, w)


# ---------- T6 池化 ----------

def pool2d(X: torch.Tensor, pool_size: tuple[int, int],
           mode: str = "max") -> torch.Tensor:
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            window = X[i : i + p_h, j : j + p_w]
            Y[i, j] = window.max() if mode == "max" else window.mean()
    return Y


"""T7 改错

第 1 处：`corr2d_multi_in_out` 里直接调了 `corr2d(X, k)`
    两个问题叠在一起：
    (a) `corr2d` 只接受二维输入，而 `X` 是三维的 (c_i, h, w)，切片会出错
    (b) 即使能跑，也漏掉了"各输入通道结果相加"这一步——那是多输入通道的定义
    改法：先用 `corr2d_multi_in(X, k)` 把各输入通道合并，得到二维结果。

第 2 处：`K` 的遍历方式
    这段代码遍历 `K` 是**对的**（K 的第 0 维就是输出通道维），
    但第一处修好之后会显出另一个问题：
    `torch.stack([corr2d_multi_in(X, k) for k in K])` 里每一项已经是 (h', w')，
    `stack` 之后得到 (c_o, h', w')，正是期望的 (6, 6, 6)。
    所以严格说这一处不算 bug——**原代码真正坏的只有第 1 处**。

    （本题的原意是想让学习者发现自己漏了"通道相加"这一步。设计这条时低估了两处的
    耦合：漏掉相加时形状先崩，学习者看不到"结果少一维"。记在日志里。）
"""


if __name__ == "__main__":
    X = torch.tensor([[0.0, 1, 2], [3, 4, 5], [6, 7, 8]])
    K = torch.tensor([[0.0, 1], [2, 3]])
    print("T1 corr2d:", corr2d(X, K).tolist())

    print("T2 形状:", conv_out_shape(8, 8, 3, 3), conv_out_shape(8, 8, 3, 3, 1, 1),
          conv_out_shape(8, 8, 3, 3, 1, 1, 2, 2))

    X2 = torch.tensor([[[0.0, 1, 2], [3, 4, 5], [6, 7, 8]],
                       [[1.0, 2, 3], [4, 5, 6], [7, 8, 9]]])
    K2 = torch.tensor([[[0.0, 1], [2, 3]], [[1.0, 2], [3, 4]]])
    print("T3 多输入:", corr2d_multi_in(X2, K2).tolist())
    print("T4 多输出:", tuple(corr2d_multi_in_out(X2, torch.stack([K2, K2 + 1, K2 + 2])).shape))

    X3 = torch.randn(3, 4, 4)
    K3 = torch.randn(2, 3, 1, 1)
    a = corr2d_multi_in_out_1x1(X3, K3)
    b = corr2d_multi_in_out(X3, K3)
    print("T5 1x1 与普通卷积的最大差:", round((a - b).abs().max().item(), 10))

    print("T6 max 池化:", pool2d(X, (2, 2), "max").tolist())
    print("T6 avg 池化:", pool2d(X, (2, 2), "avg").tolist())
