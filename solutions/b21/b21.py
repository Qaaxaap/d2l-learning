"""B21 参考答案。做完再看。

跑一遍：
    python3 solutions/b21/b21.py
或者拿它验断言：
    python3 tools/check_b21.py solutions/b21/b21.py
"""

import torch
import torch.nn as nn


# ---------- T1 按书上的公式实现 GRU ----------

def gru_step_book(X, H, params):
    """书上的 GRU 单步。

    params 依次是 (W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_xh, W_hh, b_h)。
    返回新的 H。
    """
    W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_xh, W_hh, b_h = params
    R = torch.sigmoid(X @ W_xr + H @ W_hr + b_r)
    Z = torch.sigmoid(X @ W_xz + H @ W_hz + b_z)
    H_tilde = torch.tanh(X @ W_xh + (R * H) @ W_hh + b_h)
    return Z * H + (1 - Z) * H_tilde


# ---------- T2 按 torch 的约定实现 GRU ----------

def gru_step_torch(X, H, params):
    """torch nn.GRU 的约定。

    params 依次是 (W_ir, W_hr, b_ir, b_hr, W_iz, W_hz, b_iz, b_hz,
                   W_in, W_hn, b_in, b_hn)。
    权重形状 (d, h) 或 (h, h)，偏置 (h,)。
    """
    (W_ir, W_hr, b_ir, b_hr,
     W_iz, W_hz, b_iz, b_hz,
     W_in, W_hn, b_in, b_hn) = params
    r = torch.sigmoid(X @ W_ir + b_ir + H @ W_hr + b_hr)
    z = torch.sigmoid(X @ W_iz + b_iz + H @ W_hz + b_hz)
    n = torch.tanh(X @ W_in + b_in + r * (H @ W_hn + b_hn))
    return (1 - z) * n + z * H


# ---------- T3 LSTM ----------

def lstm_step(X, H, C, params):
    """书上的 LSTM 单步。

    params 依次是 (W_xi, W_hi, b_i, W_xf, W_hf, b_f,
                   W_xo, W_ho, b_o, W_xc, W_hc, b_c)。
    返回 (H_new, C_new)。
    """
    (W_xi, W_hi, b_i, W_xf, W_hf, b_f,
     W_xo, W_ho, b_o, W_xc, W_hc, b_c) = params
    I = torch.sigmoid(X @ W_xi + H @ W_hi + b_i)
    F = torch.sigmoid(X @ W_xf + H @ W_hf + b_f)
    O = torch.sigmoid(X @ W_xo + H @ W_ho + b_o)
    C_tilde = torch.tanh(X @ W_xc + H @ W_hc + b_c)
    C_new = F * C + I * C_tilde
    H_new = O * torch.tanh(C_new)
    return H_new, C_new


# ---------- T4 深层与双向的隐藏状态形状 ----------

def hidden_state_shape(num_layers, batch_size, num_hiddens,
                       bidirectional=False):
    """返回 nn.RNN/nn.GRU/nn.LSTM 的 h_n 形状（元组）。"""
    factor = 2 if bidirectional else 1
    return (num_layers * factor, batch_size, num_hiddens)


def output_feature_dim(num_hiddens, bidirectional=False):
    """双向时输出的最后一维是 2h。"""
    return num_hiddens * (2 if bidirectional else 1)


"""T5 改错

第 1 处：门的激活用了 tanh
    门的值域必须是 [0,1]（表示"通过多少"）。tanh 值域是 [-1,1]，负的门值会让
    "保留比例"变成负数，语义完全乱掉。
    改法：`torch.sigmoid`。

第 2 处：隐藏状态的组合写反了
    书上约定是 `H = Z * H_prev + (1 - Z) * H_tilde`（Z 是保留旧状态的比例）。
    写成 `(1-Z) * H_prev + Z * H_tilde` 就成了 torch 的约定，与手算对不上。
    改法：按题目的约定写。

第 3 处：LSTM 把 `C` 也当隐藏状态直接返回，忘了过输出门
    `H_t = O_t * tanh(C_t)`，`C_t` 与 `H_t` 是两个不同的量。
    直接返回 `C` 当 `H` 会让输出值域不受 [−1,1] 控制、且少了输出门的门控。
    改法：`H_new = O * torch.tanh(C_new)`，两者分别返回。
"""


if __name__ == "__main__":
    torch.manual_seed(0)
    d, h, n = 3, 4, 2
    X = torch.randn(n, d)
    H = torch.randn(n, h)
    params = tuple(torch.randn(d, h) for _ in range(3)) + \
             tuple(torch.randn(h, h) for _ in range(1)) + \
             (torch.randn(h),)
    # 简化：直接给一组完整的
    def rnd(*shape):
        return torch.randn(*shape)
    book = [rnd(d, h), rnd(h, h), rnd(h), rnd(d, h), rnd(h, h), rnd(h),
            rnd(d, h), rnd(h, h), rnd(h)]
    print("T1 GRU(书上):", tuple(gru_step_book(X, H, book).shape))

    torch_params = [rnd(d, h), rnd(h, h), rnd(h), rnd(h),
                    rnd(d, h), rnd(h, h), rnd(h), rnd(h),
                    rnd(d, h), rnd(h, h), rnd(h), rnd(h)]
    print("T2 GRU(torch):", tuple(gru_step_torch(X, H, torch_params).shape))

    C = torch.randn(n, h)
    lstm_params = [rnd(d, h), rnd(h, h), rnd(h)] * 3 + [rnd(d, h), rnd(h, h), rnd(h)]
    hn, cn = lstm_step(X, H, C, lstm_params)
    print("T3 LSTM:", tuple(hn.shape), tuple(cn.shape))
    print("T4 深层形状:", hidden_state_shape(3, 8, 16))
    print("T4 双向形状:", hidden_state_shape(3, 8, 16, bidirectional=True))
