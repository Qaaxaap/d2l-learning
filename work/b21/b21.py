"""
def gru_step(X, H, params):
    W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_xh, W_hh, b_h = params
    R = torch.tanh(X @ W_xr + H @ W_hr + b_r) 要用sigmoid
    Z = torch.sigmoid(X @ W_xz + H @ W_hz + b_z)
    H_tilde = torch.tanh(X @ W_xh + (R * H) @ W_hh + b_h)
    return (1 - Z) * H + Z * H_tilde Z和1-Z反了，这是torch的不是书上的

def lstm_step(X, H, C, params):
    (W_xi, W_hi, b_i, W_xf, W_hf, b_f,
     W_xo, W_ho, b_o, W_xc, W_hc, b_c) = params
    I = torch.sigmoid(X @ W_xi + H @ W_hi + b_i)
    F = torch.sigmoid(X @ W_xf + H @ W_hf + b_f) 没算O
    C_tilde = torch.tanh(X @ W_xc + H @ W_hc + b_c)
    C_new = F * C + I * C_tilde
    return C_new
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gru_step_book(
    X: torch.Tensor, H: torch.Tensor, params: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """一步 GRU（按书上的约定）。返回新的 H，形状 (batch, h)。

    params 依次是 (W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_xh, W_hh, b_h)。

    权重形状：W_x* 是 (d, h)，W_h* 是 (h, h)，偏置是 (h,)。
    返回新的 H，形状 (batch, h)。
    """
    W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_xh, W_hh, b_h = params
    R = torch.sigmoid(X @ W_xr + H @ W_hr + b_r)
    Z = torch.sigmoid(X @ W_xz + H @ W_hz + b_z)
    H_l = torch.tanh(X @ W_xh + (H * R) @ W_hh + b_h)
    return Z * H + (1 - Z) * H_l


def gru_step_torch(
    X: torch.Tensor, H: torch.Tensor, params: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """一步 GRU（按 torch 的约定）。返回新的 H，形状 (batch, h)。

    params 依次是 (W_ir, W_hr, b_ir, b_hr, W_iz, W_hz, b_iz, b_hz,
                      W_in, W_hn, b_in, b_hn)。

    权重形状 (d, h) 或 (h, h)，偏置 (h,)。
    """
    W_ir, W_hr, b_ir, b_hr, W_iz, W_hz, b_iz, b_hz, W_in, W_hn, b_in, b_hn = params
    R = torch.sigmoid(X @ W_ir + b_ir + H @ W_hr + b_hr)
    Z = torch.sigmoid(X @ W_iz + b_iz + H @ W_hz + b_hz)
    H_l = torch.tanh(X @ W_in + b_in + R * (H @ W_hn + b_hn))
    return (1 - Z) * H_l + Z * H


def lstm_step(
    X: torch.Tensor, H: torch.Tensor, C: torch.Tensor, params: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """一步 LSTM。返回 (H_new, C_new)，两个都是 (batch, h)。

    顺序是 H 在前、C 在后，与 nn.LSTM 的 (h_n, c_n) 一致。

    params 依次是 (W_xi, W_hi, b_i, W_xf, W_hf, b_f,
                      W_xo, W_ho, b_o, W_xc, W_hc, b_c)。

    返回 (H_new, C_new)，两个形状都是 (batch, h)。
    """
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c = params
    I = torch.sigmoid(X @ W_xi + H @ W_hi + b_i)
    F = torch.sigmoid(X @ W_xf + H @ W_hf + b_f)
    O = torch.sigmoid(X @ W_xo + H @ W_ho + b_o)
    C_l = torch.tanh(X @ W_xc + H @ W_hc + b_c)
    C = C * F + C_l * I
    return (O * torch.tanh(C), C)


def hidden_state_shape(
    num_layers: int, batch_size: int, num_hiddens: int, bidirectional: bool = False
) -> tuple[int, int, int]:
    """返回 nn.RNN/nn.GRU/nn.LSTM 的 h_n 形状，只算不建模型。"""
    if not bidirectional:
        return (num_layers, batch_size, num_hiddens)
    else:
        return (2 * num_layers, batch_size, num_hiddens)


def output_feature_dim(num_hiddens: int, bidirectional: bool = False) -> int:
    """返回输出的最后一维。单向是 h，双向是 2h。"""
    if not bidirectional:
        return num_hiddens
    else:
        return 2 * num_hiddens
