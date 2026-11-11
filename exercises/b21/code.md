# B21 代码题

把你的实现写在 `work/b21/b21.py` 里，函数名与签名必须和下面一致。

验收：`just check b21`。断言全过，并且我读过代码之后，这一题才算过。

---

## 需要的导入

```python
import torch
import torch.nn as nn
```

`params` 是 `tuple[torch.Tensor, ...]`，元组里每个元素是一个权重或偏置。

## 先看一段：这一章做什么

手写 GRU 与 LSTM 的单步计算，并**用两种 GRU 约定分别与 `nn.GRU` 对比**。

这一章不训练。重点是**门的公式与约定**——写错一个系数不会报错，但训练结果完全不同。

---

## T1 按书上的公式实现 GRU

```python
def gru_step_book(X: torch.Tensor, H: torch.Tensor,
                  params: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """一步 GRU（按书上的约定）。返回新的 H，形状 (batch, h)。

    params 依次是 (W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_xh, W_hh, b_h)。

    权重形状：W_x* 是 (d, h)，W_h* 是 (h, h)，偏置是 (h,)。
    返回新的 H，形状 (batch, h)。
    """
```

公式见讲义 2.1 节。**`Z` 是保留旧状态的比例**：

```
H = Z * H_prev + (1 - Z) * H_tilde
```

| 检查 | 期望 |
|---|---|
| 输出形状 | `(batch, h)` |
| 与手算一致 | 是 |
| 让更新门饱和到 1 | 输出接近旧的 `H` |

## T2 按 torch 的约定实现 GRU

```python
def gru_step_torch(X: torch.Tensor, H: torch.Tensor,
                   params: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """一步 GRU（按 torch 的约定）。返回新的 H，形状 (batch, h)。

    params 依次是 (W_ir, W_hr, b_ir, b_hr, W_iz, W_hz, b_iz, b_hz,
                      W_in, W_hn, b_in, b_hn)。

    权重形状 (d, h) 或 (h, h)，偏置 (h,)。
    """
```

**torch 的约定**（讲义第 3 节）：

```
r = sigmoid(X @ W_ir + b_ir + H @ W_hr + b_hr)
z = sigmoid(X @ W_iz + b_iz + H @ W_hz + b_hz)      ← z 是接受新状态的比例
n = tanh(X @ W_in + b_in + r * (H @ W_hn + b_hn))   ← 先矩阵乘、再乘门
H_new = (1 - z) * n + z * H_prev
```

| 检查 | 期望 |
|---|---|
| 与 `nn.GRU` 单步输出 | 一致 |

**`nn.GRU` 的权重布局**：`weight_ih_l0` 形状 `(3h, d)`，三块依次是 `r`、`z`、`n`。
断言自己做切分与转置，你只需要按上面的公式写。

## T3 LSTM

```python
def lstm_step(X: torch.Tensor, H: torch.Tensor, C: torch.Tensor,
              params: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """一步 LSTM。返回 (H_new, C_new)，两个都是 (batch, h)。

    顺序是 H 在前、C 在后，与 nn.LSTM 的 (h_n, c_n) 一致。

    params 依次是 (W_xi, W_hi, b_i, W_xf, W_hf, b_f,
                      W_xo, W_ho, b_o, W_xc, W_hc, b_c)。

    返回 (H_new, C_new)，两个形状都是 (batch, h)。
    """
```

| 检查 | 期望 |
|---|---|
| `C_new` 与手算 | `C_new = F*C + I*C_tilde` |
| `H_new` 与手算 | `H_new = O * tanh(C_new)` |
| 与 `nn.LSTM` | 一致 |

## T4 深层与双向的形状

```python
def hidden_state_shape(num_layers: int, batch_size: int, num_hiddens: int,
                       bidirectional: bool = False) -> tuple[int, int, int]:
    """返回 nn.RNN/nn.GRU/nn.LSTM 的 h_n 形状，只算不建模型。"""


def output_feature_dim(num_hiddens: int, bidirectional: bool = False) -> int:
    """返回输出的最后一维。单向是 h，双向是 2h。"""
```

| 检查 | 期望 |
|---|---|
| `hidden_state_shape(3, 8, 16)` | `(3, 8, 16)` |
| `hidden_state_shape(3, 8, 16, True)` | `(6, 8, 16)` |
| `output_feature_dim(16)` | 16 |
| `output_feature_dim(16, True)` | 32 |
| 与真实 `nn.RNN` 核对 | 一致 |

## T5 改错

下面这段想实现书上的 GRU。**三处**问题。

```python
def gru_step(X, H, params):
    W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_xh, W_hh, b_h = params
    R = torch.tanh(X @ W_xr + H @ W_hr + b_r)
    Z = torch.sigmoid(X @ W_xz + H @ W_hz + b_z)
    H_tilde = torch.tanh(X @ W_xh + (R * H) @ W_hh + b_h)
    return (1 - Z) * H + Z * H_tilde

def lstm_step(X, H, C, params):
    (W_xi, W_hi, b_i, W_xf, W_hf, b_f,
     W_xo, W_ho, b_o, W_xc, W_hc, b_c) = params
    I = torch.sigmoid(X @ W_xi + H @ W_hi + b_i)
    F = torch.sigmoid(X @ W_xf + H @ W_hf + b_f)
    C_tilde = torch.tanh(X @ W_xc + H @ W_hc + b_c)
    C_new = F * C + I * C_tilde
    return C_new
```

三处分别指：**GRU 门的激活**、**GRU 隐藏状态的系数**、**LSTM 少算了什么**。

三段提示：第一处让门有了负值；第二处与书上的约定相反；第三处让输出值域不受控。
