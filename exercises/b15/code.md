# B15 代码题

把你的实现写在 `work/b15/b15.py` 里，函数名与签名必须和下面一致。

验收：`just check b15`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章做什么

全是实现运算，不训练。七道题把这一章的六个运算各写一遍：
`corr2d`、输出形状、多输入、多输出、1×1 卷积、池化。最后一道改错。

**性能不重要**——双重 for 循环写得慢是对的，它把运算的含义摆出来了。框架的实现
`nn.Conv2d` 走优化的底层，B16 才用。

验收分三类：**精确**（与书上的例子一致、与 `nn.Conv2d` 一致）、
**范围**（形状在合理范围）、**趋势**（1×1 卷积与全连接等价）。

---

## T1 手写二维互相关

```python
def corr2d(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """二维互相关。X 形状 (n_h, n_w)、K 形状 (k_h, k_w)，都只含一个样本一个通道。

    输出形状 (n_h - k_h + 1, n_w - k_w + 1)。
    不许调用 nn.Conv2d 或 torch.nn.functional.conv2d。
    """
```

| 检查 | 期望 |
|---|---|
| `X=[[0,1,2],[3,4,5],[6,7,8]]`、`K=[[0,1],[2,3]]` | `[[19,25],[37,43]]` |
| 输出形状 | 按公式 |
| 用 `nn.Conv2d` 交叉验证 | 与手写结果一致（同一个核）|
| `X` 与 `K` | 都不被修改 |

## T2 输出形状

```python
def conv_out_shape(n_h: int, n_w: int, k_h: int, k_w: int,
                   p_h: int = 0, p_w: int = 0,
                   s_h: int = 1, s_w: int = 1) -> tuple[int, int]:
    """按公式算出卷积输出的高与宽。

    高：(n_h - k_h + p_h) // s_h + 1
    宽：(n_w - k_w + p_w) // s_w + 1

    **p_h / p_w 按讲义的定义是"两侧一共填充的量"**，不是每侧。
    例如输入 8、核 3、要输出不变，p_h 要传 2（每侧 1）。
    而 nn.Conv2d(padding=1) 的 1 是"每侧"，对应这里的 p_h=2。
    """
```

| 调用 | 期望 |
|---|---|
| `(8, 8, 3, 3)` | `(6, 6)` |
| `(8, 8, 3, 3, 1, 1)` | `(8, 8)` |
| `(8, 8, 3, 3, 1, 1, 2, 2)` | `(4, 4)` |
| `(8, 8, 5, 3, 2, 1)` | `(8, 8)` |
| 与 `nn.Conv2d` 实测对比 | 5 组以上都要一致 |

## T3 多输入通道

```python
def corr2d_multi_in(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """多输入通道的互相关，单输出通道。

    X 形状 (c_i, h, w)、K 形状 (c_i, k_h, k_w)。
    对每个通道各做一次 corr2d，再把结果按元素相加。输出形状 (h', w')。
    """
```

| 检查 | 期望 |
|---|---|
| `X` 与 `K` 都是单通道时 | 与 `corr2d` 结果相同 |
| 双通道的书中例子 | 输出 `[[56, 72], [104, 120]]` |
| 输出维度 | 二维（`dim() == 2`）|

**书中例子的数据**：

```
X = [[[0,1,2],[3,4,5],[6,7,8]],
     [[1,2,3],[4,5,6],[7,8,9]]]
K = [[[0,1],[2,3]],
     [[1,2],[3,4]]]
```

## T4 多输出通道

```python
def corr2d_multi_in_out(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """多输入、多输出通道的互相关。

    X 形状 (c_i, h, w)、K 形状 (c_o, c_i, k_h, k_w)。
    输出形状 (c_o, h', w')。
    """
```

| 检查 | 期望 |
|---|---|
| `K` 只有一个输出通道时 | 与 `corr2d_multi_in` 结果相同 |
| 用 `K`、`K+1`、`K+2` 组成 3 个输出通道 | 0 维与 `corr2d_multi_in(X, K)` 相同 |
| 输出形状 | `(c_o, h', w')` |

## T5 用矩阵乘实现 1×1 卷积

```python
def corr2d_multi_in_out_1x1(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """用 reshape + 矩阵乘实现 1×1 卷积。

    X 形状 (c_i, h, w)、K 形状 (c_o, c_i, 1, 1)。输出 (c_o, h, w)。
    实现方式：把 X 变形成 (c_i, h*w)、K 变形成 (c_o, c_i)，做矩阵乘，再变回来。
    """
```

| 检查 | 期望 |
|---|---|
| 与 `corr2d_multi_in_out` 对比 | 最大差小于 1e-6 |
| 输出形状 | `(c_o, h, w)`（与输入同高宽）|
| 输入 3 通道、输出 2 通道 | 形状 `(2, h, w)` |

## T6 池化

```python
def pool2d(X: torch.Tensor, pool_size: tuple[int, int],
           mode: str = "max") -> torch.Tensor:
    """二维池化。X 形状 (n_h, n_w)，pool_size 是 (p_h, p_w)。

    mode="max" 取窗口最大值、"avg" 取平均。输出形状
    (n_h - p_h + 1, n_w - p_w + 1)。不许调用 nn.MaxPool2d 或 F.max_pool2d。
    """
```

| 检查 | 期望 |
|---|---|
| `X=[[0,1,2],[3,4,5],[6,7,8]]`、`(2,2)`、max | `[[4,5],[7,8]]` |
| 同样输入、avg | `[[2,3],[5,6]]` |
| 与 `nn.MaxPool2d(2)` 交叉验证 | 一致 |
| 非方形窗口 `(2,3)` | 形状按公式 |

## T7 改错

下面这段代码想实现"多输入、多输出通道的卷积"。**两处**问题。

```python
import torch

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

def corr2d_multi_in_out(X, K):
    # X: (c_i, h, w)   K: (c_o, c_i, k_h, k_w)
    return torch.stack([corr2d(X, k) for k in K])

X = torch.randn(3, 8, 8)
K = torch.randn(6, 3, 3, 3)
Y = corr2d_multi_in_out(X, K)
print(Y.shape)      # 期望 (6, 6, 6)
```

两处分别指两个位置：**多输入通道怎么处理**、**对 `corr2d` 的调用方式**。

两段提示：第一处漏掉了"各通道结果相加"这一步，而且 `corr2d` 只吃二维输入；
修好这一处之后，第二处的症状会变——从"形状对不上"变成"结果少了一维"。

**改完应当输出 `(6, 6, 6)`。**
