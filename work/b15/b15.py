"""
import torch

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y
看起来corr2d是针对单输入单输出的，但下面函数却将其当做了多输入单输出
需要补一个多输入单输出的
def corr2d_multi_in_out(X, K):
    # X: (c_i, h, w)   K: (c_o, c_i, k_h, k_w)
    return torch.stack([corr2d(X, k) for k in K])

X = torch.randn(3, 8, 8)
K = torch.randn(6, 3, 3, 3)
Y = corr2d_multi_in_out(X, K)
print(Y.shape)      # 期望 (6, 6, 6)
"""
import torch

def corr2d(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """二维互相关。X 形状 (n_h, n_w)、K 形状 (k_h, k_w)，都只含一个样本一个通道。

    输出形状 (n_h - k_h + 1, n_w - k_w + 1)。
    不许调用 nn.Conv2d 或 torch.nn.functional.conv2d。
    """
    n_h, k_h, n_w, k_w = X.shape[0], K.shape[0], X.shape[1], K.shape[1]
    ret = torch.zeros(n_h - k_h + 1, n_w - k_w + 1)
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            for x in range(k_h):
                for y in range(k_w):
                    ret[i][j] += X[i + x][j + y] * K[x][y]
    return ret

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
    return ((n_h - k_h + p_h) // s_h + 1, (n_w - k_w + p_w) // s_w + 1)

def corr2d_multi_in(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """多输入通道的互相关，单输出通道。

    X 形状 (c_i, h, w)、K 形状 (c_i, k_h, k_w)。
    对每个通道各做一次 corr2d，再把结果按元素相加。输出形状 (h', w')。
    """
    sum = torch.zeros(1)
    for i in range(X.shape[0]):
        sum = sum + corr2d(X[i], K[i])
    return sum

def corr2d_multi_in_out(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """多输入、多输出通道的互相关。

    X 形状 (c_i, h, w)、K 形状 (c_o, c_i, k_h, k_w)。
    输出形状 (c_o, h', w')。
    """
    chs = []
    for i in range(K.shape[0]):
        chs.append(corr2d_multi_in(X, K[i]))
    return torch.stack(chs)

def corr2d_multi_in_out_1x1(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """用 reshape + 矩阵乘实现 1×1 卷积。

    X 形状 (c_i, h, w)、K 形状 (c_o, c_i, 1, 1)。输出 (c_o, h, w)。
    实现方式：把 X 变形成 (c_i, h*w)、K 变形成 (c_o, c_i)，做矩阵乘，再变回来。
    """
    x = X.reshape(X.shape[0], -1)
    k = K.reshape(K.shape[0], -1)
    return (k @ x).reshape(-1, X.shape[1], X.shape[2])

def pool2d(X: torch.Tensor, pool_size: tuple[int, int],
           mode: str = "max") -> torch.Tensor:
    """二维池化。X 形状 (n_h, n_w)，pool_size 是 (p_h, p_w)。

    mode="max" 取窗口最大值、"avg" 取平均。输出形状
    (n_h - p_h + 1, n_w - p_w + 1)。不许调用 nn.MaxPool2d 或 F.max_pool2d。
    """
    n_h, n_w = X.shape[0], X.shape[1]
    p_h, p_w = pool_size
    out = torch.zeros(n_h - p_h + 1, n_w - p_w + 1)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            sum = 0
            mx = -1e9
            for x in range(p_h):
                for y in range(p_w):
                    sum += X[i + x][j + y]
                    mx = max(mx, X[i + x][j + y])
            if mode == "avg":
                out[i][j] = sum / (p_w * p_h)
            if mode == "max":
                out[i][j] = mx
    return out
