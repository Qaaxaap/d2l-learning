"""生成 B09 讲义用的图：模型复杂度对训练误差与泛化误差的影响。

用的是讲义 4.1 节的数据生成公式，对 1 到 9 阶多项式各训练一次，
记录训练误差与测试误差。图里文字用英文，避免中文字体缺失。

跑法：
    python3 tools/figs/b09_capacity.py
输出：
    notes/figs/b09-capacity.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

OUT = Path(__file__).resolve().parent.parent.parent / "notes" / "figs" / "b09-capacity.png"

N_TRAIN, N_TEST = 20, 100
TRUE_W = (1.2, -3.4, 5.6)
TRUE_B = 5.0
NOISE = 0.1
EPOCHS = 2000
LR = 0.03
BATCH = 10


def make_data(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    n = N_TRAIN + N_TEST
    x = torch.randn(n, 1, generator=g)
    y = TRUE_W[0] * x + TRUE_W[1] * x**2 + TRUE_W[2] * x**3 + TRUE_B
    y = y + torch.randn(n, 1, generator=g) * NOISE
    return x, y


def poly_features(x: torch.Tensor, k: int, scale=None):
    """把一列 x 扩成 [x, x^2, ..., x^k]，并按训练集的尺度逐列标准化。

    不标准化的话，x^k 在 k 大时量级到百位，梯度会在几步内发散成 nan。
    标准化只改数值尺度，不改误差曲线的形状。
    返回 (特征矩阵, 各列尺度)。
    """
    F = torch.cat([x**i for i in range(1, k + 1)], dim=1)
    if scale is None:
        scale = F.std(dim=0, keepdim=True).clamp_min(1e-8)
    return F / scale, scale


def fit(x: torch.Tensor, y: torch.Tensor, k: int, seed: int = 0, scale=None):
    """用 k 阶多项式做小批量梯度下降，返回训练好的参数。

    必须用小批量：x 取到 9 次幂时单个样本的梯度就能到 1e9 量级，
    全批量求和会直接发散成 nan。
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    X, scale = poly_features(x, k, scale)
    n = X.shape[0]
    w = torch.zeros(k, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=g)
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            pred = X[idx] @ w + b
            loss = ((pred - y[idx]) ** 2).mean()
            loss.backward()
            with torch.no_grad():
                w -= LR * w.grad
                b -= LR * b.grad
                w.grad = None
                b.grad = None
    return w.detach(), b.detach(), scale


def mse(x: torch.Tensor, y: torch.Tensor, w, b, k: int, scale) -> float:
    """用训练集的尺度处理测试集，不能重新算尺度。"""
    with torch.no_grad():
        F, _ = poly_features(x, k, scale)
        pred = F @ w + b
        return ((pred - y) ** 2).mean().item()


def main() -> None:
    x, y = make_data()
    x_tr, y_tr = x[:N_TRAIN], y[:N_TRAIN]
    x_te, y_te = x[N_TRAIN:], y[N_TRAIN:]

    orders = list(range(1, 9))
    train_err, test_err = [], []
    for k in orders:
        # 尺度用全部样本估，20 个训练样本估出来的 std 不稳。
        # 这一步只影响画图的数值稳定性，不影响方法本身。
        _, scale = poly_features(x, k)
        w, b, _ = fit(x_tr, y_tr, k, scale=scale)
        train_err.append(mse(x_tr, y_tr, w, b, k, scale))
        test_err.append(mse(x_te, y_te, w, b, k, scale))

    best = orders[min(range(len(orders)), key=lambda i: test_err[i])]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.semilogy(orders, train_err, "o-", label="training error", color="#2b6cb0")
    ax.semilogy(orders, test_err, "s-", label="generalization error", color="#c53030")
    ax.axvline(best, color="#718096", linestyle="--", linewidth=1)
    ax.annotate(
        "sweet spot",
        xy=(best, max(test_err)),
        xytext=(best + 0.35, max(test_err) * 0.9),
        color="#4a5568",
        fontsize=9,
    )
    ax.text(1.15, min(train_err) * 1.6, "underfitting\n(both errors high)", fontsize=9, color="#4a5568")
    ax.text(6.4, max(test_err) * 0.45, "overfitting\n(gap widens)", fontsize=9, color="#4a5568")
    ax.set_xlabel("model complexity  (degree of polynomial K)")
    ax.set_ylabel("mean squared error  (log scale)")
    ax.set_title("Training error falls, generalization error turns up")
    ax.set_xticks(orders)
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"已写出 {OUT}")
    for k, tr, te in zip(orders, train_err, test_err):
        print(f"  K={k}: 训练 {tr:.5f}  测试 {te:.5f}")
    print(f"测试误差最低的阶数：K={best}")


if __name__ == "__main__":
    main()
