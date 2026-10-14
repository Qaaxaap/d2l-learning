"""B09 参考答案。做完再看。

跑一遍：
    python3 solutions/b09/b09.py
或者拿它验断言：
    python3 tools/check_b09.py solutions/b09/b09.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


# ---------- T1 生成数据 ----------

def make_data(n_train: int = 100, n_test: int = 100, seed: int = 0) -> tuple:
    """y = 1.2x - 3.4x^2 + 5.6x^3 + 5 + 噪声，x 与噪声都取标准正态。"""
    g = torch.Generator().manual_seed(seed)
    n = n_train + n_test
    x = torch.randn(n, 1, generator=g)
    noise = torch.randn(n, 1, generator=g) * 0.1
    y = 1.2 * x - 3.4 * x**2 + 5.6 * x**3 + 5.0 + noise
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


# ---------- T2 多项式特征 ----------

def poly_features(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.cat([x**i for i in range(1, k + 1)], dim=1)


# ---------- 从 B06 抄来的零件 ----------

def squared_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """返回这一批的平方误差之和，与 B06 的约定一致。"""
    return ((pred - y) ** 2).sum()


def sgd(params: list, lr: float, batch_size: int) -> None:
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad / batch_size
            p.grad = None


def mean_loss(pred: torch.Tensor, y: torch.Tensor) -> float:
    """记录曲线时用的平均损失。"""
    with torch.no_grad():
        return ((pred - y) ** 2).mean().item()


# ---------- T3 训练并记录 ----------

def train_poly(x_train: torch.Tensor, y_train: torch.Tensor,
               x_test: torch.Tensor, y_test: torch.Tensor, k: int,
               num_epochs: int = 100, lr: float = 0.01,
               batch_size: int = 10, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    X_tr = poly_features(x_train, k)
    X_te = poly_features(x_test, k)

    # 全零初始化：这个实验里可以，因为输入各列不同，不存在对称性问题
    w = torch.zeros(k, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    n = X_tr.shape[0]

    train_loss, test_loss = [], []
    for _ in range(num_epochs):
        order = torch.randperm(n, generator=g)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            loss = squared_loss(X_tr[idx] @ w + b, y_train[idx])
            loss.backward()
            sgd([w, b], lr, idx.numel())
        # 记录时换成平均损失，否则与测试损失差一个批量大小的倍数
        train_loss.append(mean_loss(X_tr @ w + b, y_train))
        test_loss.append(mean_loss(X_te @ w + b, y_test))
    return {"train_loss": train_loss, "test_loss": test_loss}


# ---------- T4 画图 ----------

def plot_errors(train_loss: list, test_loss: list, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(train_loss) + 1)
    ax.semilogy(epochs, train_loss, label="training error")
    ax.semilogy(epochs, test_loss, label="generalization error")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared error (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------- T5 k 折切分 ----------

def k_fold_indices(n: int, k: int, seed: int = 0) -> list:
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g).tolist()
    base, extra = divmod(n, k)
    parts, start = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        parts.append(sorted(order[start:start + size]))
        start += size
    return parts


# ---------- T6 诊断 ----------

def diagnose(train_loss: list, test_loss: list) -> str:
    if train_loss[-1] > 0.5:
        return "underfitting"
    if test_loss[-1] > train_loss[-1] * 10:
        return "overfitting"
    return "good"


"""T7 改错

第 1 处：`loss = ((pred - y) ** 2).sum()` 记录进曲线
    训练时返回"和"没问题，但 `train_curve.append(loss.item())` 记的是这个和，
    而测试那条是另一个量级，两条曲线没法直接比。差的是一个批量大小的倍数。
    改法：记录时换成平均，`(loss / X.shape[0]).item()`；
    测试那条同样换成平均。

第 2 处：更新参数前梯度没有清零
    `loss.backward()` 会把这一批的梯度**累加**到 `.grad` 上（B04 的 4.7 节），
    循环里从不清零，梯度会越积越大，训练很快发散。
    改法：在 `with torch.no_grad()` 里更新完顺手 `w.grad = None`、`b.grad = None`，
    或者在 `backward()` 之前清零。

第 3 处：画图那两行
    `plt.plot(train_curve, test_curve)` 把训练误差当成了横轴、测试误差当成了纵轴，
    画出来的是两条误差的关系图，不是随轮数的变化。而且 `plt.show()` 只把图显示出来，
    不落盘，脚本跑完什么也没留下。
    改法：`plt.plot(range(len(train_curve)), train_curve, label="train")`，
    测试那条同样传轮数；最后 `plt.savefig(path)`。
    另外纵轴该用对数刻度，`plt.semilogy`。
"""


if __name__ == "__main__":
    x_tr, y_tr, x_te, y_te = make_data()
    for k in (1, 3, 8):
        res = train_poly(x_tr, y_tr, x_te, y_te, k, num_epochs=100)
        print(
            f"k={k}: 训练 {res['train_loss'][-1]:.4f}  测试 {res['test_loss'][-1]:.4f}  "
            f"-> {diagnose(res['train_loss'], res['test_loss'])}"
        )
    print("k 折切分（10 个下标切 3 份）：", k_fold_indices(10, 3))
