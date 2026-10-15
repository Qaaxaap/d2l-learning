"""
import matplotlib.pyplot as plt
import torch

w = torch.zeros(3, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
train_curve, test_curve = [], []

for epoch in range(100):
    for X, y in train_iter:
        pred = X @ w + b 梯度没有清零产生了累加
        loss = ((pred - y) ** 2).sum()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
    train_curve.append(loss.item()) 要记录平均，但这里记录的是和
    test_curve.append(((X_test @ w + b - y_test) ** 2).sum())

plt.plot(train_curve, test_curve)没有使用对数y坐标
plt.show()没有保存
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def make_data(n_train: int = 100, n_test: int = 100,
              seed: int = 0) -> tuple:
    """按上面的公式生成数据。

    返回 (x_train, y_train, x_test, y_test)，四个都是形状 (n, 1) 的 float32 张量。
    x 取标准正态随机数，噪声标准差 0.1。同一个 seed 给出相同结果。
    训练集在前、测试集在后也可以，但四个返回值的样本数要和参数对上。
    """
    torch.manual_seed(seed)
    x_train = torch.normal(mean=0, std=1, size=(n_train, 1))
    x_test = torch.normal(mean=0, std=1, size=(n_test, 1))
    eps_train = torch.normal(mean=0, std=0.1, size=(n_train, 1))
    eps_test = torch.normal(mean=0, std=0.1, size=(n_test, 1))
    y_train = 1.2 * x_train - 3.4 * (x_train ** 2) + 5.6 * (x_train ** 3) + 5 + eps_train
    y_test = 1.2 * x_test - 3.4 * (x_test ** 2) + 5.6 * (x_test ** 3) + 5 + eps_test
    return (x_train, y_train, x_test, y_test)
    
def poly_features(x: torch.Tensor, k: int) -> torch.Tensor:
    """把形状 (n, 1) 的 x 扩成 (n, k) 的 [x, x^2, ..., x^k]。"""
    p = torch.arange(1, k + 1, device=x.device, dtype=x.dtype)
    return x ** p

def train_poly(x_train: torch.Tensor, y_train: torch.Tensor,
               x_test: torch.Tensor, y_test: torch.Tensor, k: int,
               num_epochs: int = 100, lr: float = 0.01,
               batch_size: int = 10, seed: int = 0) -> dict:
    """用 k 阶多项式做小批量梯度下降。

    返回 {"train_loss": [...], "test_loss": [...]}，两个列表长度都是 num_epochs，
    元素是那一轮结束时的**平均**平方损失（Python 浮点数）。

    模型就是 y = X @ w + b，w 形状 (k, 1)、b 形状 (1,)。
    参数初始化为全零（这个实验里可以，因为不存在对称性问题）。
    损失用平方损失，训练时可以返回"和"，但记录进列表的必须是平均值。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = nn.Sequential(nn.Linear(k, 1)).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    d = {}
    d["train_loss"] = []
    d["test_loss"] = []
    X_train = poly_features(x_train, k).to(device)
    y_train = y_train.to(device)
    X_test = poly_features(x_test, k).to(device)
    y_test = y_test.to(device)
    n = X_train.size(0)
    for _ in range(num_epochs):
        total_loss = 0
        for i in range(0, n, batch_size):
            r = min(n - 1, i + batch_size)
            x, y = X_train[i:r], y_train[i:r]
            optimizer.zero_grad()
            l = loss_fn(net(x), y)
            l.backward()
            optimizer.step()
            total_loss += l.item() * y.size(0)
        d["train_loss"].append(total_loss / n)
        with torch.no_grad():
            total_loss = loss_fn(net(X_test), y_test).item()
        d["test_loss"].append(total_loss)
    return d
            
def plot_errors(train_loss: list, test_loss: list, path: str) -> None:
    """把两条误差曲线画在一张图上并存到 path。

    要求：横轴是轮数，纵轴用**对数刻度**，两条曲线要有图例区分，
    另存为文件（不要 plt.show()）。
    """
    plt.figure(figsize=(7, 4))
    x = range(1, len(train_loss) + 1)
    plt.semilogy(x, train_loss, label="train loss")
    plt.semilogy(x, test_loss, label="test loss")
    plt.xlabel("epoch")
    plt.ylabel("mean loss(log scale)")
    plt.legend()
    plt.savefig(fname=path, dpi=200)

def k_fold_indices(n: int, k: int, seed: int = 0) -> list:
    """把 0 到 n-1 这 n 个下标切成 k 份。返回一个长度 k 的列表，
    每一项是一个 Python 整数列表。

    要求：k 份合起来正好是 0 到 n-1 各一次，不重不漏；各份大小相差不超过 1。
    """
    l = []
    last = 0
    batch_size = n // k
    big_batch = n - batch_size * k
    small_batch = k - big_batch
    for _ in range(small_batch):
        l.append(list(range(last, last + batch_size)))
        last += batch_size
    for _ in range(big_batch):
        l.append(list(range(last, last + batch_size + 1)))
        last += batch_size + 1
    return l

def diagnose(train_loss: list, test_loss: list) -> str:
    """给定两条曲线，返回 "underfitting"、"overfitting" 或 "good"。

    判据按顺序：
      1. 最后一轮训练损失 > 0.5           -> "underfitting"
      2. 最后一轮测试损失 > 训练损失 * 10  -> "overfitting"
      3. 其余                              -> "good"
    """
    if (train_loss[-1] > 0.5):
        return "underfitting"
    elif (test_loss[-1] > train_loss[-1] * 10):
        return "overfitting"
    else:
        return "good"