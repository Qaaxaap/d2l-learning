"""B08 参考答案。做完再看。

跑一遍：
    python3 solutions/b08/b08.py
或者拿它验断言：
    python3 tools/check_b08.py solutions/b08/b08.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------- T1 激活函数 ----------

def relu(x: torch.Tensor) -> torch.Tensor:
    """逐元素取 x 与 0 的较大值。"""
    return torch.maximum(x, torch.zeros_like(x))


def relu_grad(x: torch.Tensor) -> torch.Tensor:
    """ReLU 的导数：正为 1，其余为 0。"""
    return (x > 0).float()


# ---------- T2 参数与模型 ----------

def init_params(num_inputs: int = 784, num_hiddens: int = 256,
                num_outputs: int = 10, seed: int = 0) -> tuple:
    g = torch.Generator().manual_seed(seed)
    W1 = (torch.randn(num_inputs, num_hiddens, generator=g) * 0.01).requires_grad_(True)
    b1 = torch.zeros(num_hiddens, requires_grad=True)
    W2 = (torch.randn(num_hiddens, num_outputs, generator=g) * 0.01).requires_grad_(True)
    b2 = torch.zeros(num_outputs, requires_grad=True)
    return W1, b1, W2, b2


def mlp_forward(X: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor,
                W2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """拉平、隐藏层过 ReLU、输出层。返回 logits。"""
    X = X.reshape(X.shape[0], -1)
    H = relu(X @ W1 + b1)
    return H @ W2 + b2


# ---------- T3 参数量 ----------

def count_params(net) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


# ---------- 从 B07 抄过来的零件 ----------

def cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """返回这一批的交叉熵之和（不取平均），与 B07 的约定一致。"""
    m = logits.max(dim=-1, keepdim=True).values
    shifted = logits - m
    log_sum_exp = shifted.exp().sum(dim=-1, keepdim=True).log() + m
    log_probs = logits - log_sum_exp
    n = logits.shape[0]
    return -log_probs[torch.arange(n), y].sum()


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def evaluate_accuracy(net, loader) -> float:
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X = X.reshape(X.shape[0], -1)
            correct += (net(X).argmax(dim=1) == y).sum().item()
            total += y.numel()
    return correct / total


def sgd(params: list, lr: float, batch_size: int) -> None:
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad / batch_size
            p.grad.zero_()


# ---------- T4 从零实现 ----------

def train_from_scratch(train_iter, test_iter, num_epochs: int = 10,
                       lr: float = 0.1, batch_size: int = 256,
                       seed: int = 0) -> dict:
    W1, b1, W2, b2 = init_params(seed=seed)
    net = lambda X: mlp_forward(X, W1, b1, W2, b2)  # noqa: E731

    train_loss, train_acc, test_acc = [], [], []
    for _ in range(num_epochs):
        total_loss, n_samples = 0.0, 0
        for X, y in train_iter:
            X = X.reshape(X.shape[0], -1)
            l = cross_entropy(net(X), y)
            l.backward()
            sgd([W1, b1, W2, b2], lr, batch_size)
            # 用 item() 转成浮点再累加，避免把每批的计算图都留着
            total_loss += l.item()
            n_samples += y.numel()
        train_loss.append(total_loss / n_samples)
        train_acc.append(evaluate_accuracy(net, train_iter))
        test_acc.append(evaluate_accuracy(net, test_iter))
    return {"train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc}


# ---------- T5 简洁实现 ----------

def make_net(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


def train_concise(train_iter, test_iter, num_epochs: int = 10,
                  lr: float = 0.1, seed: int = 0) -> dict:
    net = make_net(seed=seed)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    train_loss, train_acc, test_acc = [], [], []
    for _ in range(num_epochs):
        total_loss, n_samples = 0.0, 0
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss_fn(net(X), y)
            l.backward()
            optimizer.step()
            # nn.CrossEntropyLoss 默认返回这一批的平均损失，所以要乘回样本数
            # 才能与 T4（返回和）用同一种方式算 epoch 平均
            total_loss += l.item() * y.numel()
            n_samples += y.numel()
        train_loss.append(total_loss / n_samples)
        train_acc.append(evaluate_accuracy(net, train_iter))
        test_acc.append(evaluate_accuracy(net, test_iter))
    return {"train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc}


"""T6 改错

第 1 处：两个 `nn.Linear` 之间没有激活函数
    `nn.Linear(784, 256)` 后面直接接 `nn.Linear(256, 10)`，两层线性叠在一起
    等价于一个线性层（讲义 1.2 节的联立推导），模型容量与单层线性模型相同。
    不报错，只是精度上不去。
    改法：在两个 Linear 之间加 `nn.ReLU()`。

第 2 处：`nn.ReLU()` 放在最后一层之后
    它在输出层之后，会把 logits 里的负数截成 0。这些 logits 要送进
    `CrossEntropyLoss`，被截断之后类别之间的相对大小关系就变了。
    改法：把 `nn.ReLU()` 移到两个 Linear 之间，输出层保持原样。

第 3 处：参数用 `p.zero_()` 初始化
    全零初始化让每个隐藏单元的前向输出与梯度完全相同，反向传播后它们仍然相同，
    等于隐藏层只有 1 个单元在起作用。这叫对称性问题，不报错，只是精度停在低位。
    改法：权重用小的随机数（例如均值 0、标准差 0.01 的正态分布），偏置可以留全零。

修好之后模型是：
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
"""


if __name__ == "__main__":
    tf = transforms.ToTensor()
    tr = datasets.FashionMNIST(root="data", train=True, download=False, transform=tf)
    te = datasets.FashionMNIST(root="data", train=False, download=False, transform=tf)
    train_iter = DataLoader(tr, batch_size=256, shuffle=True)
    test_iter = DataLoader(te, batch_size=256)

    print("relu：", relu(torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])).tolist())
    net = make_net()
    print("MLP 参数量：", count_params(net), " 单层：", count_params(nn.Linear(784, 10)))
    res = train_concise(train_iter, test_iter, num_epochs=2)
    print("简洁实现 2 轮的测试精度：", [round(v, 4) for v in res["test_acc"]])
