"""B07 参考答案。做完再看。

跑一遍：
    python3 solutions/b07/b07.py
或者拿它验断言：
    python3 tools/check_b07.py solutions/b07/b07.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def softmax(o: torch.Tensor) -> torch.Tensor:
    """按最后一维做 softmax。

    先减去该维最大值再取指数：exp 里的数最大是 0，不会上溢。
    这个平移是恒等变形（分子分母同乘 exp(-m)），不改变结果。
    """
    m = o.max(dim=-1, keepdim=True).values
    e = torch.exp(o - m)
    return e / e.sum(dim=-1, keepdim=True)


def cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """交叉熵，输入 logits 与整数标签。

    log softmax 手写：
        log_softmax(o)_j = o_j - logsumexp(o)
        logsumexp(o) = m + log(sum(exp(o - m))),  m = max_j o_j
    取负对数似然后只挑正确类别那一项。
    """
    m = logits.max(dim=-1, keepdim=True).values
    shifted = logits - m
    log_sum_exp = shifted.exp().sum(dim=-1, keepdim=True).log() + m
    log_probs = logits - log_sum_exp
    n = logits.shape[0]
    picked = log_probs[torch.arange(n), y]
    # 返回和，不取平均。除以批量大小那一步在 sgd 里做，与 B06 的约定一致。
    return -picked.sum()


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """命中率。softmax 保序，所以直接对 logits 取 argmax 与对概率取一样。"""
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def evaluate_accuracy(net, data_iter) -> float:
    """整个数据集上的准确率，不建图。"""
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X = X.reshape(X.shape[0], -1)
            pred = net(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def make_net(num_inputs: int = 784, num_outputs: int = 10, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(num_inputs, num_outputs, generator=g) * 0.01).requires_grad_(True)
    b = torch.zeros(num_outputs, requires_grad=True)
    return W, b


def sgd(params, lr, batch_size):
    """B06 里写的那个，这里再用一次。"""
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad / batch_size
            p.grad.zero_()


def train_from_scratch(train_iter, test_iter, num_epochs: int = 5,
                       lr: float = 0.1, batch_size: int = 256,
                       seed: int = 0) -> dict:
    W, b = make_net(seed=seed)
    net = lambda X: X @ W + b  # noqa: E731

    train_loss, train_acc, test_acc = [], [], []
    for _ in range(num_epochs):
        total_loss, n_samples = 0.0, 0
        for X, y in train_iter:
            X = X.reshape(X.shape[0], -1)
            l = cross_entropy(net(X), y)
            l.backward()
            sgd([W, b], lr, batch_size)
            total_loss += l.item() * y.numel()
            n_samples += y.numel()
        train_loss.append(total_loss / n_samples)
        train_acc.append(evaluate_accuracy(net, train_iter))
        test_acc.append(evaluate_accuracy(net, test_iter))
    return {"train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc}


def confusion_matrix(net, data_iter, q: int = 10) -> torch.Tensor:
    cm = torch.zeros(q, q, dtype=torch.long)
    with torch.no_grad():
        for X, y in data_iter:
            X = X.reshape(X.shape[0], -1)
            pred = net(X).argmax(dim=1)
            for t, p in zip(y.tolist(), pred.tolist()):
                cm[t, p] += 1
    return cm


def train_concise(train_iter, test_iter, num_epochs: int = 5,
                  lr: float = 0.1, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(784, 10))
    loss_fn = nn.CrossEntropyLoss()      # 接收 logits，内部做 log_softmax
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    train_loss, train_acc, test_acc = [], [], []
    for _ in range(num_epochs):
        total_loss, n_samples = 0.0, 0
        for X, y in train_iter:
            X = X.reshape(X.shape[0], -1)
            l = loss_fn(net(X), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            total_loss += l.item() * y.numel()
            n_samples += y.numel()
        train_loss.append(total_loss / n_samples)
        train_acc.append(evaluate_accuracy(net, train_iter))
        test_acc.append(evaluate_accuracy(net, test_iter))
    return {"train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc}


def load_data(root="data", n_train=None, n_test=None, batch_size=256, seed=0):
    """读 Fashion-MNIST。n_train / n_test 给定时只取前若干个样本（断言里用来加速）。"""
    tf = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root=root, train=True, download=True, transform=tf)
    test_ds = datasets.FashionMNIST(root=root, train=False, download=True, transform=tf)
    if n_train:
        train_ds = torch.utils.data.Subset(train_ds, range(n_train))
    if n_test:
        test_ds = torch.utils.data.Subset(test_ds, range(n_test))
    g = torch.Generator().manual_seed(seed)
    train_iter = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_iter = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter


"""T7 改错

第 1 处：`out = torch.softmax(net(X), dim=1)`
    `nn.CrossEntropyLoss` 内部已经做了 log_softmax，它要的是 logits。
    这里先 softmax 再传进去，等于做了两次 softmax，梯度被压低，损失降得很慢。
    而且 softmax 之后再取 log，数值精度也差一截。
    改法：直接把 `net(X)` 传给损失函数，删掉这一行的 softmax。

第 2 处：整个循环里没有 `optimizer.zero_grad()`
    torch 的 `.grad` 是累加的，不清零会把上一批的梯度一起算进来，
    等效学习率随步数增长。症状是损失震荡或发散，不报错。
    改法：在 `l.backward()` 之前加一句 `optimizer.zero_grad()`。

第 3 处：`optimizer.step()` 写在 `l.backward()` 之前
    更新参数要用刚算出来的梯度，而反传还没发生。第一次调用时参数的 `.grad` 是 None，
    torch 的 SGD 会跳过这个参数；之后每次用的都是上一批的旧梯度，
    相当于优化滞后一步，收敛变慢且不稳定。
    改法：把两句的顺序换过来，`l.backward()` 在前、`optimizer.step()` 在后。

修好之后循环体是这四句，顺序固定：
    optimizer.zero_grad()
    l = loss_fn(net(X), y)
    l.backward()
    optimizer.step()
"""


if __name__ == "__main__":
    train_iter, test_iter = load_data(n_train=10000, n_test=2000)
    print("softmax 数值稳定性：", softmax(torch.tensor([[1000.0, 0.0]])).tolist())
    print("交叉熵（单样本）：", round(cross_entropy(
        torch.tensor([[2.0, 1.0, 0.1]]), torch.tensor([0])).item(), 6))
    res = train_from_scratch(train_iter, test_iter, num_epochs=2)
    print("从零实现 2 轮的测试精度：", [round(v, 4) for v in res["test_acc"]])
    cm = confusion_matrix(lambda X: X @ make_net()[0] + make_net()[1], test_iter)
    print("混淆矩阵形状：", tuple(cm.shape), "元素和：", cm.sum().item())
