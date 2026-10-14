import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def relu(x: torch.Tensor) -> torch.Tensor:
    """逐元素取 x 与 0 的较大值。

    不许用 torch.relu、torch.nn.functional.relu 或 nn.ReLU。
    """
    return x.maximum(torch.zeros_like(x, dtype=torch.float32))

def relu_grad(x: torch.Tensor) -> torch.Tensor:
    """ReLU 的导数，逐元素。

    x > 0 处为 1，x < 0 处为 0，x == 0 处取 0。
    """
    return torch.where(x > torch.zeros_like(x), 1., 0.)

def init_params(num_inputs: int = 784, num_hiddens: int = 256,
                num_outputs: int = 10, seed: int = 0) -> tuple:
    """返回 (W1, b1, W2, b2)。

    W1 形状 (784, 256)，b1 形状 (256,)，W2 形状 (256, 10)，b2 形状 (10,)。
    两个 W 用均值 0、标准差 0.01 的正态随机数，两个 b 全零。
    四个都要开 requires_grad。同一个 seed 给出相同结果。
    """
    torch.manual_seed(seed)
    W1 = torch.normal(std=0.01, mean=0, size=(num_inputs, num_hiddens), requires_grad=True)
    W2 = torch.normal(std=0.01, mean=0, size=(num_hiddens, num_outputs), requires_grad=True)
    b1 = torch.normal(std=0.01, mean=0, size=(num_hiddens,), requires_grad=True)
    b2 = torch.normal(std=0.01, mean=0, size=(num_outputs,), requires_grad=True)
    return (W1, b1, W2, b2)

def mlp_forward(X: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor,
                W2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """X 形状 (n, 1, 28, 28)。返回 (n, 10) 的 logits（不做 softmax）。"""
    X = X.reshape(X.size(0), -1)
    X1 = relu(X @ W1 + b1)
    return X1 @ W2 + b2

def count_params(net: nn.Sequential) -> int:
    """返回 net 里所有可训练参数的个数之和，Python 整数。"""
    sum = 0
    for i in net.parameters():
        sum += i.numel()
    return sum

def sgd(params: list, lr: float, batch_size: int) -> None:
    """与 B07 的 T4 相同，这一章自己再写一遍。

    对每个参数做 p ← p - lr * p.grad / batch_size，更新后就地清零它的梯度。
    就地更新，过程中不建计算图。
    """
    for p in params:
        with torch.no_grad():
            p -= lr * p.grad / batch_size
            p.grad = None
            
def cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """交叉熵损失。

    - logits 形状 (n, q)，是未经 softmax 的分数
    - y 形状 (n,)，元素是 0 到 q-1 的整数标签（不是独热）
    - 返回标量：这一批样本的交叉熵**之和**（不取平均）

    不许调用 nn.CrossEntropyLoss、torch.log_softmax 或 torch.logsumexp。
    """
    mx = logits.max(dim = -1, keepdim=True).values
    lse = mx + torch.log(torch.exp(logits - mx).sum(dim=1, keepdim=True))
    n = logits.size(0)
    idx = torch.arange(n, device=logits.device)
    lt = logits - lse
    return -(lt[idx, y]).sum()

def evaluate_accuracy(net, loader) -> float:
    """net 是可调用对象，接受 (n, 784) 返回 (n, 10) 的 logits。

    在整个 loader 上算准确率，返回 Python 浮点数。
    评估过程不应建立计算图。
    """
    t, s = 0, 0
    with torch.no_grad():
        for X, y in loader:
           ya = net(X.reshape(X.size(0), -1)) 
           t += int((ya.argmax(dim=1) == y).sum())
           s += ya.size(0)
    return t / s

def train_from_scratch(train_iter, test_iter, num_epochs: int = 10,
                       lr: float = 0.1, batch_size: int = 256,
                       seed: int = 0) -> dict:
    """用手写的 relu、mlp_forward、cross_entropy 与 sgd 训练。

    交叉熵与准确率用 B07 里自己写的那两份（`cross_entropy` 与 `evaluate_accuracy`，
    抄进本章或重新实现一遍都行）。
    返回一个字典，键固定为：
        train_loss -> 每个 epoch 的平均训练损失，浮点列表
        train_acc  -> 每个 epoch 结束时的训练集准确率，浮点列表
        test_acc   -> 每个 epoch 结束时的测试集准确率，浮点列表
    三个列表长度都是 num_epochs，**且必须是三个独立的列表**。
    """
    d = {}
    d["train_loss"], d["train_acc"], d["test_acc"] = [], [], []
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    W1, b1, W2, b2 = init_params(seed=seed) 
    W1, b1, W2, b2 = W1.to(device).requires_grad_(), b1.to(device).requires_grad_(), W2.to(device).requires_grad_(), b2.to(device).requires_grad_()
    for _ in range(num_epochs):
        total_loss, total_size = 0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            X = X.reshape(X.size(0), -1)
            y1 = mlp_forward(X, W1, b1, W2, b2)
            l = cross_entropy(y1, y)
            l.backward()
            sgd([W1, W2, b1, b2], lr, X.size(0))
            total_loss += l.item()
            total_size += W1.size(0)
        net = lambda X: relu(X @ W1 + b1) @ W2 + b2
        d["train_loss"].append(total_loss / total_size)
        d["train_acc"].append(evaluate_accuracy(net, train_iter))
        d["test_acc"].append(evaluate_accuracy(net, test_iter))
    return d
        
