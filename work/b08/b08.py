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
            p -= lr * p.grad
            p.grad = None


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
    for _ in num_epochs:
        