"""
import torch
import torch.nn as nn

def test_acc(net, loader):
    correct, total = 0, 0
    with torch.no_grad(): 没有关丢弃，测试的时候还在丢弃
        for X, y in loader:
            correct += (net(X).argmax(dim=1) == y).sum().item()
            total += y.numel()
    return correct / total

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.Dropout(0.2),
    nn.ReLU(), 反了，丢弃的是输出，这丢错东西了实际丢弃量打折扣
    nn.Linear(256, 10),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5, weight_decay=0.01) 这个实验要看丢弃法，不需要wd,要控制变量

for epoch in range(5):
    total, n = 0, 0
    for X, y in train_iter:
        optimizer.zero_grad()
        loss = loss_fn(net(X), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * y.numel()
        n += y.numel()
    print(f"epoch {epoch}: 训练损失 {total / n:.4f}  测试精度 {test_acc(net, test_iter):.4f}")
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, dataloader
from torchvision import datasets, transforms

def make_high_dim(n_train: int = 20, n_test: int = 100,
                  num_inputs: int = 200, seed: int = 0) -> tuple:
    """按讲义 1.3 节的公式生成数据。

    返回 (x_train, y_train, x_test, y_test)：
    x 形状 (n, 200)，取标准正态随机数；y 形状 (n, 1)。
    标签按 y = 0.05 + 0.01 * sum(x 的每一维) + 噪声，噪声标准差 0.01。
    四个张量都是 float32，同一个 seed 给出相同结果。
    """
    torch.manual_seed(seed)
    x_train = torch.randn(size=(n_train, num_inputs),dtype=torch.float32)
    x_test = torch.randn(size=(n_test, num_inputs), dtype=torch.float32)
    noise = torch.normal(size=(n_train, 1), std=0.01, mean=0, dtype=torch.float32)
    y_train = 0.05 + 0.01 * x_train.sum(dim=1, keepdim=True) + noise
    noise = torch.normal(size=(n_test, 1), std=0.01, mean=0, dtype=torch.float32)
    y_test = 0.05 + 0.01 * x_test.sum(dim=1, keepdim=True) + noise
    return (x_train, y_train, x_test, y_test)

def l2_penalty(w: torch.Tensor, lam: float) -> torch.Tensor:
    """返回 (lam / 2) * sum(w ** 2)，一个标量张量。"""
    return (lam / 2) * (w ** 2).sum()

def train_high_dim(train_iter, test_iter, lam: float = 0.0,
                   num_epochs: int = 3, lr: float = 0.003,
                   seed: int = 0) -> dict:
    """在高维数据上训练线性回归，返回 {"train_loss": [...], "test_loss": [...]}。

    - train_iter 与 test_iter 由调用方给（DataLoader），产出 (X, y)，形状分别是
      (batch, 200) 与 (batch, 1)
    - 权重 w 形状 (200, 1)，偏置 b 形状 (1,)，都用正态随机数初始化（标准差 0.01）
    - 训练时的损失 = 这一批的平方损失之和 + lam/2 * ||w||^2 / 批量大小
    - **记录进列表的是不含惩罚项的平均平方损失**（讲义 1.4 节）
    - 两个列表长度都是 num_epochs
    """
    torch.manual_seed(seed)
    net = nn.Linear(200, 1)
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    train_loss_list, test_loss_list = [], []
    for _ in range(num_epochs):
        net.train()
        train_sum, train_count = 0.0, 0
        for X, y in train_iter:
            batch_size = X.shape[0]
            y_hat = net(X)
            sq_sum = loss_fn(y_hat, y)
            loss = sq_sum + l2_penalty(net.weight, lam) / batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_sum   += sq_sum.item()
            train_count += batch_size
        train_loss_list.append(train_sum / train_count)
        net.eval()
        test_sum, test_count = 0.0, 0
        with torch.no_grad():
            for X, y in test_iter:
                y_hat = net(X)
                test_sum   += loss_fn(y_hat, y).item()
                test_count += X.shape[0]
        test_loss_list.append(test_sum / test_count)
    return {"train_loss": train_loss_list, "test_loss": test_loss_list}
    
def dropout_layer(X: torch.Tensor, dropout: float) -> torch.Tensor:
    """按丢弃概率 dropout 随机清零 X 的元素，并把留下的元素除以 (1 - dropout)。

    三个边界：
      dropout == 0 时原样返回
      dropout == 1 时返回全零（与 X 同形状、同 dtype、同设备）
    其余情况随机决定每个元素是否保留。

    返回与 X 同形状同 dtype 的新张量，不修改 X 本身。
    """
    if dropout == 0:
        return X
    if dropout == 1:
        return torch.zeros_like(X, dtype=X.dtype, device=X.device)
    return ((torch.rand(X.shape) > dropout) * X) / (1 - dropout)

def make_dropout_net(num_inputs: int = 784, num_hiddens1: int = 256,
                     num_hiddens2: int = 256, num_outputs: int = 10,
                     dropout1: float = 0.2, dropout2: float = 0.5,
                     seed: int = 0):
    """返回一个 nn.Sequential：
    Flatten → Linear(784,256) → ReLU → Dropout(dropout1)
            → Linear(256,256) → ReLU → Dropout(dropout2)
            → Linear(256,10)
    """
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, num_hiddens1),
        nn.ReLU(),
        nn.Dropout(dropout1),
        nn.Linear(num_hiddens1, num_hiddens2),
        nn.ReLU(),
        nn.Dropout(dropout2),
        nn.Linear(num_hiddens2, num_outputs)
    )

def evaluate(net:nn.Sequential, loader:DataLoader) -> float:
    """net 接受 (batch, 1, 28, 28) 返回 (batch, 10) 的 logits。

    在整个 loader 上算准确率，返回 Python 浮点数。
    **评估时丢弃层不应生效**，而且评估前后要正确恢复模型原来的模式。
    """
    is_training = net.training
    net.eval()
    total_size, total_loss = 0, 0
    with torch.no_grad():
        for X, y in loader:
            y_hat = net(X)
            total_size += X.size(0)
            total_loss += (y_hat.argmax(dim=1) == y).sum().item()
    if is_training:
        net.train()
    return total_loss / total_size


def train_dropout(train_iter:DataLoader, test_iter:DataLoader, use_dropout: bool = True,
                  num_epochs: int = 5, lr: float = 0.5,
                  batch_size: int = 256, seed: int = 0) -> dict:
    """训练上面那个 MLP，返回 {"train_loss": [...], "train_acc": [...],
    "test_acc": [...]}，三个列表长度都是 num_epochs，元素是 Python 浮点数。

    use_dropout=False 时两个丢弃层的概率都设成 0。
    损失用 nn.CrossEntropyLoss（它返回的是平均损失，记录的也是平均，不必再除）。
    """
    torch.manual_seed(seed)
    train_loss, train_acc, test_acc = [], [], []
    d1, d2 = 0.2, 0.5
    if use_dropout is False:
        d1, d2 = 0, 0
    net = make_dropout_net(dropout1=d1, dropout2=d2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0)
    for _ in range(num_epochs):
        train_total_size, train_total_loss = 0, 0
        for X, y in train_iter:
            optimizer.zero_grad()
            net.train()
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item() * X.size(0)
            train_total_size += X.size(0)
        train_loss.append(train_total_loss / train_total_size)
        net.eval()
        train_acc.append(evaluate(net, train_iter))
        test_acc.append(evaluate(net, test_iter))
    return {
        "train_loss" : train_loss, 
        "train_acc" : train_acc, 
        "test_acc" : test_acc
    }

