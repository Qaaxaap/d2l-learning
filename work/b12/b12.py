"""
import pandas as pd
import torch

def preprocess(train_data, test_data):
    n_train = train_data.shape[0]

    # 各自处理，不合并
    train_features = train_data.iloc[:, 1:-1].copy()
    test_features = test_data.iloc[:, 1:].copy() 不能各自处理，否则one-hot、标准化出问题。

    # 分别标准化
    for col in train_features.columns:
        if train_features[col].dtype != "object":
            train_features[col] = (train_features[col] - train_features[col].mean()) / train_features[col].std()
    for col in test_features.columns:
        if test_features[col].dtype != "object": 应当一起标准化
            test_features[col] = (test_features[col] - test_features[col].mean()) / test_features[col].std()

    # 先填缺失，再 one-hot
    train_features = train_features.fillna(0)
    test_features = test_features.fillna(0) 先one-hot,否则0也会被one-hot拆开。

    # 分别 one-hot
    train_features = pd.get_dummies(train_features) 应该一起one-hot
    test_features = pd.get_dummies(test_features)

    return (
        torch.tensor(train_features.values, dtype=torch.float32),
        torch.tensor(test_features.values, dtype=torch.float32),
        torch.tensor(train_data["SalePrice"].values.reshape(-1, 1), dtype=torch.float32),
    )
"""
import pandas as pd
import torch
import torch.nn as nn

def load_houses(csv_dir: str = "data") -> tuple:
    """读两个 CSV，返回 (train_data, test_data) 两个 DataFrame。

    csv_dir 是数据目录，两个文件分别叫 kaggle_house_pred_train.csv
    与 kaggle_house_pred_test.csv。
    """
    csvd = csv_dir + "/kaggle_house_pred"
    train_data = pd.read_csv(csvd + "_train.csv")
    test_data = pd.read_csv(csvd + "_test.csv")
    return (train_data, test_data)

def preprocess(train_data: pd.DataFrame, test_data: pd.DataFrame
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把两张表变成三个张量，返回 (train_features, test_features, train_labels)。

    步骤（讲义 2 节）：
      1. 去掉 Id 列
      2. 训练集与测试集的特征按行合并
      3. 数值特征标准化（减均值除标准差），然后把缺失值填 0
      4. 类别特征转 one-hot，缺失也当成一个类别
      5. 按原来的行数切回训练/测试，转成 float32 张量

    形状要求：train_features (1460, 330)、test_features (1459, 330)、
    train_labels (1460, 1)。
    """
    label = torch.tensor(train_data["SalePrice"].values.reshape(-1, 1), dtype=torch.float32)
    train_data = train_data.iloc[:,1:-1]
    test_data = test_data.iloc[:,1:]
    data = pd.concat((train_data, test_data))
    numeric_cols = data.dtypes[data.dtypes != "object"].index
    data[numeric_cols] = data[numeric_cols].apply(
        lambda x: (x - x.mean()) / (x.std())
    )
    data = pd.get_dummies(data, dummy_na=True).astype(float)
    data = data.fillna(0)
    train = torch.tensor(data.iloc[:train_data.shape[0],:].values, dtype=torch.float32)
    test = torch.tensor(data.iloc[train_data.shape[0]:,:].values, dtype=torch.float32)
    return (train, test, label)
    
def log_rmse(preds: torch.Tensor, labels: torch.Tensor) -> float:
    clipped = torch.clamp(preds, min=1.0)
    return torch.sqrt(((clipped.log() - labels.log()) ** 2).mean()).item()

def train(net: torch.nn.Module,
          train_features: torch.Tensor, train_labels: torch.Tensor,
          test_features: torch.Tensor | None, test_labels: torch.Tensor | None,
          num_epochs: int, lr: float, weight_decay: float,
          batch_size: int) -> tuple[list[float], list[float]]:
    """训练 net，返回 (train_ls, test_ls)。

    两个列表长度都是 num_epochs，元素是那一轮结束后的 log_rmse（Python 浮点数）。
    test_features 与 test_labels 可以是 None——那时只返回训练曲线，
    第二个列表给空列表。

    优化器用 torch.optim.Adam，损失用 log_rmse（不是 MSE）。
    """
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        n = train_features.shape[0]
        for i in range(0, n, batch_size):
            optimizer.zero_grad()
            X, y = train_features[i:min(n,i+batch_size)], train_labels[i:min(n,i+batch_size)]
            l = loss_fn(net(X), y)
            l.backward()
            optimizer.step()
        if test_features is not None:
            test_ls.append(log_rmse(net(test_features), test_labels))
        train_ls.append(log_rmse(net(train_features), train_labels))
    return (train_ls, test_ls)

def get_k_fold_data(k: int, i: int, X: torch.Tensor, y: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回第 i 折的 (X_train, y_train, X_valid, y_valid)。

    k 份的大小相差不超过 1。**用 Python 算切分索引是正常的**——没有"纯 Python"
    与"张量版"的对立；切完之后拼接才用张量操作（`torch.cat`）。
    余数怎么分配：`base, extra = divmod(n, k)`，前 extra 份各 base+1 个、其余各 base 个。
    也可以直接 `torch.tensor_split(X, k)`，两种都收。

    第 i 份做验证，其余拼接做训练。所有样本都必须被用到且只用一次。
    """
    sp = lambda x : list(torch.tensor_split(x, k))[i]
    spn = lambda x: torch.cat(torch.tensor_split(x, k)[0:i] + torch.tensor_split(x, k)[i+1:])
    return (spn(X), spn(y), sp(X), sp(y))


def k_fold(k: int, X: torch.Tensor, y: torch.Tensor, num_epochs: int,
           lr: float, weight_decay: float, batch_size: int
           ) -> tuple[float, float]:
    """跑 k 折，返回 (avg_train_rmse, avg_valid_rmse) 两个浮点数。"""
    total_train_loss, total_test_loss = 0, 0
    for i in range(k):
        net = nn.Sequential(nn.Flatten(), nn.Linear(X.numel() // X.size(0), y.numel() // y.size(0)))
        x_train, y_train, x_test, y_test = get_k_fold_data(k, i, X, y)
        train_ls, test_ls = train(net, x_train, y_train, x_test, y_test, num_epochs, lr, weight_decay, batch_size)
        total_train_loss += train_ls[-1]
        total_test_loss += test_ls[-1]
    return total_train_loss / k, total_test_loss / k

def train_and_pred(train_features: torch.Tensor,
                   test_features: torch.Tensor,
                   train_labels: torch.Tensor,
                   test_data: pd.DataFrame,
                   num_epochs: int, lr: float, weight_decay: float,
                   batch_size: int) -> pd.DataFrame:
    """用全部训练数据重训一个模型，对测试集预测，返回提交用的 DataFrame。

    返回的 DataFrame 有两列：Id 与 SalePrice，行数与测试集相同（1459）。
    同时把它存到 work/b12/submission.csv（不带行号）。
    """
    net = nn.Sequential(nn.Linear(train_features.shape[1], train_labels.shape[1]))
    train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    test_data["SalePrice"] = pd.Series(net(test_features).detach().numpy().reshape(-1))
    submission = pd.concat([test_data["Id"], test_data["SalePrice"]], axis=1)
    submission.to_csv("submission.csv", index=False)
    return submission