"""B12 参考答案。做完再看。

跑一遍：
    python3 solutions/b12/b12.py
或者拿它验断言：
    python3 tools/check_b12.py solutions/b12/b12.py
"""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path.cwd()   # 需要在仓库根目录运行；从 __file__ 推在拷到别处时会错


# ---------- T1 读数据 ----------

def load_houses(csv_dir: str = "data") -> tuple:
    train = pd.read_csv(f"{csv_dir}/kaggle_house_pred_train.csv")
    test = pd.read_csv(f"{csv_dir}/kaggle_house_pred_test.csv")
    return train, test


# ---------- T2 预处理 ----------

def preprocess(train_data, test_data) -> tuple:
    n_train = train_data.shape[0]

    # 去掉 Id（第 0 列）与标签（最后一列）；测试集没有标签，所以到末尾
    all_features = pd.concat(
        (train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]), ignore_index=True
    )

    # 数值特征：标准化，然后把缺失填 0（标准化后均值是 0，填 0 等于填均值）
    numeric = all_features.dtypes[all_features.dtypes != "object"].index
    all_features[numeric] = all_features[numeric].apply(
        lambda x: (x - x.mean()) / x.std()
    )
    all_features[numeric] = all_features[numeric].fillna(0)

    # 类别特征：转 one-hot，缺失单独成一列
    all_features = pd.get_dummies(all_features, dummy_na=True)

    # get_dummies 生成的是 bool 列，与 float 列混在一起时 .values 会变成 object 数组，
    # torch.tensor 吃不下。先统一转成 float。
    all_values = all_features.astype(float).values
    train_features = torch.tensor(all_values[:n_train], dtype=torch.float32)
    test_features = torch.tensor(all_values[n_train:], dtype=torch.float32)
    train_labels = torch.tensor(
        train_data["SalePrice"].values, dtype=torch.float32
    ).reshape(-1, 1)
    return train_features, test_features, train_labels


# ---------- T3 对数均方根误差 ----------

def log_rmse(preds: torch.Tensor, labels: torch.Tensor) -> float:
    # log 0 会得到 -inf，负数会得到 nan，所以先截到不小于 1
    clipped = torch.clamp(preds, min=1.0)
    diff = clipped.log() - labels.log()
    return torch.sqrt((diff**2).mean()).item()


# ---------- T4 训练 ----------

def get_net(num_features: int = 331):
    return nn.Sequential(nn.Linear(num_features, 1))


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs: int, lr: float, weight_decay: float,
          batch_size: int) -> tuple:
    g = torch.Generator().manual_seed(0)
    train_iter = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size, shuffle=True, generator=g,
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            loss_fn(net(X), y).backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            train_ls.append(log_rmse(net(train_features), train_labels))
            if test_labels is not None:
                test_ls.append(log_rmse(net(test_features), test_labels))
    return train_ls, test_ls


# ---------- T5 k 折 ----------

def get_k_fold_data(k: int, i: int, X: torch.Tensor, y: torch.Tensor) -> tuple:
    assert k > 1, "k 至少是 2"
    n = X.shape[0]
    base, extra = divmod(n, k)
    # 前 extra 份各多一个样本，保证各份大小相差不超过 1
    sizes = [base + (1 if j < extra else 0) for j in range(k)]
    starts = [sum(sizes[:j]) for j in range(k)]

    parts_x, parts_y = [], []
    for j in range(k):
        idx = slice(starts[j], starts[j] + sizes[j])
        parts_x.append(X[idx])
        parts_y.append(y[idx])

    valid_x, valid_y = parts_x[i], parts_y[i]
    train_x = torch.cat([parts_x[j] for j in range(k) if j != i], dim=0)
    train_y = torch.cat([parts_y[j] for j in range(k) if j != i], dim=0)
    return train_x, train_y, valid_x, valid_y


def k_fold(k: int, X: torch.Tensor, y: torch.Tensor, num_epochs: int,
           lr: float, weight_decay: float, batch_size: int) -> tuple:
    train_sum, valid_sum = 0.0, 0.0
    for i in range(k):
        tx, ty, vx, vy = get_k_fold_data(k, i, X, y)
        net = get_net(X.shape[1])
        train_ls, valid_ls = train(
            net, tx, ty, vx, vy, num_epochs, lr, weight_decay, batch_size
        )
        train_sum += train_ls[-1]
        valid_sum += valid_ls[-1]
    return train_sum / k, valid_sum / k


# ---------- T6 预测与提交 ----------

def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs: int, lr: float, weight_decay: float,
                   batch_size: int):
    net = get_net(train_features.shape[1])
    train_ls, _ = train(
        net, train_features, train_labels, None, None,
        num_epochs, lr, weight_decay, batch_size,
    )
    net.eval()
    with torch.no_grad():
        preds = net(test_features).squeeze(1).numpy()

    submission = pd.DataFrame({
        "Id": test_data["Id"].values,
        "SalePrice": preds,
    })
    out_dir = REPO / "work" / "b12"
    out_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_dir / "submission.csv", index=False)
    return submission


"""T7 改错

第 1 处：训练集与测试集**没有合并**就往下走
    后果：标准化用的是各自的统计量（尺度不同），one-hot 的类别集合也可能不同
    （列数对不上，或者列的顺序不同）。要先把两边的特征 `pd.concat` 起来一起处理，
    处理完再按行数切开。

第 2 处：标准化的统计量用的是**各自的**均值与标准差
    表面上对，实际上测试集被缩放到自己的尺度上，与训练集的尺度不一致。
    模型在训练集上学的权重用到测试集上就错了。
    正确做法：用合并后全部数据的均值与标准差。

第 3 处：`fillna(0)` 写在 `get_dummies` **之前**
    数值特征的缺失被填成 0，这没错；但类别特征的缺失也被填成 0，
    "缺失"这个信息就丢了——它本该是一个独立的类别。
    正确做法：类别特征不填，让 `pd.get_dummies(..., dummy_na=True)` 给缺失造一列。

第 4 处：训练集与测试集**分别** `get_dummies`
    两边的类别集合不同，生成的列数与列顺序都会不同，后面拼起来或者喂给模型就报错。
    正确做法：合并后一次 `get_dummies`。

（另有一处小问题：`train_data.iloc[:, 1:-1]` 去掉了 Id 与标签，这是对的；
但 `test_data.iloc[:, 1:]` 保留全部特征列，两边列数应当一致。原代码这里是对的。）
"""


if __name__ == "__main__":
    train_data, test_data = load_houses(str(REPO / "data"))
    print("训练集", train_data.shape, " 测试集", test_data.shape)

    tr_x, te_x, tr_y = preprocess(train_data, test_data)
    print("预处理后", tuple(tr_x.shape), tuple(te_x.shape), tuple(tr_y.shape))
    print("数值特征均值范围", round(tr_x.mean().item(), 3), " 有 nan 吗：", torch.isnan(tr_x).any().item())
    print("标签均值", round(tr_y.mean().item(), 1))

    # 用随机预测检验 log_rmse
    fake = tr_y.clone()
    print("log_rmse(真值, 真值) =", log_rmse(fake, tr_y))
    print("log_rmse(真值*2, 真值) =", round(log_rmse(fake * 2, tr_y), 4), " 应为 log(2) =", round(torch.log(torch.tensor(2.0)).item(), 4))

    # 训练一小会儿看趋势
    net = get_net(tr_x.shape[1])
    tr_ls, te_ls = train(net, tr_x, tr_y, None, None, 20, 5, 0, 64)
    print("前 3 轮训练 rmse", [round(v, 4) for v in tr_ls[:3]])
    print("后 3 轮训练 rmse", [round(v, 4) for v in tr_ls[-3:]])
