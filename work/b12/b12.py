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
    optimizer = torch.optim.Adam()
    loss_fn = 
