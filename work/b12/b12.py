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
    train_data = train_data.iloc[:,1:]
    test_data = test_data.iloc[:,1:]
    data = pd.concat(train_data, test_data)
