# B12 代码题

把你的实现写在 `work/b12/b12.py` 里，函数名与签名必须和下面一致。

验收：`just check b12`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章做什么

一份真实数据集：1460 个训练样本、79 个特征、标签是房价。要做的事有一套固定顺序：

```
读 CSV → 去掉 Id → 合并训练/测试特征 → 标准化 → 填缺失 → one-hot → 转张量
      → 定义指标 → 训练 → k 折选超参数 → 用全部数据重训 → 预测 → 存 CSV
```

模型只有一个线性层。这一章的代码量比前几章大，但每一步都不难——难点在数据上的细节。

**数据已经放在仓库的 `data/` 目录下**：`kaggle_house_pred_train.csv` 与
`kaggle_house_pred_test.csv`。不用自己下载。

**标签的位置**：训练集的最后一列是 `SalePrice`，测试集没有这一列。
所以训练集的特征是 `iloc[:, 1:-1]`、测试集是 `iloc[:, 1:]`。这个 `-1` 只因这份数据
把标签放在最后一列，换一份数据先看标签在哪。

**类型注解**：`pd.DataFrame` 是 pandas 的二维表，`torch.Tensor` 是张量，
`torch.nn.Module` 是模型的基类（B06 讲过），`tuple[X, Y]` 表示"返回两个值，分别是 X 和 Y"。
`X | None` 表示"可以是 X，也可以是 None"。

**这一章需要 pandas**。讲义 1.1 与 2 节里用到的每个 API 都讲过
（`read_csv`、`shape`、`dtypes`、`iloc`、`concat`、`apply`、`fillna`、`get_dummies`、
`values`、`to_csv`），不必查文档。

验收分三类：**精确**（列数、形状、`log_rmse` 的数值）、
**范围**（k 折的平均验证误差）、**趋势**（训练误差随轮数下降）。

---

## T1 读数据

```python
def load_houses(csv_dir: str = "data") -> tuple:
    """读两个 CSV，返回 (train_data, test_data) 两个 DataFrame。

    csv_dir 是数据目录，两个文件分别叫 kaggle_house_pred_train.csv
    与 kaggle_house_pred_test.csv。
    """
```

| 检查 | 期望 |
|---|---|
| 返回两个 `DataFrame` | 是 |
| `train_data.shape` | `(1460, 80)` |
| `test_data.shape` | `(1459, 80)` |
| 训练集有 `SalePrice` 列 | 是 |
| 测试集**没有** `SalePrice` 列 | 是 |

## T2 预处理

```python
def preprocess(train_data: pd.DataFrame, test_data: pd.DataFrame
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把两张表变成三个张量，返回 (train_features, test_features, train_labels)。

    步骤（讲义 2 节）：
      1. 去掉不该进特征的列。训练集要去掉 Id（第 0 列）**与标签 SalePrice（最后一列）**；
         测试集只去 Id（它本来就没有 SalePrice）。标签留在特征里等于把答案告诉模型
      2. 训练集与测试集的特征按行合并
      3. 数值特征标准化（减均值除标准差），然后把缺失值填 0
      4. 类别特征转 one-hot，缺失也当成一个类别
      5. 按原来的行数切回训练/测试，转成 float32 张量

    形状要求：train_features (1460, 330)、test_features (1459, 330)、
    train_labels (1460, 1)。
    """
```

| 检查 | 期望 |
|---|---|
| 三个返回值的形状 | `(1460, 330)`、`(1459, 330)`、`(1460, 1)` |
| `dtype` | 都是 `float32` |
| 特征里有没有 `nan` | 一个都没有 |
| 数值列标准化后 | 均值接近 0、标准差接近 1（允许 0.05 的误差）|
| `train_labels` 的均值 | 在 15 万到 25 万之间（真实房价的量级）|

## T3 对数均方根误差

```python
def log_rmse(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """返回对数均方根误差，Python 浮点数。

    预测值先截到不小于 1（避免 log 出问题），再算
        sqrt(mean((log(preds) - log(labels)) ** 2))
    """
```

| 检查 | 期望 |
|---|---|
| 预测与标签完全相同 | 0.0 |
| 预测全都乘 2 | `log(2) ≈ 0.693` |
| 预测里含 0 或负数 | 不报错、不返回 `nan`（被 clip 到 1）|
| 返回类型 | `float` |

## T4 训练

```python
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
```

| 检查 | 期望 |
|---|---|
| 两个返回列表的长度 | 都是 `num_epochs` |
| 训练误差 | 逐轮下降（最后一轮低于第一轮的 80%）|
| 传 `None` 做验证集 | 不报错，第二个列表是空列表 |
| 与 `log_rmse` 配合 | 列表里的值与单独调 `log_rmse` 的结果一致 |

## T5 k 折

```python
def get_k_fold_data(k: int, i: int, X: torch.Tensor, y: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回第 i 折的 (X_train, y_train, X_valid, y_valid)。

    k 份的大小相差不超过 1（用 divmod 分配余数，B09 的 T5 做过类似的事）。
    第 i 份做验证，其余拼接做训练。所有样本都必须被用到且只用一次。
    """


def k_fold(k: int, X: torch.Tensor, y: torch.Tensor, num_epochs: int,
           lr: float, weight_decay: float, batch_size: int
           ) -> tuple[float, float]:
    """跑 k 折，返回 (avg_train_rmse, avg_valid_rmse) 两个浮点数。"""
```

| 检查 | 期望 |
|---|---|
| 每折的训练与验证样本数之和 | 等于总数 1460 |
| k 份的大小 | 相差不超过 1 |
| 所有折的验证集拼起来 | 覆盖全部 1460 个样本，没有重复 |
| `k=5`、50 轮跑完 | 平均验证误差在 0.18 到 0.35 之间（对数均方根误差）|

## T6 预测与提交

```python
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
```

| 检查 | 期望 |
|---|---|
| 返回的 `DataFrame` 列名 | `["Id", "SalePrice"]` |
| 行数 | 1459 |
| `Id` 列 | 与测试集原来的 Id 一致（顺序也一致）|
| `SalePrice` | 都是正数，量级在 5 万到 100 万之间 |
| `work/b12/submission.csv` | 存在，1459 行（不含表头），第一行是列名 |

## T7 改错

下面这段预处理代码有四处问题。把代码抄进 `work/b12/b12.py` 顶部的注释里，
每处标出**位置**、**为什么错**、**怎么改**。

```python
import pandas as pd
import torch

def preprocess(train_data, test_data):
    n_train = train_data.shape[0]

    # 各自处理，不合并
    train_features = train_data.iloc[:, 1:-1].copy()
    test_features = test_data.iloc[:, 1:].copy()

    # 分别标准化
    for col in train_features.columns:
        if train_features[col].dtype != "object":
            train_features[col] = (train_features[col] - train_features[col].mean()) / train_features[col].std()
    for col in test_features.columns:
        if test_features[col].dtype != "object":
            test_features[col] = (test_features[col] - test_features[col].mean()) / test_features[col].std()

    # 先填缺失，再 one-hot
    train_features = train_features.fillna(0)
    test_features = test_features.fillna(0)

    # 分别 one-hot
    train_features = pd.get_dummies(train_features)
    test_features = pd.get_dummies(test_features)

    return (
        torch.tensor(train_features.values, dtype=torch.float32),
        torch.tensor(test_features.values, dtype=torch.float32),
        torch.tensor(train_data["SalePrice"].values.reshape(-1, 1), dtype=torch.float32),
    )
```

四处分别指四个位置：**合并与不合并**、**标准化的统计量**、**填缺失与 one-hot 的顺序**、
**分别 one-hot**。

四段提示：第一处影响的是后面三步能不能一致；第二处两种写法都能跑，但有一种会让测试集
落到与训练集不同的尺度上；第三处只影响类别特征；第四处会直接报形状错。
