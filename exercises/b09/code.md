# B09 代码题

把你的实现写在 `work/b09/b09.py` 里，函数名与签名必须和下面一致。

验收：`just check b09`。断言全过，并且我读过代码之后，这一题才算过。

---

## 先看一段：这一章要做什么

这一章不换模型，做的是**实验**：造一批数据，用不同复杂度的模型去拟合，
从两条误差曲线上看欠拟合与过拟合。

用到的零件大部分是旧的：平方损失、`sgd`、小批量。新的只有四件：

- 造多项式特征（把一列 $x$ 扩成 $[x, x^2, \ldots, x^k]$）
- 用**平均**平方损失记录每一轮的误差（训练时可以仍用"和"）
- 用对数纵轴画两条曲线
- 按 k 折切分下标

**数据生成公式**（讲义 4.1 节）：

$$y = 1.2x - 3.4x^2 + 5.6x^3 + 5 + \epsilon,\qquad \epsilon \sim \mathcal{N}(0, 0.1^2),\quad x \sim \mathcal{N}(0, 1).$$

验收分三类：**精确**（形状、切分的性质、诊断结果）、**范围**（各阶模型的误差量级）、
**趋势**（训练误差单调下降）。

---

## T1 生成数据

```python
def make_data(n_train: int = 100, n_test: int = 100,
              seed: int = 0) -> tuple:
    """按上面的公式生成数据。

    返回 (x_train, y_train, x_test, y_test)，四个都是形状 (n, 1) 的 float32 张量。
    x 取标准正态随机数，噪声标准差 0.1。同一个 seed 给出相同结果。
    训练集在前、测试集在后也可以，但四个返回值的样本数要和参数对上。
    """
```

| 检查 | 期望 |
|---|---|
| 四个张量的形状 | `(100, 1)`、`(100, 1)`、`(100, 1)`、`(100, 1)` |
| `dtype` | `float32` |
| 同一 seed 调两次 | 结果完全相同 |
| 标签的量级 | 与公式相符（`y` 的标准差在 3 到 6 之间）|

## T2 多项式特征

```python
def poly_features(x: torch.Tensor, k: int) -> torch.Tensor:
    """把形状 (n, 1) 的 x 扩成 (n, k) 的 [x, x^2, ..., x^k]。"""
```

| 调用 | 期望 |
|---|---|
| `poly_features(torch.tensor([[2.0]]), 3)` | `[[2., 4., 8.]]` |
| `poly_features(x, 1)` | 与输入形状相同 |
| `poly_features(x, 5)` | 第 3 列等于第 1 列的 3 次方 |

## T3 训练并记录

```python
def train_poly(x_train: torch.Tensor, y_train: torch.Tensor,
               x_test: torch.Tensor, y_test: torch.Tensor, k: int,
               num_epochs: int = 100, lr: float = 0.01,
               batch_size: int = 10, seed: int = 0) -> dict:
    """用 k 阶多项式做小批量梯度下降。

    返回 {"train_loss": [...], "test_loss": [...]}，两个列表长度都是 num_epochs。

    列表里存的是**每一轮结束时用当前参数算出的平均平方损失**：
    这一轮的所有批次跑完之后，用最新的 w、b 在整个训练集上前向一次，取平均平方误差；
    测试集用同一组参数再算一次。元素是 Python 浮点数。

    不要存这两种：这一轮各批次的平均（受批次划分影响，而且训练中参数在变）；
    从开始到这一轮的累计平均（曲线会滞后，后期降得比真实慢）。

    模型就是 y = X @ w + b，w 形状 (k, 1)、b 形状 (1,)。
    参数初始化为全零（这个实验里可以，因为不存在对称性问题）。
    训练时的损失可以按 B06 的约定用"和"，但记录进列表的必须是平均值。
    """
```

| 检查 | 期望 |
|---|---|
| 两个列表的长度 | 都等于 `num_epochs` |
| `k=3`、`num_epochs=100` 跑完 | 训练损失低于 0.5，测试损失低于 1.0 |
| 训练损失的趋势 | 第一轮明显高于最后一轮 |
| `k=1` 跑完 | 训练损失高于 30（线性模型拟合不了三次函数）|

## T4 画图

```python
def plot_errors(train_loss: list, test_loss: list, path: str) -> None:
    """把两条误差曲线画在一张图上并存到 path。

    要求：横轴是轮数，纵轴用**对数刻度**，两条曲线要有图例区分，
    另存为文件（不要 plt.show()）。

    path 由调用方给，别把文件名写死。验收脚本传的是 "work/b09/curve.png"，
    跑完 just check b09 之后那张图就在那里，自己打开看。
    """
```

画法与四个要点见讲义 4.3 节。

| 检查 | 期望 |
|---|---|
| 调用后 | `path` 指向的文件存在且非空 |
| 纵轴 | 对数刻度 |
| 两条曲线 | 都有图例 |
| 图存到哪 | `work/b09/curve.png`，跑完自己打开看 |

## T5 k 折切分

```python
def k_fold_indices(n: int, k: int, seed: int = 0) -> list:
    """把 0 到 n-1 这 n 个下标切成 k 份。返回一个长度 k 的列表，
    每一项是一个 Python 整数列表。

    要求：k 份合起来正好是 0 到 n-1 各一次，不重不漏；各份大小相差不超过 1。
    """
```

| 调用 | 期望 |
|---|---|
| `k_fold_indices(10, 3)` | 三份，大小是 4、3、3 的某个排列 |
| 所有份合起来排序 | 等于 `list(range(10))` |
| 同一 seed 调两次 | 结果相同 |

## T6 看曲线下结论

```python
def diagnose(train_loss: list, test_loss: list) -> str:
    """给定两条曲线，返回 "underfitting"、"overfitting" 或 "good"。

    判据按顺序：
      1. 最后一轮训练损失 > 0.5           -> "underfitting"
      2. 最后一轮测试损失 > 训练损失 * 10  -> "overfitting"
      3. 其余                              -> "good"
    """
```

| 调用 | 期望 |
|---|---|
| `diagnose([5.0, 4.0], [5.5, 4.5])` | `"underfitting"` |
| `diagnose([0.01, 0.005], [0.02, 0.9])` | `"overfitting"` |
| `diagnose([0.1, 0.02], [0.12, 0.03])` | `"good"` |
| `diagnose([0.01], [5.0])` | `"overfitting"`（测试是训练的 500 倍）|

## T7 改错

下面这段代码想画"训练误差与测试误差随轮数变化"的曲线，有三处问题。
把代码抄进 `work/b09/b09.py` 顶部的注释里，每处标出**位置**、**为什么错**、**怎么改**。

```python
import matplotlib.pyplot as plt
import torch

w = torch.zeros(3, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
train_curve, test_curve = [], []

for epoch in range(100):
    for X, y in train_iter:
        pred = X @ w + b
        loss = ((pred - y) ** 2).sum()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
    train_curve.append(loss.item())
    test_curve.append(((X_test @ w + b - y_test) ** 2).sum())

plt.plot(train_curve, test_curve)
plt.show()
```

三处分别指三个位置：**记录的数值是"和"还是"平均"**、**梯度有没有累加**、
**画图那两行**。

三段提示：第一处会让两条曲线差一个批量大小的倍数，没法直接比；
第二处会让梯度越积越大，训练发散；第三处有两个毛病，一个关于坐标轴，一个关于怎么把图留下来。
