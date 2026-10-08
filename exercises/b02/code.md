# B02 代码题

把你的实现写在 `work/b02/b02.py` 里，函数名和签名必须和下面一致。
验收：`just check b02`。断言全过，并且我读过代码之后，这一题才算过。

不许用 `notes/b02-data-ops.md` 之外的库函数绕过要求，比如题目说"不许用 `expand`"就是不许用。

---

## T1 张量自述

`dict` 是 Python 的字典（`{"键": 值}`），写法与用法见 A1 讲义第 1.2 节。
`-> dict` 那个位置是类型注解，不参与运行，只是给人看的。

```python
def tensor_info(x: torch.Tensor) -> dict:
    """返回关于 x 的一份描述，键固定为：
    shape           -> tuple，例如 (3, 4)
    numel           -> int，元素总数
    dtype           -> str，例如 'float32'
    device          -> str，例如 'cpu' 或 'cuda:0'
    is_contiguous   -> bool
    """
```

要求：`shape` 必须是普通的 Python `tuple`（不是 `torch.Size`），`dtype` 必须是字符串。

## T2 自己判断能不能广播

```python
def safe_broadcast_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
```

行为：
- 两个形状可以广播时，返回相加结果。
- 不能广播时，抛 `ValueError`，且异常信息里必须包含**第一个不匹配的维度是第几维**（从 0 开始数，按左对齐后的结果维度计，最右边一维是最后一维）。

举例：`a.shape == (3, 4)`、`b.shape == (2, 4)` 时，不匹配的是倒数第二维。
形状不同长度的对齐规则按讲义第 4 节。

这一题**不许用** `torch.broadcast_shapes` 这类现成的形状工具，也不许直接把两个张量相加了事。
形状对齐、逐维判断、报错信息全部自己写——这道题考的就是广播规则本身。

## T3 视图还是副本

```python
def slice_is_view() -> bool:
    """构造一个张量，取它的一个切片，修改切片，返回原张量是否跟着变。"""


def clone_is_copy() -> bool:
    """同上，但中间做一次 clone，返回原张量是否跟着变。"""
```

两个函数都不接受参数，自己造数据。返回值必须是 `bool`，且要真的做实验得出，不许直接 `return True`。

## T4 从 CSV 到张量

```python
def load_csv(path: str) -> tuple[torch.Tensor, list[str], dict]:
    """读取 CSV，返回 (特征张量, 特征名列表, 列统计信息)。
```

返回的是一个三元组：张量、字符串列表、字典。`tuple[...]`、`list[...]`、`dict` 都是类型注解。

要求：
- 数值列：用该列**均值**填缺失值。
- 字符串列：做 one-hot 编码，每一类一列，列名形如 `原列名_类别值`。
- 返回的特征张量 dtype 为 `float32`。
- 特征名列表的顺序必须和特征张量各列一一对应。
- 列统计信息是 dict：`{"数值列": [...], "类别列": {...}}`，其中 `类别列` 的值是"该列各取值的个数"。

测试用的 CSV 由 `tools/check_b02.py` 生成，你不用管路径。

## T5 改错

下面这段代码有三处问题。把代码抄进 `work/b02/b02.py` 顶部的注释里，
每处标出三样：**位置**（第几行）、**为什么错**、**怎么改**。

```python
import torch

# 造一批数据，准备拿去做浮点运算
x = torch.arange(12)

# 变形成 3 行 4 列
X = x.reshape(3, 4)

# 打印它的形状
print(X.size)

# 取第一行改一改，认为 X 不会受影响
row = X[0]
row[0] = 100.0
print(X[0, 0])
```

三处分别关于：**元素类型**、**属性与方法的区别**、**切片是视图还是副本**。

最后一处给的是"作者的预期"，你要指出实际结果与预期差在哪。

可以先跑一遍验证自己的判断，但先把三处分析写下来再跑。

