# B03 线性代数

纸质书没有这一章，第一版把它压缩在附录 A.1 的几行公式里，第二版是 2.3 节（看 PyTorch tab）。
本单元要做的是把附录里那几行公式落到张量运算上：标量到张量的层级、`*` 与 `@` 的分工、求和
降维的形状规则、范数。后面线性回归、softmax、卷积的每一行代码都建立在这套符号与 API 上。

本文的 torch 行为在 torch 2.14 上实测，报错文本照抄输出。

## 0. 电子版 MXNet 代码 → PyTorch

第一版没有这一章的代码。电子版 MXNet tab 用的是 numpy 兼容接口 `np`（第一版书里对应的模块是 `nd`）。
对照如下，注意最后四行是第一版和 torch 差别最大的地方：

| 电子版 MXNet（`np.`） | PyTorch |
|---|---|
| `np.array(3.0)` | `torch.tensor(3.0)` |
| `np.arange(4)`、`np.arange(20).reshape(5, 4)` | 同名同义 |
| `A.T` | `A.T`，二维一致；三维以上语义不同，见第 2 节 |
| `A.copy()` | `A.clone()` |
| `A.sum(axis=0)`、`A.sum(axis=1, keepdims=True)` | `A.sum(dim=0)`、`A.sum(dim=1, keepdim=True)`，`axis` 与 `keepdim` 在 torch 里也能用 |
| `A.mean(axis=0)`、`A.cumsum(axis=0)` | `A.mean(dim=0)`、`A.cumsum(dim=0)` |
| `np.dot(x, y)` | `torch.dot(x, y)`，只接受两个一维向量 |
| `np.dot(A, x)` | `torch.mv(A, x)` 或 `A @ x` |
| `np.dot(A, B)` | `torch.mm(A, B)` 或 `A @ B` |
| `np.linalg.norm(u)`、`np.linalg.norm(X)` | `torch.linalg.vector_norm(u)`、`torch.linalg.norm(X)` |
| `np.abs(u).sum()` | `torch.abs(u).sum()` |

MXNet 的 `dot` 一个函数管点积、矩阵向量积、矩阵乘法；torch 把它们拆成 `dot`、`mv`、`mm`、`matmul`。
用错函数会直接报错，不算坏事。

## 1. 标量、向量、矩阵、张量

v1 附录 A.1 的符号约定：向量写作列向量 $\mathbf{x}\in\mathbb{R}^n$，矩阵 $\mathbf{A}\in\mathbb{R}^{m\times n}$，
元素记 $a_{ij}$（第 $i$ 行第 $j$ 列）。代码里轴数与数学对象的对应关系：

```python
import torch

s = torch.tensor(3.0)      # 标量：0 维张量，形状 torch.Size([])
s.shape, s.dim()           # (torch.Size([]), 0)
len(s)                     # TypeError: len() of a 0-d tensor

x = torch.arange(4)        # 向量：一维
len(x), x.shape, x[3]      # (4, torch.Size([4]), tensor(3))

A = torch.arange(20).reshape(5, 4)      # 矩阵：两轴
X = torch.arange(24).reshape(2, 3, 4)   # 三轴张量
X.dim(), X.shape[0], X.shape            # (3, 2, torch.Size([2, 3, 4]))
```

“维度”一词有两种用法，v2 原文专门澄清过：讲向量或某个轴时，维数是它的长度（$n$ 维向量）；
讲张量时，维度是轴的数量（三阶张量）。代码里 `x.dim()` 给轴数，`x.shape[i]` 给第 $i$ 轴长度，
两者不要混。

数据集的存放约定与数学写法相反的地方也在这里：数学上向量默认是列向量，代码里一个样本占一行，
一个 batch 是形状 `(batch_size, 特征数)` 的矩阵，最外轴用来遍历样本。

## 2. 转置与对称矩阵

```python
A = torch.arange(20).reshape(5, 4)
A.T.shape                     # (4, 5)；A.T 是视图，不是拷贝

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B == B.T                      # 逐元素比较，返回 bool 张量
torch.equal(B, B.T)           # True，两个张量完全相同

C = A.clone()                 # 复制一份；C = A 只是多一个名字，改 C 会改 A
```

对称矩阵指 $\mathbf{A}=\mathbf{A}^\top$。上面 `B` 是对称的，`torch.equal` 给出单个布尔值，
比 `B == B.T` 更适合放进断言。

三轴以上 `.T` 的行为与电子版 MXNet 不同，torch 里它翻转**所有**轴：

```python
X = torch.arange(24).reshape(2, 3, 4)
X.T.shape              # (4, 3, 2)，同时抛 UserWarning: The use of `x.T` on tensors of dimension
                       # other than 2 to reverse their shape is deprecated
X.transpose(0, 2).shape    # (4, 3, 2)，只换指定的两轴
X.permute(2, 0, 1).shape   # (4, 2, 3)，按给定顺序重排
```

实测该警告在 torch 2.14 上就会出现，未来版本会报错。三轴以上的转置一律写 `transpose` 或 `permute`。

## 3. 按元素乘法与矩阵乘法

标量、向量、矩阵到高阶张量有一批共用的性质：一元按元素运算不改变形状，同形状的两个张量做
二元按元素运算，结果也是同形状。逐元素相乘叫 Hadamard 积（符号 $\odot$，v1 附录 A.1 有定义）：

```python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
A * B                      # Hadamard 积，仍是 (5, 4)
2 + A, (2 * A).shape       # 标量与张量运算按元素做，形状不变
```

矩阵乘法是另一回事。设 $\mathbf{A}\in\mathbb{R}^{m\times p}$、$\mathbf{B}\in\mathbb{R}^{p\times n}$，
乘积 $\mathbf{C}=\mathbf{AB}$ 的第 $i$ 行第 $j$ 列元素是

$$c_{ij}=\sum_{k=1}^{p}a_{ik}b_{kj}.$$

把 $\mathbf{A}$ 按行拆成行向量 $\mathbf{a}_i^\top$，把 $\mathbf{B}$ 按列拆成列向量 $\mathbf{b}_j$，
上式就是 $c_{ij}=\mathbf{a}_i^\top\mathbf{b}_j$。由此得到一个理解方式：$\mathbf{AB}$ 相当于对
$\mathbf{B}$ 的每一列做一次矩阵-向量积，共 $n$ 次，再把结果拼成 $m\times n$ 的矩阵。
形状规则是左操作数的最后一维必须等于右操作数的倒数第二维，即 `(m, p) @ (p, n) -> (m, n)`。

```python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
C = torch.ones(4, 3)
A @ C                  # (5, 3)
torch.mm(A, C)         # 同上，只接受两个二维张量
```

四个函数的分工，用错就报错：

| 函数 | 接受的输入 | 典型调用 |
|---|---|---|
| `torch.dot` | 两个一维、dtype 相同 | `torch.dot(x, y)` |
| `torch.mv` | 二维 + 一维 | `torch.mv(A, x)` |
| `torch.mm` | 两个二维 | `torch.mm(A, C)` |
| `torch.matmul` 或 `@` | 任意，含批量 | `A @ C` |

实测报错：`torch.dot(A, x)` 报 `1D tensors expected, but got 2D and 1D tensors`；
`torch.mm(A, x)` 报 `mat2 must be a matrix`；形状不匹配报
`mat1 and mat2 shapes cannot be multiplied (5x4 and 3x2)`；两侧 dtype 不同报
`expected m1 and m2 to have the same dtype`。

`*` 与 `@` 不能互换，因为形状规则不同，且矩阵乘法不满足交换律。两个方阵写反顺序不会报错，
结果却不同，这类错误在实现线性层时最常见（`X @ W` 与 `W @ X` 的形状约束不一样）。

## 4. 降维求和与 keepdim

```python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
A.sum()                  # 0 维张量
A.sum(dim=0)             # (4,)，压掉轴 0，剩下的轴 1 长度不变
A.sum(dim=1)             # (5,)
A.sum(dim=[0, 1])        # 等价于 A.sum()
A.mean(), A.sum() / A.numel()
A.mean(dim=0)            # (4,)
A.cumsum(dim=0)          # 累积和，(5, 4)，不降维
```

规则是：指定哪个 `dim`，输出里那个轴就消失。`dim` 可以传一个列表一次压掉多个轴。

`keepdim=True` 让被压掉的轴以长度 1 的形式留在原位，这一步是为了后面的广播：

```python
sum_A = A.sum(dim=1, keepdim=True)   # (5, 1)
A / sum_A                            # 每行除以该行之和，形状仍是 (5, 4)
A / A.sum(dim=1)                     # RuntimeError: The size of tensor a (4) must match the
                                     # size of tensor b (5) at non-singleton dimension 1
```

不加 `keepdim` 时得到 `(5,)`，右对齐到最后一维与 4 冲突，广播不成立；加上之后得到 `(5, 1)`，
长度为 1 的那一维可以广播到 4。实测 `keepdims=True` 与 `keepdim=True`、`axis=` 与 `dim=`
在 torch 2.14 上都可用（电子版 pytorch tab 写的是 `keepdims`），为与官方文档一致，新代码写 `keepdim`。

## 5. 点积

两个向量 $\mathbf{x},\mathbf{y}\in\mathbb{R}^d$ 的点积是相同位置元素乘积之和：

$$\mathbf{x}^\top\mathbf{y}=\sum_{i=1}^{d}x_iy_i .$$

```python
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4)
torch.dot(x, y)          # tensor(6.)
(x * y).sum()            # tensor(6.)，与上式定义一致
```

`torch.dot` 只接受一维且 dtype 相同的张量，两个条件缺一个就报错；用 `(x * y).sum()` 不受此限制。
点积的两个常见用途：权重向量与特征向量的加权和；把两个向量归一化到单位长度后，点积就是夹角余弦，
后者要用到下一节的范数。

## 6. 矩阵-向量积

矩阵 $\mathbf{A}\in\mathbb{R}^{m\times n}$ 与向量 $\mathbf{x}\in\mathbb{R}^n$ 相乘，结果是长度 $m$ 的向量，
第 $i$ 个元素是 $\mathbf{A}$ 的第 $i$ 行与 $\mathbf{x}$ 的点积：

$$\mathbf{A}\mathbf{x}=\begin{bmatrix}\mathbf{a}_1^\top\mathbf{x}\\ \vdots\\ \mathbf{a}_m^\top\mathbf{x}\end{bmatrix}.$$

```python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
x = torch.ones(4)
torch.mv(A, x)           # (5,)
A @ x                    # 同上；torch.dot(A, x) 会报错，它不处理矩阵
```

写成数学形式是 $\mathbb{R}^n\to\mathbb{R}^m$ 的映射，形状检查只看 $\mathbf{A}$ 的列数是否等于
$\mathbf{x}$ 的长度。深度学习里每一层的前向计算就是一次矩阵-向量积（一个 batch 时是矩阵-矩阵乘法）。

## 7. 范数

范数把向量映射成标量，用来衡量“有多大”。v2 列出四条性质，用它们可以判断一个函数算不算范数：
$f(\alpha\mathbf{x})=|\alpha|f(\mathbf{x})$（正齐次）、$f(\mathbf{x}+\mathbf{y})\le f(\mathbf{x})+f(\mathbf{y})$
（三角不等式）、$f(\mathbf{x})\ge 0$、以及 $f(\mathbf{x})=0$ 当且仅当 $\mathbf{x}=\mathbf{0}$。

$L_2$ 范数是元素平方和的平方根，$L_1$ 范数是元素绝对值之和，两者的通式是
$\|\mathbf{x}\|_p=(\sum_i|x_i|^p)^{1/p}$。矩阵的 Frobenius 范数把所有元素拉平后求 $L_2$ 范数：

$$\|\mathbf{X}\|_F=\sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n}x_{ij}^2}.$$

```python
u = torch.tensor([3.0, -4.0])
torch.linalg.vector_norm(u)             # tensor(5.)
torch.abs(u).sum()                      # tensor(7.)，L1
torch.linalg.norm(torch.ones(4, 9))     # tensor(6.)，Frobenius
torch.linalg.vector_norm(torch.arange(4))
# RuntimeError: linalg.vector_norm: Expected a floating point or complex tensor as input. Got Long
torch.norm(torch.ones(2, 3, 4))         # tensor(4.8990) = sqrt(24)，把所有元素当一个长向量
```

三个坑：范数函数不接受整型张量（上面第二段报错），建张量时给 `float32`；`torch.norm` 是历史接口，
对任意形状的张量都按“全部元素拉平求 $L_2$”处理，上面三轴张量得到 $\sqrt{24}$；torch 2.14 实测
`torch.norm` 没有弃用警告，但官方文档推荐 `torch.linalg.norm` 与 `torch.linalg.vector_norm`，
新代码用后者，含义更明确。第一版的 `X.norm().asscalar()` 对应
`torch.linalg.vector_norm(X).item()`。

深度学习里用 $L_2$ 范数的平方多于 $L_2$ 本身，因为不必开方，而且梯度形式简单：
v1 附录 A.1 给出 $\nabla_{\mathbf{x}}\|\mathbf{x}\|^2=2\mathbf{x}$。$L_1$ 范数的梯度是
$\mathrm{sign}(\mathbf{x})$，每个分量的梯度幅度固定为 1，不随偏差放大，所以 $L_1$ 对异常值不如
$L_2$ 敏感（v2 原文提到这一点）。

## 8. 广播在矩阵运算中的语义、按轴求和与拼接

广播规则本身在 B02 第 4 节，这里只看它在矩阵运算里的三种常见形态：

```python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
2 + A                                # 标量广播到每个元素，(5, 4)
A + torch.ones(4)                    # (4,) 按最后一维对齐，(5, 4)
A / A.sum(dim=1, keepdim=True)       # (5, 1) 广播成 (5, 4)，逐行归一化
torch.ones(3, 1) + torch.ones(1, 2)  # (3, 2)，两个方向同时广播
```

第三行是后面 softmax、批量归一化都会用到的写法：先沿某一轴降维得到每行或每列的统计量，
`keepdim=True` 保住轴的位置，再做广播除法。

按轴求和与拼接放在一起看形状更清楚：

```python
A.sum(dim=0).shape, A.sum(dim=1).shape        # (torch.Size([4]), torch.Size([5]))
torch.cat([A, A], dim=0).shape                # (10, 4)，轴 0 相加，轴 1 必须相等
torch.cat([A, A], dim=1).shape                # (5, 8)
torch.stack([A, A], dim=0).shape              # (2, 5, 4)，新增一个轴
```

`cat` 在已有轴上接长，要求其他轴长度一致；`stack` 新建一个轴，要求所有输入形状完全相同。
电子版对应的写法是 `np.concatenate([A, A], axis=0)`，第一版是 `nd.concat(A, A, dim=0)`。

## 自测题

1. `s = torch.tensor(3.0)` 与 `x = torch.arange(4)` 的 `dim()`、`shape`、`len()` 各是什么？`len(s)` 会怎样？
2. `A` 是 `(5, 4)` 的矩阵。`A.T` 的形状是什么？`X = torch.arange(24).reshape(2, 3, 4)` 时 `X.T` 的形状又是什么？后者与电子版 MXNet 的行为差别在哪，应该改用哪个函数？
3. `A` 是 `(5, 4)` 的矩阵，`C` 是 `(4, 3)`。`A * C` 和 `A @ C` 分别发生什么？`torch.dot(A, C)` 和 `torch.mv(A, C)` 呢？
4. 从 $c_{ij}=\sum_k a_{ik}b_{kj}$ 出发，说明为什么 $\mathbf{AB}$ 可以看作 $n$ 次矩阵-向量积的拼接。
5. `A` 是 `(5, 4)`。`A.sum(dim=0)`、`A.sum(dim=1)`、`A.sum(dim=1, keepdim=True)` 的形状各是什么？为什么 `A / A.sum(dim=1)` 报错而 `A / A.sum(dim=1, keepdim=True)` 正确？
6. 写出 $L_1$、$L_2$ 与 Frobenius 范数的定义，并说明 `torch.linalg.vector_norm` 用在 `torch.arange(4)` 上会发生什么、为什么。
7. 给定矩阵 `A`（形状 `(5, 4)`），写出用广播把每一列减去该列均值、每一列再除以该列标准差的表达式（不要求除零保护），并说明每一步的形状。
8. `torch.cat([A, A], dim=0)`、`torch.cat([A, A], dim=1)`、`torch.stack([A, A], dim=0)` 在 `A` 是 `(5, 4)` 时各得到什么形状？三者的区别是什么？

## 答案（做完再看）

1. `s` 是 `dim() == 0`、`shape == torch.Size([])`，`len(s)` 报 `TypeError: len() of a 0-d tensor`；`x` 是 `dim() == 1`、`shape == torch.Size([4])`、`len(x) == 4`。
2. `A.T` 是 `(4, 5)`。`X.T` 是 `(4, 3, 2)`，因为 torch 的 `.T` 对非二维张量翻转所有轴，且实测会抛弃用警告；电子版的 `np.transpose` 与 MXNet 的 `A.T` 在低维上一致，但按轴交换应当用 `X.transpose(0, 2)` 或 `X.permute(2, 0, 1)`。
3. `A * C` 触发广播规则，`(5, 4)` 与 `(4, 3)` 从右对齐后第 0 维 5 与 3 都不为 1，报形状错误；`A @ C` 正常得到 `(5, 3)`。`torch.dot(A, C)` 报 `1D tensors expected, but got 2D and 2D tensors`；`torch.mv(A, C)` 报 `vector + matrix @ vector expected, got 1, 2, 2`，它要求第二个参数是一维。
4. 固定 $j$，对每个 $i$ 有 $c_{ij}=\mathbf{a}_i^\top\mathbf{b}_j$，这正是 $\mathbf{A}$ 与 $\mathbf{B}$ 第 $j$ 列做矩阵-向量积得到的第 $i$ 个分量；$j$ 取遍 $n$ 个值就得到 $\mathbf{AB}$ 的全部列，所以整体上是 $n$ 次矩阵-向量积的拼接。
5. 分别是 `(4,)`、`(5,)`、`(5, 1)`。`A.sum(dim=1)` 得到 `(5,)`，广播时右对齐到最后一维与 4 冲突，报 `The size of tensor a (4) must match the size of tensor b (5) at non-singleton dimension 1`；`keepdim=True` 得到 `(5, 1)`，第 1 维长度为 1，可广播成 `(5, 4)`。
6. $\|\mathbf{x}\|_1=\sum_i|x_i|$，$\|\mathbf{x}\|_2=\sqrt{\sum_ix_i^2}$，$\|\mathbf{X}\|_F=\sqrt{\sum_{ij}x_{ij}^2}$。`torch.arange(4)` 是 `int64`，`torch.linalg.vector_norm` 报 `Expected a floating point or complex tensor as input. Got Long`，需要先转成浮点。
7. 形如 `(A - A.mean(dim=0, keepdim=True)) / A.std(dim=0, keepdim=True)`。`A.mean(dim=0, keepdim=True)` 是 `(1, 4)`，与 `(5, 4)` 广播成 `(5, 4)`；`std` 同理；除法也是逐元素广播。
8. 分别是 `(10, 4)`、`(5, 8)`、`(2, 5, 4)`。前两个在已有轴上接长，要求另一轴长度相等；`stack` 新建一个轴，要求输入形状完全相同，因此多出一个 2。
