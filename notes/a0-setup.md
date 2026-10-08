# A0 环境与工作流

这一单元不涉及深度学习。目的是让你能顺畅地写代码、跑代码、看图、提交。

**讲义里不会给你可以直接粘贴的完整代码**，但会**把每个 API 直接讲清楚**，
不需要你去别处找知识。你读完讲义，凭理解把代码写出来。

---

## 1. 代码在哪

所有代码在开发机上，登录后进入仓库目录即可：

```bash
cd <仓库目录>
```

机器规格：Linux，i9-14900K（32 线程），62G 内存，RTX 4070 SUPER（12G 显存）。
连接方式和路径你自己清楚，仓库里不写。

仓库目录分工：

| 目录 | 谁写 |
|---|---|
| `work/` | **你**。你的实现放这，AI 不会覆盖 |
| `notes/` | AI 写的讲义 |
| `exercises/` | AI 出的题 |
| `tools/` | 验收脚本、同步脚本 |

`work/` 现在还不存在：

```bash
mkdir -p work/a0
```

## 2. 环境分层

Python 环境分三层，搞清每层管什么，后面的报错才好定位：

| 层 | 谁提供 | 内容 |
|---|---|---|
| 解释器 | 系统包管理器 | `/usr/bin/python3`，版本 3.14 |
| 科学计算包 | 系统包管理器 | `python-pytorch-cuda`（torch 2.14.0）、`python-numpy`、`python-matplotlib`、`python-pandas`、`python-tqdm`、`python-requests` |
| 工具链 | nix flake | `uv`、`just`、`git` |

先看一眼实际状态：

```bash
just env
```

`just` 是任务执行器，`just env` 会跑 `tools/env.py` 那个脚本。
别的命令用 `just --list` 看。

应该看到 torch 2.14.0、CUDA 可用、RTX 4070 SUPER。

**为什么不用 conda**：conda 会往 shell 启动脚本里塞钩子，每次开终端都要跑一遍初始化。
这里解释器和包由系统包管理器管，nix 只补工具链，在任何目录 `python3 xxx.py` 都能跑。

**需要额外包时**（比如第二版官方的 `d2l` 包）才用到 uv：

```bash
just setup          # 第一次：建 .venv，带 --system-site-packages，能看到系统包
uv add d2l          # 装进 .venv，同时写进 pyproject.toml
uv run python work/a0/hello_tensor.py    # 用 .venv 跑
```

注意 `python3 xxx.py` 看不到 `.venv` 里的包，`uv run python xxx.py` 才看得到。

## 3. Python 的四个约定

### 3.1 import：用库之前先加载

```python
import torch
```

这一行执行之后，`torch` 这个名字才能在下面用，所有功能都挂在它下面，写成 `torch.xxx`。

和 C++ 的 `#include` 不一样：`#include` 是编译期把代码文本插进来，`import` 是运行时去加载一个
已经编译好的模块，加载完把模块对象绑到名字上。

也可以只取一部分，或者起别名：

```python
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt      # as plt 是社区惯例，为了少打字
```

### 3.2 属性与方法

```python
x.shape        # 没有括号
x.numel()      # 有括号
```

两种东西，用法不同：

- **属性**是对象上已经存好的数据，直接取，不加括号
- **方法**是一段可以执行的逻辑，加括号才是"调用它"

写成 `x.numel` 不报错，但你拿到的是方法对象本身而不是数字。打印出来长这样：

```
<built-in method numel of Tensor object at 0x7f...>
```

看到这种输出，就是漏了括号。

哪些是属性哪些是方法，只能记。张量上常用的：

| 属性（不加括号） | 方法（加括号） |
|---|---|
| `shape`、`dtype`、`device`、`T` | `numel()`、`size()`、`item()`、`tolist()`、`reshape()`、`to()` |

### 3.3 用缩进代替大括号

冒号加缩进就是代码块。缩进错了是语法错误，统一四空格，别混 Tab。

### 3.4 注释

`#` 到行尾。写"为什么这样写"，不写"这行在做什么"。

## 4. 你要用到的 API

这一节是任务一和任务二需要的全部东西。**我在这里讲清楚，不用你去别处查。**
想深入的时候再看第 7 节的链接。

### 4.1 版本号存在 `__version__` 里

```python
print("torch 版本：", torch.__version__)
```

`print` 是 Python 内置函数，接受任意多个参数，用逗号隔开，输出时自动插空格。

Python 的库有一个通行约定：**用 `__version__` 这个属性暴露版本号字符串**。
torch、numpy、matplotlib 全都遵守。名字两边各两条下划线的写法叫 dunder，
是 Python 里"语言或框架约定的特殊名字"的标记，A1 会专门讲。
现在只要记住：torch 的版本号是 `torch.__version__`，一个字符串。

### 4.2 判断显卡能不能用

```python
torch.cuda.is_available()
```

`torch.cuda` 是 torch 里管 NVIDIA 显卡的子模块。`is_available` 是它下面的一个函数，
**返回布尔值**，告诉你这台机器的 torch 能不能用上显卡。

它是个函数，所以要写 `is_available()`。这是第 3.2 节那条规则的第一个实例。

顺便说一句命名：CUDA 是 NVIDIA 的并行计算平台。torch 里凡是只管 N 卡的功能都放在
`torch.cuda` 下面。PyTorch 后来加了个更通用的 `torch.accelerator`，泛指各类加速器，
教程里偶尔会见到，功能等价。本机只有 N 卡，用 `torch.cuda` 更直接。

### 4.3 造一个张量

**张量**是这门课的主角，可以理解成"能放在显卡上算的多维数组"。

```python
torch.arange(12)
```

`arange` 是 "array range" 的缩写，`arange(12)` 给出从 0 到 11 的一维张量，**不含 12**，
和 Python 的 `range` 一样是左闭右开。默认元素类型是 **int64**，也就是整数。

要浮点得显式指定：

```python
torch.arange(12, dtype=torch.float32)
```

`dtype=` 是**关键字参数**：函数签名里写死的参数名，传的时候带上名字。
好处是不用记住参数顺序。Python 里函数可以有任意多个关键字参数。

为什么这里必须指定浮点，B02 会讲清楚，现在照做就行。

### 4.4 改形状

```python
x.reshape(3, 4)
```

把一个 12 个元素的一维张量排成 3 行 4 列。元素总数必须对得上，否则报错。
也可以写 `x.reshape((3, 4))`，两种写法等价。

### 4.5 看形状和元素个数

```python
x.shape      # torch.Size([3, 4])，可以当元组用
x.numel()    # 12，int
```

`shape` 是属性，不加括号；`numel` 是方法，要加括号。`numel` 是 "number of elements" 的缩写。

### 4.6 把张量搬到显卡

张量默认建在内存（CPU）上。搬到显卡：

```python
g = x.to("cuda")
```

`.to("cuda")` **返回一个新张量**，原来那个 `x` 还在 CPU 上，没动。

新张量在哪个设备上，用 `.device` 属性看，输出形如 `cuda:0`（0 是显卡编号）。

搬回内存：

```python
g.cpu()
```

**为什么必须搬回来**：显卡上的张量 Python 打印不了，`print` 会报错。
要看内容就得先 `.cpu()`。

### 4.7 画图

四个调用：

```python
plt.figure(figsize=(5, 3))    # 新建画布，figsize 单位是英寸
plt.plot(x, y)                # 画折线，两个参数是横坐标和纵坐标
plt.xlabel("x")               # 横轴标签
plt.ylabel("sin(x)")          # 纵轴标签
plt.savefig("out.png", dpi=120)   # 存成 PNG，dpi 是每英寸像素数
```

**matplotlib 只认 numpy 数组，不认 torch 张量。** 传进去之前要转换：

```python
x.numpy()
```

`.numpy()` 是张量的方法，返回对应的 numpy 数组。要求张量在 CPU 上、且不连着计算图，
这两个条件在 A0 里都满足。

生成横坐标用：

```python
torch.linspace(0, 2 * 3.14159, 200)
```

`linspace` 是 "linear space"，在起点和终点之间**均匀取 N 个点**（含两端）。
和 `arange` 的区别：`arange` 按步长，`linspace` 按点的个数。

求正弦：

```python
torch.sin(x)
```

逐元素求正弦，返回形状一样的张量。

### 4.8 为什么画图前要设 `Agg` 后端

**后端**指的是 matplotlib 用什么把图画出来。默认后端会去开一个窗口显示。
这台机器没有显示器，开窗口会报错或卡住。

`Agg` 后端只往内存里的位图渲染，不显示，由 `savefig` 写进文件。

```python
import matplotlib
matplotlib.use("Agg")
```

**必须在 `import matplotlib.pyplot` 之前调用**。因为 pyplot 在导入的那一刻就把后端定下来
并创建了相关对象，之后再改不生效。

## 5. 任务一：确认环境活着

写 `work/a0/hello_tensor.py`。

### 要求

1. 打印 torch 的版本号
2. 打印 CUDA 是否可用
3. 造一个 3 行 4 列、元素类型 float32、内容是 0 到 11 的张量，绑到名字 `x`
4. 打印 `x` 的内容、`x` 的形状、`x` 的元素个数
5. 如果 CUDA 可用：把 `x` 搬到显卡上，打印它现在在哪个设备；在显卡上乘 2，再搬回 CPU 打印结果

第 4 节的 4.1 到 4.6 已经把这五步需要的东西全讲了，直接写。

### 骨架

`____` 是你要填的。填不出来回去看第 4 节对应小节，别看提示。

```python
import torch

print("torch 版本：", ____)
print("CUDA 可用：", ____)

x = ____

print("x =\n", x)
print("形状：", ____, "元素个数：", ____)

if ____:
    g = ____
    print("搬到显卡上：", ____)
    print("乘 2 再搬回来：", ____)
```

### 期望输出

版本号和你可能略有差别，其余应该一致。

```
torch 版本： 2.14.0
CUDA 可用： True
x =
 tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])
形状： torch.Size([3, 4]) 元素个数： 12
搬到显卡上： cuda:0
乘 2 再搬回来： tensor([[ 0.,  2.,  4.,  6.],
        [ 8., 10., 12., 14.],
        [16., 18., 20., 22.]])
```

## 6. 任务二：把图画出来

写 `work/a0/sin.py`，在 `work/a0/sin.png` 生成一张 sin 曲线图。

### 要求

1. 在 0 到 2π 之间均匀取 200 个点，作为横坐标
2. 对每个点求正弦，作为纵坐标
3. 画折线，给两个轴加标签
4. 存成 PNG 文件

第 4 节的 4.7 和 4.8 讲了全部需要的东西。

### 骨架

```python
import matplotlib
____                    # 4.8 讲了这里填什么、为什么必须在下一行之前

import matplotlib.pyplot as plt
import torch

x = ____                # 横坐标，见 4.7

plt.figure(figsize=(5, 3))
plt.plot(____, ____)    # 注意 4.7 最后那条：不能直接把张量传进去
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.savefig("work/a0/sin.png", dpi=120)
print("已保存 work/a0/sin.png")
```

### 跑起来

```bash
just run work/a0/hello_tensor.py
just run work/a0/sin.py
```

`just run` 只是 `python3 <路径>` 的简写。

图存好之后，在 VSCodium 的远程文件树里点开 `work/a0/sin.png` 就能看。

## 7. 想深入的时候看这些

讲义是自足的，这些链接是给"想多知道一点"用的，不是必经之路。

| 想了解 | 去哪 |
|---|---|
| 张量的完整介绍 | [Learn the Basics 系列](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) 的 **1. Tensors** 一页。它的 Initializing、Attributes、Bridge with NumPy 三节和第 4 节讲的是同一批东西，可以对照 |
| 整个训练流程长什么样 | 同系列的 **0. Quickstart**。现在看信息量太大，B06 学完线性回归回来读正合适 |
| `arange` 的完整参数 | [API 文档](https://pytorch.org/docs/stable/generated/torch.arange.html) |
| 画图的更多用法 | [Pyplot tutorial](https://matplotlib.org/stable/tutorials/pyplot.html)，读到 "Formatting the style of your plot" 为止就够 A0 用了 |
| Python 语法本身 | [Python 官方中文教程](https://docs.python.org/zh-cn/3/tutorial/) |

## 8. 数据集从哪来

`d2l` 包默认的数据源在国内不通，不要照着书上的下载代码抄。
可用的路子是用 `torchvision.datasets`，具体到 B07 那一章会给能跑通的代码。

数据统一放 `data/`（已在 `.gitignore` 里，不会提交）。

## 9. git

开发机上的仓库是权威副本，也是要公开的学习记录。每完成一个小任务提交一次，别攒着：

```bash
git status
git add work/a0/hello_tensor.py
git commit -m "a0: 跑通第一个张量脚本"
```

历史用 `git log --oneline`，撤未提交的改动用 `git restore <文件>`。

## 10. 常用命令速查

| 想干什么 | 命令 |
|---|---|
| 看环境 | `just env` |
| 跑脚本 | `just run work/a0/hello_tensor.py` |
| 看有哪些命令 | `just --list` |
| 跑某单元验收 | `just check b02` |
| 进 nix 工具环境 | `nix develop`（可选） |
| 装额外的包 | `just setup` 然后 `uv add <包>` |
| 让 AI 看到你的改动 | 跟 AI 说一声，它自己拉 |
| 看讲义 | `notes/a0-setup.md` |

## 自测

写完两个脚本之后关掉讲义答。答不上来的回去看对应小节。

1. `import torch` 之后才能写 `torch.xxx`，为什么？它和 C++ 的 `#include` 有什么不同？
2. `x.shape` 和 `x.numel()` 一个不加括号一个加，怎么判断该不该加？
3. `torch.arange(12)` 造出来的张量是什么类型？为什么这里要显式写 `dtype=torch.float32`？
4. `x.to("cuda")` 之后，`x` 本身变了吗？
5. 为什么 `(g * 2).cpu()` 要加括号？
6. 画图那行 `matplotlib.use("Agg")` 为什么必须写在 `import matplotlib.pyplot` 之前？
7. 为什么不能把 torch 张量直接传给 `plt.plot`？
8. `torch.arange` 和 `torch.linspace` 有什么区别？
9. `just run work/a0/sin.py` 展开成什么命令？

<details>
<summary>做完再看：答案</summary>

1. import 是运行时加载模块，加载完把模块对象绑到名字上，之后才能通过这个名字访问里面的东西。
   `#include` 是编译期把源文本插进来，两者发生的时机和机制都不同。
2. 要一个已经存好的值就用属性（不加括号）；要执行一段逻辑拿返回值就调用方法（加括号）。
   拿不准就看打印出来是不是 `<built-in method ...>`。
3. `torch.arange(12)` 默认 `int64`。浮点运算和梯度要求浮点类型，整数张量送进线性层会报 dtype 不匹配。
4. 没变。`.to()` 返回新张量，`x` 还是原来那个在 CPU 上的。
5. `.cpu()` 作用于 `g * 2` 的结果。不加括号会改变运算顺序。
6. pyplot 在导入时就把后端定下来并创建相关对象，之后再调 `use()` 不生效。
7. matplotlib 只认 numpy 数组，torch 张量要先 `.numpy()` 转过去。
8. `arange` 按**步长**生成（默认步长 1），`linspace` 按**点的个数**在两端之间均匀取。
9. `python3 work/a0/sin.py`。

</details>
