# A0 环境与工作流

这一单元不涉及深度学习。目的是让你能顺畅地写代码、跑代码、看图、提交。
做完这一单元，你应该能自己回答"我的代码在哪、环境怎么用、报错去哪看"。

**讲义里不会给你可以直接粘贴的完整代码。** 给的是要求、骨架、**该读哪一段材料**和分级提示。
自己写出来才算过。

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

## 3. 先认识四件事

写第一个脚本之前，有四个 Python 的约定。都不难，但不认识就会处处卡。

### 3.1 import：用库之前先加载

```python
import torch
```

这一行执行之后，`torch` 这个名字才能在下面用。所有功能都挂在它下面，写成 `torch.xxx`。

和 C++ 的 `#include` 不一样：`#include` 是编译期把代码文本插进来，`import` 是运行时去加载一个
已经编译好的模块，加载完把模块对象绑到名字上。

也有只取一部分的写法：

```python
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
```

第二种里的 `as plt` 是起别名，社区惯例，为了少打字。后面会看到大量这种写法。

### 3.2 属性与方法：括号加不加，意思完全不同

```python
x.shape        # 属性，直接取一个值
x.numel()      # 方法，调用它，返回一个值
```

`x.shape` 不加括号，拿到的是"形状"这个属性本身。
`x.numel()` 加括号，是"调用 numel 这个方法"。

写成 `x.numel` 不报错，但你拿到的是方法对象而不是数字。这个错误很难看出来，
打印出来像 `<built-in method numel of Tensor object at 0x...>`。看到这种输出，就是漏了括号。

### 3.3 用缩进代替大括号

冒号加缩进就是代码块。缩进错了是语法错误，统一四空格，别混 Tab。

### 3.4 注释

`#` 到行尾。写"为什么这样写"，不写"这行在做什么"。

## 4. 该读什么

### 4.1 从官方教程入手，别从 API 文档入手

API 文档是给已经知道自己在找什么的人查的。你现在需要的是有人按顺序讲一遍的材料。

PyTorch 官方的入门系列在这里：

<https://docs.pytorch.org/tutorials/beginner/basics/intro.html>

那一页自己写了读法（原文）：

> If you're familiar with other deep learning frameworks, check out the 0. Quickstart first
> to quickly familiarize yourself with PyTorch's API. If you're new to deep learning frameworks,
> head right into the first section of our step-by-step guide: 1. Tensors.

你没有别的框架经验，所以**跳过 0. Quickstart，从 1. Tensors 读起**：

<https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html>

（Quickstart 会把下载数据、建模型、训练、保存整套流程一次过完，现在看信息量太大。
B06 学完线性回归之后回来读它正合适。）

Tensors 那一页里和 A0 有关的小节：

| 小节 | 讲什么 | 用在 |
|---|---|---|
| Initializing a Tensor | 造张量的四种方式：从数据、从 NumPy、从另一个张量、用随机数或常量 | 任务一第 3 步 |
| Attributes of a Tensor | 三个属性：`shape`、`dtype`、`device` | 任务一第 4 步 |
| Operations on Tensors 开头几段 | 张量默认在 CPU；用 `.to()` 搬到加速器 | 任务一第 5 步 |
| Bridge with NumPy | 张量与 NumPy 数组共享内存，`.numpy()` | 任务二 |

再往后的 Datasets & DataLoaders、Transforms、Build Model、Autograd、Optimization、
Save & Load Model 现在不用看，那对应 B06 到 B07。

一处提醒：教程里搬设备写的是 `torch.accelerator`，那是新 API，泛指 CUDA / MPS / XPU 各类
加速器。本机只有 NVIDIA 显卡，写 `torch.cuda` 更直接，两种都对。

### 4.2 教程没讲到的三个，查 API 文档

Tensors 那页没有 `arange`、`reshape`、`numel`。这三个查 API：

| 查什么 | 去哪 | 重点看 |
|---|---|---|
| 怎么造一串连续数字 | [torch.arange](https://pytorch.org/docs/stable/generated/torch.arange.html) | Parameters 里 `dtype` 的默认值 |
| 怎么改形状 | [Tensor.reshape](https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html) | 参数能不能写成一个 tuple |
| 怎么看元素个数 | [Tensor.numel](https://pytorch.org/docs/stable/generated/torch.Tensor.numel.html) | 返回值是什么类型 |

API 文档的读法：先看签名和第一句话，再看 Parameters 表。表里写了默认值的参数可以不传。

### 4.3 任务二的材料

- `plt.plot`、`plt.xlabel`、`plt.ylabel`：看
  [Pyplot tutorial](https://matplotlib.org/stable/tutorials/pyplot.html) 的
  **Introduction to pyplot** 那一节。再往后的 "Formatting the style of your plot" 讲颜色和线型，
  现在用不上
- `plt.figure`：[figure 文档](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html)，
  只需要 `figsize`
- `plt.savefig`：[savefig 文档](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)，
  只需要 `fname` 和 `dpi`

## 5. 任务一：确认环境活着

写 `work/a0/hello_tensor.py`，跑出来的输出和下面的"期望输出"对得上。

### 要求

1. 打印 torch 的版本号
2. 打印 CUDA 是否可用
3. 造一个 3 行 4 列、元素类型 float32、内容是 0 到 11 的张量，绑到名字 `x`
4. 打印 `x` 的内容、`x` 的形状、`x` 的元素个数
5. 如果 CUDA 可用：把 `x` 搬到显卡上，打印它现在在哪个设备；在显卡上乘 2，再搬回 CPU 打印结果

### 骨架

`____` 是你要填的。先读第 4.1、4.2 节的材料，再回来填。

```python
import torch

print("torch 版本：", ____)
print("CUDA 可用：", ____)

# 逐步来：先造一维的，再变形
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

### 提示（卡住再看，从上往下）

<details><summary>提示 1：版本号与 CUDA</summary>

版本号是 torch 模块自带的一个属性，名字两边各两条下划线。

CUDA 能不能用，是 `torch.cuda` 下面一个函数返回的布尔值，函数名是 `is_available`，要加括号。

按第 3.2 节的原则判断：你要的是"值"还是"调用结果"。

</details>

<details><summary>提示 2：造张量</summary>

`torch.arange(12)` 得到 0 到 11 的一维张量，**默认是整数**。
要浮点得传 `dtype=torch.float32`，这是关键字参数。

变形用 `.reshape(3, 4)`。为什么必须指定 float32，B02 会讲，现在照做。

</details>

<details><summary>提示 3：形状与元素个数</summary>

一个不加括号，一个加括号。去 Tensors 教程的 **Attributes of a Tensor** 一节，
对照 `shape` 和 `dtype` 的写法，再看 `numel` 的 API 文档。

</details>

<details><summary>提示 4：搬显卡</summary>

Tensors 教程的 **Operations on Tensors** 开头有 `.to()` 的用法。
在它基础上把"当前加速器"换成 `"cuda"`。

它现在在哪，看 `.device` 属性——这个属性在 **Attributes of a Tensor** 一节里出现过。

搬回内存用 `.cpu()`。

注意 `(g * 2).cpu()` 里的括号：先算乘法，再搬回来，最后打印。

</details>

## 6. 任务二：把图画出来

写 `work/a0/sin.py`，在 `work/a0/sin.png` 生成一张 sin 曲线图。

### 要求

1. 在 0 到 2π 之间均匀取 200 个点，作为横坐标
2. 对每个点求正弦，作为纵坐标
3. 画折线，给两个轴加标签
4. 存成 PNG 文件

### 骨架

```python
import matplotlib
____                    # 这一行必须在下一行之前

import matplotlib.pyplot as plt
import torch

x = ____                # 提示：torch.linspace

plt.figure(figsize=(5, 3))
plt.plot(____, ____)    # 两个参数：横坐标、纵坐标
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.savefig("work/a0/sin.png", dpi=120)
print("已保存 work/a0/sin.png")
```

### 那个空行是干什么的

这台机器没有显示器。matplotlib 默认会去开一个窗口，开不了就报错或者卡住。
要换成 `Agg` 后端，它只往内存里的位图渲染，最后由 `savefig` 写进文件。

后端必须在 `import matplotlib.pyplot` **之前**选定。pyplot 在导入的那一刻就把后端定下来，
导入之后再改不生效。要填的那个词是后端名字，三个字母。

### 提示

<details><summary>提示 1：横坐标</summary>

`torch.linspace(起点, 终点, 点的个数)`，linspace 是 "linear space"。
它没出现在第 4 节的表里，可以顺手查一下它的 API 文档。

2π 可以写 `2 * 3.14159`，也可以 `import math` 之后用 `math.pi`。

</details>

<details><summary>提示 2：纵坐标</summary>

有个逐元素求正弦的函数，叫 `torch.sin`，返回同样形状的张量。

</details>

<details><summary>提示 3：为什么不能把张量直接交给 plot</summary>

matplotlib 不认识 torch 张量，只认 numpy 数组。
Tensor 上有个方法能转过去，看 Tensors 教程的 **Bridge with NumPy** 一节。

转换要求张量在 CPU 上、且不连着计算图。这里两个条件都满足，不会有问题。

</details>

### 跑起来

```bash
just run work/a0/hello_tensor.py
just run work/a0/sin.py
```

`just run` 只是 `python3 <路径>` 的简写。

图存好之后，在 VSCodium 的远程文件树里点开 `work/a0/sin.png` 就能看。

## 7. 数据集从哪来

`d2l` 包默认的数据源在国内不通，不要照着书上的下载代码抄。
可用的路子是用 `torchvision.datasets`，具体到 B07 那一章会给能跑通的代码。

数据统一放 `data/`（已在 `.gitignore` 里，不会提交）。

## 8. git

开发机上的仓库是权威副本，也是要公开的学习记录。每完成一个小任务提交一次，别攒着：

```bash
git status
git add work/a0/hello_tensor.py
git commit -m "a0: 跑通第一个张量脚本"
```

历史用 `git log --oneline`，撤未提交的改动用 `git restore <文件>`。

## 9. 常用命令速查

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
8. `just run work/a0/sin.py` 展开成什么命令？

<details>
<summary>做完再看：答案</summary>

1. import 是运行时加载模块，加载完把模块对象绑到名字上，之后才能通过这个名字访问里面的东西。
   `#include` 是编译期把源文本插进来，两者发生的时机和机制都不同。
2. 要一个已经存好的值就用属性（不加括号）；要执行一段逻辑拿返回值就调用方法（加括号）。
   拿不准就看文档，或者打印出来看：是 `<built-in method ...>` 就是漏了括号。
3. `torch.arange(12)` 默认 `int64`。浮点运算和梯度要求浮点类型，整数张量送进线性层会报 dtype 不匹配。
4. 没变。`.to()` 返回新张量，`x` 还是原来那个在 CPU 上的。
5. `.cpu()` 作用于 `g * 2` 的结果。不加括号会改变运算顺序。
6. pyplot 在导入时就把后端定下来并创建相关对象，之后再调 `use()` 不生效。
7. matplotlib 只认 numpy 数组，torch 张量要先 `.numpy()` 转过去。
8. `python3 work/a0/sin.py`。

</details>
