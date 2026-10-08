# A0 环境与工作流

这一单元不涉及深度学习。目的是让你能在这台机器上顺畅地写代码、跑代码、看图、提交。
做完这一单元，你应该能自己回答"我的代码在哪、环境怎么进、报错去哪看"。

## 1. 机器和目录

代码全部在远程：

```
ssh -p 2222 Qaaxaap@192.168.1.155
cd ~/Projects/d2l
```

远程配置：Arch Linux，i9-14900K（32 线程），62G 内存，RTX 4070 SUPER（12G 显存）。

仓库目录分工：

| 目录 | 谁写 |
|---|---|
| `work/` | **你**。你的实现放这，AI 不会覆盖 |
| `notes/` | AI 写的讲义 |
| `exercises/` | AI 出的题 |
| `tools/` | 验收脚本、同步脚本 |
| `refs/` | 参考资料，做题时别看 |

## 2. 环境分层

这台机器上 Python 环境分三层，搞清每层管什么，后面的报错才好定位：

| 层 | 谁提供 | 内容 |
|---|---|---|
| 解释器 | Arch pacman | `/usr/bin/python3`，版本 3.14 |
| 科学计算包 | Arch pacman | `python-pytorch-cuda`（torch 2.14.0）、`python-numpy`、`python-matplotlib`、`python-pandas`、`python-tqdm`、`python-requests` |
| 工具链 | nix flake | `uv`、`just`、`git` |

先看一眼实际状态：

```bash
cd ~/Projects/d2l
just env
```

应该看到 torch 2.14.0、CUDA 可用、RTX 4070 SUPER。

**为什么不用 conda**：conda 会往 shell 启动脚本里塞钩子，每次开终端都要跑一遍初始化。
这里解释器和包由 pacman 管，nix 只补工具链，你在任何目录 `python3 xxx.py` 都能跑。

**需要额外包时**（比如第二版官方的 `d2l` 包）才用到 uv：

```bash
just setup          # 第一次：建 .venv，带 --system-site-packages，能看到系统包
uv add d2l          # 装进 .venv，同时写进 pyproject.toml
uv run python work/a0/hello_tensor.py    # 用 .venv 跑
```

注意 `python3 xxx.py` 看不到 `.venv` 里的包，`uv run python xxx.py` 才看得到。

## 3. 写第一个脚本

新建 `work/a0/hello_tensor.py`：

```python
import torch

print("torch 版本：", torch.__version__)
print("CUDA 可用：", torch.cuda.is_available())

x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print("x =\n", x)
print("形状：", x.shape, "元素个数：", x.numel())

if torch.cuda.is_available():
    g = x.to("cuda")
    print("搬到显卡上：", g.device)
    print("乘 2 再搬回来：", (g * 2).cpu())
```

跑：

```bash
just run work/a0/hello_tensor.py
```

等价于 `python3 work/a0/hello_tensor.py`。`just run` 只是省得你每次打全路径。

## 4. 怎么看图

远程没有显示器，`plt.show()` 会卡住或者报错。**每个画图的脚本开头都要切到 Agg 后端**：

```python
import matplotlib
matplotlib.use("Agg")          # 必须在 import pyplot 之前
import matplotlib.pyplot as plt

import torch

x = torch.linspace(0, 2 * 3.14159, 200)
plt.figure(figsize=(5, 3))
plt.plot(x.numpy(), torch.sin(x).numpy())
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.savefig("work/a0/sin.png", dpi=120)   # 存文件，不 show
print("已保存 work/a0/sin.png")
```

图存成 PNG 之后，两种看法：

1. 本地终端拉下来看：

```bash
scp -P 2222 Qaaxaap@192.168.1.155:~/Projects/d2l/work/a0/sin.png /tmp/
```

2. 用 VSCodium 的 Remote-SSH 连上去，直接点开文件。

（本地终端如果支持图片协议，装个 `chafa` 或 `timg` 可以直接在终端里看图，需要的话说一声。）

## 5. 数据集从哪来

`d2l` 包默认的数据源在国内不通（`ap-northeast-1.d2l.ai` 超时，
`d2l-data.s3-accelerate.amazonaws.com` 返回 403），不要照着书上的下载代码抄。
可用的路子是用 `torchvision.datasets`，具体到 B07 那一章会给能跑通的代码。

数据统一放 `data/`（已在 `.gitignore` 里，不会提交）。

## 6. git

远程仓库是唯一权威副本。每完成一个小任务提交一次，别攒着：

```bash
cd ~/Projects/d2l
git status
git add work/a0/hello_tensor.py
git commit -m "a0: 跑通第一个张量脚本"
```

查看历史：`git log --oneline`。撤掉还没提交的改动：`git restore <文件>`。

## 7. 常用命令速查

| 想干什么 | 命令 |
|---|---|
| 看环境 | `just env` |
| 跑脚本 | `just run work/a0/hello_tensor.py` |
| 跑某单元验收 | `just check b02` |
| 进 nix 工具环境 | `nix develop`（可选） |
| 装额外的包 | `just setup` 然后 `uv add <包>` |
| 让 AI 看到你的改动 | 跟 AI 说一声，它自己拉 |
| 看讲义 | `less notes/a0-setup.md` |

## 自测

1. `python3` 指向哪里？为什么不指向 `/nix/store` 里的某个 Python？
2. 三层环境各自管什么？我要装 `scikit-learn`，走哪条路？
3. 为什么画图脚本必须写 `matplotlib.use("Agg")`，而且必须写在 `import matplotlib.pyplot` 之前？
4. `just run work/a0/hello_tensor.py` 展开成什么命令？
5. `python3 xxx.py` 和 `uv run python xxx.py` 有什么区别？
6. 你在远程改了 `work/a0/hello_tensor.py`，怎么让本地的 AI 看到？

<details>
<summary>做完再看：答案</summary>

1. 指向 `/usr/bin/python3`（Arch 的 3.14）。nix flake 里没有提供 python，
   免得遮蔽系统解释器，也免得和系统里那份编译好的 torch C 扩展对不上。
2. 解释器归 pacman，科学计算包归 pacman，工具链归 nix。装 `scikit-learn`
   先看 pacman 有没有（`pacman -Ss python-scikit-learn`），有就用 pacman；
   没有就 `uv add scikit-learn` 装进项目 `.venv`。
3. 远程没有 X/Wayland 显示服务，默认后端（TkAgg/QtAgg）找不到显示目标。
   后端必须在 pyplot 导入时选定，导入之后再改不生效。
4. `python3 work/a0/hello_tensor.py`。
5. 前者用系统解释器，只看得到 pacman 装的包；后者用项目 `.venv`，
   既看得到 `.venv` 里的包，也看得到系统包。
6. 告诉 AI 一声，AI 用 `tools/sync.sh pull` 拉回去看。

</details>
