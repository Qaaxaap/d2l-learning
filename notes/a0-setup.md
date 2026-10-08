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
不用在本地写代码再传过去，直接在远程写。

仓库目录分工：

| 目录 | 谁写 |
|---|---|
| `work/` | **你**。你的实现放这，AI 不会覆盖 |
| `notes/` | AI 写的讲义 |
| `exercises/` | AI 出的题 |
| `tools/` | 验收脚本、同步脚本 |
| `refs/` | 参考资料，做题时别看 |

## 2. 进环境

```bash
cd ~/Projects/d2l
nix develop
```

进去之后提示符会多一行 `d2l devShell: Python 3.12.15 | uv 0.12.22`。
这时候 `python`、`uv`、`make` 都能用了。退出敲 `exit` 或者 `Ctrl-D`。

不用 `conda activate`，不用改 `.zshrc`，不拖慢 shell 启动。环境是 `flake.nix` 声明的，
nix 把它整个装进 `/nix/store`，跟你系统里的 Python 完全隔离。

第一次跑 `nix develop` 会下载依赖，之后是秒进（除非改了 `flake.nix`）。

### 依赖装在哪

Python 的第三方包（torch、matplotlib 等）不在 nix store 里，在**项目内**的 `.venv/`：

```
~/Projects/d2l/.venv/
```

它由 `pyproject.toml` 声明、`uv sync` 生成。分工是：nix 管"Python 解释器本身和系统库"，
uv 管"Python 包"。这么做是因为 PyTorch 的 CUDA wheel 走 PyPI 最省事，
而解释器用 nix 管能保证版本固定。

| 想干什么 | 命令 |
|---|---|
| 装新包 | `uv add <包名>`（会同时写进 `pyproject.toml`） |
| 按 `pyproject.toml` 同步环境 | `uv sync` |
| 看装了哪些包 | `uv pip list` |
| 直接跑一个脚本而不进 shell | `uv run python xxx.py` |

## 3. 写第一个脚本

`work/a0/hello_tensor.py`：

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
make run F=work/a0/hello_tensor.py
```

等价于 `python work/a0/hello_tensor.py`。`make run` 只是省得你每次打全路径。

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

（如果你本地终端支持图片协议，装个 `chafa` 或 `timg` 可以在终端里直接看，需要的话说一声。）

## 5. 数据集从哪来

`d2l` 包默认的数据源在国内不通，不要照着书上的下载代码抄。可用的走 `torchvision`：

- Fashion-MNIST、CIFAR-10 用 `torchvision.datasets`，它会从可达的镜像拉。
- 具体到 B07 那一章会给能跑通的下载代码。

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

## 7. nvim 最小操作

（待补：等确认你 nvim 的配置情况）

## 自测

1. `nix develop` 之后，`which python` 指向哪里？为什么不指向 `/usr/bin/python3`？
2. `.venv/` 和 `/nix/store` 各自管什么？如果我要装 `scikit-learn`，用哪条命令？
3. 为什么画图脚本必须写 `matplotlib.use("Agg")`，而且必须写在 `import matplotlib.pyplot` 之前？
4. `make run F=work/a0/hello_tensor.py` 展开成什么命令？
5. 你在远程改了 `work/a0/hello_tensor.py`，怎么让本地的 AI 看到？

<details>
<summary>做完再看：答案</summary>

1. 指向 `/nix/store/...-python-3.12.15/bin/python3`。因为 `nix develop` 把 nix 环境注入 PATH 最前面，
   系统的 `/usr/bin/python3` 是 Arch 的 3.14，不在环境里。
2. `.venv/` 管 Python 包，`/nix/store` 管解释器和系统库。装包用 `uv add scikit-learn`。
3. 远程没有 X/Wayland 显示服务，默认后端（TkAgg/QtAgg）没有可用的显示目标。
   必须在 pyplot 导入时选定后端，导入之后再改不生效。
4. `python work/a0/hello_tensor.py`。
5. 告诉 AI 一声，AI 用 `tools/sync.sh pull` 拉回去看。

</details>
