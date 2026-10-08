# d2l 学习仓库

《动手学深度学习》**第一版纸质书** + **PyTorch** 实现的学习记录。
公开仓库：<https://github.com/Qaaxaap/d2l-learning>

契约见 [PLAN.md](PLAN.md)，私教（AI）的工作规则见 [AGENTS.md](AGENTS.md)，
进度见 [PROGRESS.md](PROGRESS.md)，每次交互的记录见 [log/](log/)。

权威副本在远程机器 `Qaaxaap@192.168.1.155:2222` 的 `~/Projects/d2l`，
本地 `/home/Qaaxaap/Projects/d2l-tutor` 是 AI 的写作区，用 `tools/sync.sh` 同步。

## 目录

| 路径 | 谁写 | 内容 |
|---|---|---|
| `log/` | AI | 每次交互的记录。格式契约见 [log/README.md](log/README.md) |
| `notes/` | AI | 讲义。每章一份，含公式推导、MXNet→PyTorch 对照、易错点 |
| `exercises/<单元>/` | AI | `READING.md` 阅读任务书、`oral.md` 口试题、`code.md` 代码题 |
| `solutions/` | AI | 参考答案 |
| `work/<单元>/` | **你** | 你的实现。这是你的地盘，AI 不会覆盖 |
| `tools/` | AI | 验收断言、同步脚本 |
| `PROGRESS.md` | AI | 进度、掌握度、错题本、复习队列 |

## 开始

```bash
ssh -p 2222 Qaaxaap@192.168.1.155
cd ~/Projects/d2l
just env                    # 看解释器和各包版本、CUDA 是否可用
just run work/a0/hello_tensor.py
```

跑脚本不需要先进任何环境。系统那份 `python3`（Arch 的 3.14）已经装好
torch 2.14.0、torchvision、numpy、matplotlib、pandas、tqdm、requests，CUDA 实测可用。
`nix develop` 是可选的，只为拿到固定版本的 `uv` / `just` / `git`：

```bash
nix develop                 # 可选
```

## 环境设计

原则：**能复用系统包就复用**，这样同一份代码在别的项目、别的目录下也能直接跑。

| 层 | 谁提供 | 内容 |
|---|---|---|
| 解释器 | Arch pacman | `/usr/bin/python3`（3.14） |
| 科学计算包 | Arch pacman | `python-pytorch-cuda`、`python-numpy`、`python-matplotlib`、`python-pandas` 等 |
| 工具链 | nix flake | `uv`、`just`、`git`，版本锁在 `flake.lock` |
| 额外 Python 包 | uv（按需） | 装进项目内 `.venv`，带 `--system-site-packages`，和系统包共存 |

平时就是 `python3 xxx.py`。只有当某个包系统里没有（比如第二版官方的 `d2l` 包）才需要：

```bash
just setup                      # 建 .venv（只在第一次）
uv add d2l                      # 装进 .venv，同时写进 pyproject.toml
uv run python work/xxx.py       # 用 .venv 跑（能看到 .venv 里的包）
```

注意 `python3 xxx.py` 看不到 `.venv` 里的包，`uv run python xxx.py` 才能。

其他：

- 升级 `nixpkgs`：`nix flake update`（远程要代理，见下）。
- 远程访问 github 要代理：`export https_proxy=http://127.0.0.1:7890`。
  nixpkgs 的二进制缓存已配 USTC 镜像，拉 nix 包不用代理。

## 画图怎么看

远程没显示器，脚本里必须先切后端：

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
...
plt.savefig("out.png", dpi=150)
```

然后二选一：本地终端 `scp` 拉下来看，或用 VSCodium 的 Remote-SSH 直接打开 PNG。

## 数据集

`d2l` 包默认的几个下载源在国内不通（`ap-northeast-1.d2l.ai` 超时，
`d2l-data.s3-accelerate.amazonaws.com` 403）。所以数据管道要自己写，
用 `torchvision.datasets` 或原始镜像地址，具体见 `notes/a0-setup.md`。
