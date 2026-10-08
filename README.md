# d2l 学习仓库

《动手学深度学习》**第一版纸质书** + **PyTorch** 实现。契约见 [PLAN.md](PLAN.md)，
私教（AI）的工作规则见 [AGENTS.md](AGENTS.md)，进度见 [PROGRESS.md](PROGRESS.md)。

权威副本在远程机器 `Qaaxaap@192.168.1.155:2222` 的 `~/Projects/d2l`，
本地 `/home/Qaaxaap/Projects/d2l-tutor` 是 AI 的写作区，用 `tools/sync.sh` 同步。

## 目录

| 路径 | 谁写 | 内容 |
|---|---|---|
| `notes/` | AI | 讲义。每章一份，含公式推导、MXNet→PyTorch 对照、易错点 |
| `exercises/<单元>/` | AI | `READING.md` 阅读任务书、`oral.md` 口试题、`code.md` 代码题 |
| `solutions/` | AI | 参考答案（做完再看） |
| `work/<单元>/` | **你** | 你的实现。这是你的地盘，AI 不会覆盖 |
| `tools/` | AI | 验收断言、同步脚本 |
| `refs/` | AI | 书里 MXNet 原码摘录、官方 torch 版对照（做题时别看） |

## 开始

```bash
ssh -p 2222 Qaaxaap@192.168.1.155
cd ~/Projects/d2l
nix develop                 # 进开发环境（几秒）
make setup                  # 首次：创建 .venv 并装 torch（约 4G，一次就够）
make run F=work/a1/hello.py # 跑脚本
```

进了 `nix develop` 之后，`python` / `uv` / `make` 都在 PATH 里。退出用 `exit`。
平时只要 `nix develop` 一条命令，不需要 `conda activate` 那种钩子，也不影响 shell 启动速度。

## 环境要点

- `flake.nix` 提供 `python3.12` + `uv`。nixpkgs 已锁定在 `flake.lock`。
- Python 依赖装在项目内的 `.venv`，由 `pyproject.toml` 声明，`uv sync` 更新。
- `flake.nix` 的 `shellHook` 会把宿主 NVIDIA 驱动库（`libcuda.so.1` 等）链到
  `~/.cache/d2l/driver-libs` 并加进 `LD_LIBRARY_PATH`。不这么做，nix 里的 Python 找不到驱动。
  只链这四个库，不把整个 `/usr/lib` 加进去，避免系统 glibc 盖掉 nix 的。
- 要升级 `nixpkgs`：`nix flake update`（需要代理，见下）。
- 远程访问 github 要代理：`export https_proxy=http://127.0.0.1:7890`。nixpkgs 的二进制缓存
  已经配了 USTC 镜像，拉包不用代理。

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
