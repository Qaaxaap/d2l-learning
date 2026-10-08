# d2l 学习仓库

跟着《动手学深度学习》**第一版纸质书**学深度学习，书上给的 MXNet 代码全部用 **PyTorch** 自己重写一遍。

这是 Qaaxaap 的学习记录，公开出来供参考。仓库里由 AI 私教陪着学，题目、讲义、验收断言都由它准备。

| | |
|---|---|
| 教材 | 《动手学深度学习》第一版（书上代码是 MXNet） |
| 电子版 | <https://zh.d2l.ai/>（第二版，有 PyTorch 代码，章节号与纸质书不同） |
| 框架 | PyTorch 2.14 + CUDA |

## 仓库里有什么

两类内容，归属不同：

| 类别 | 路径 | 说明 |
|---|---|---|
| **学习者的记录** | `work/` | 我写的代码与实验 |
| | `log/` | 每次学习的经过，格式见 [log/README.md](log/README.md) |
| | `PROGRESS.md` | 进度、掌握度、错题本 |
| **私教准备的** | `notes/` | 讲义：公式推导、MXNet→PyTorch 对照、易错点 |
| | `exercises/` | 每个单元的阅读任务书、口试题、代码题 |
| | `solutions/` | 参考答案 |
| | `tools/` | 验收断言 |

## 文档里的人称

这些文档由私教和学习者分别写，人称不统一，读的时候按下表对照。

| 文件 | 视角 | 第一人称指 | 第二人称指 | "学习者"指 |
|---|---|---|---|---|
| `PLAN.md`、`notes/`、`exercises/` | 私教对学习者说话 | 私教 | 学习者 | 学习者 |
| `AGENTS.md` | 给 AI 的指令 | — | — | 学习者 |
| `log/` | 客观记录每次交互 | — | — | 学习者 |
| `README.md` | 本文件 | 学习者 | — | 学习者 |

## 怎么学

每个单元走这五步，完整约定见 [PLAN.md](PLAN.md)：

1. 看 `exercises/<单元>/READING.md`，知道读纸质书哪几页、读的时候要能回答什么问题
2. 读纸质书，做批注
3. 把实现写进 `work/<单元>/`
4. `just check <单元>` 跑验收断言，红了就改
5. 找私教做口试，题目在 `exercises/<单元>/oral.md`

答不上来、写不出来就不进下一个单元。

## 环境

解释器和科学计算包用系统那份，nix flake 只管工具链。常用命令：

```bash
just env                            # 看各包版本与 CUDA 状态
just run work/a0/hello_tensor.py    # 跑一个脚本
just check b02                      # 跑某单元的验收断言
just --list                         # 看全部命令
```

搭环境、跑脚本、看图、git 的细节见 [`notes/a0-setup.md`](notes/a0-setup.md)。

## 参考材料

- [`notes/chapter-map.md`](notes/chapter-map.md) 纸质书与第二版电子版的逐节对照，第二版多出来的章节都标了
- [`notes/d2lzh-unlock-table.md`](notes/d2lzh-unlock-table.md) 书上"已封装好、以后直接用"的完整清单
- [`AGENTS.md`](AGENTS.md) 私教的工作规则
