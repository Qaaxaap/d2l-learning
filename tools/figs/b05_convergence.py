"""生成 B05 讲义用的大数定律收敛曲线。

在开发机上跑（那里有 matplotlib）：
    python3 tools/figs/b05_convergence.py
输出：notes/figs/b05-convergence.png

真掷骰子，不是画出来的示意线。
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

N = 2000
SEED = 0

g = torch.Generator().manual_seed(SEED)
probs = torch.ones(6) / 6
rolls = torch.multinomial(probs, N, replacement=True, generator=g)

# 累积计数 -> 频率，形状 (N, 6)
counts = torch.zeros(N, 6)
counts[torch.arange(N), rolls] = 1.0
cum = counts.cumsum(0) / torch.arange(1, N + 1).unsqueeze(1)

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), dpi=150)

# 左：六个面的累积频率
ax = axes[0]
for face in range(6):
    ax.plot(cum[:, face].numpy(), lw=1.4, label=f"face {face + 1}")
ax.axhline(1 / 6, color="#2c3e50", lw=1.2, ls="--")
ax.text(N * 0.98, 1 / 6 + 0.012, "1/6", ha="right", fontsize=10, color="#2c3e50")
ax.set_xlabel("number of rolls", fontsize=10)
ax.set_ylabel("estimated probability", fontsize=10)
ax.set_title("each face converges to 1/6", fontsize=11)
ax.set_ylim(0, 0.45)
ax.legend(fontsize=8, ncol=2, frameon=False)
ax.tick_params(labelsize=9)

# 右：偏差的绝对值，对数纵轴
ax = axes[1]
dev = (cum - 1 / 6).abs().max(dim=1).values.numpy()
ax.loglog(range(1, N + 1), dev, lw=1.4, color="#c0392b")
# 参考斜率：随机涨落按 1/sqrt(n) 衰减
import numpy as np

n = np.arange(1, N + 1)
ref = dev[9] * np.sqrt(10 / n)
ax.loglog(n, ref, lw=1.1, ls="--", color="#2c3e50")
ax.text(N * 0.28, ref[int(N * 0.28)] * 1.6, r"$\propto 1/\sqrt{n}$", fontsize=11, color="#2c3e50")
ax.set_xlabel("number of rolls", fontsize=10)
ax.set_ylabel("max deviation from 1/6", fontsize=10)
ax.set_title("fluctuation shrinks like $1/\\sqrt{n}$", fontsize=11)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("notes/figs/b05-convergence.png", bbox_inches="tight", facecolor="white")
print("已生成 notes/figs/b05-convergence.png")
