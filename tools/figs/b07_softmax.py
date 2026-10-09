"""生成 B07 讲义用的 softmax 示意图。

在开发机上跑（那里有 matplotlib）：
    python3 tools/figs/b07_softmax.py
输出：notes/figs/b07-softmax.png

图里只用英文与数学符号，避免字体缺失。数值是真算出来的。
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch

LOGITS = torch.tensor([2.0, 1.0, 0.1, -1.0])
PROBS = torch.softmax(LOGITS, dim=0)

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=150,
                         gridspec_kw={"wspace": 0.28})

# 左：logits，可以为负、和不为一
ax = axes[0]
colors = ["#c0392b" if v < 0 else "#5b7fa6" for v in LOGITS.tolist()]
ax.bar(range(4), LOGITS.tolist(), color=colors, width=0.6)
ax.axhline(0, color="#333", lw=1.0)
ax.set_xticks(range(4))
ax.set_xticklabels([f"class {i}" for i in range(4)], fontsize=9)
ax.set_ylim(-1.8, 2.6)
ax.set_title("logits  o = XW + b", fontsize=12)
ax.text(
    1.5,
    2.25,
    "range: any real number\nsum: not 1",
    ha="center",
    fontsize=9.5,
    color="#7a7a7a",
)
ax.tick_params(labelsize=9)

# 右：概率，都在 0-1 内、和为 1
ax = axes[1]
ax.bar(range(4), PROBS.tolist(), color="#e8f5e9", edgecolor="#5b7fa6", width=0.6)
ax.set_xticks(range(4))
ax.set_xticklabels([f"class {i}" for i in range(4)], fontsize=9)
ax.set_ylim(0, 0.8)
ax.set_title("softmax(o)", fontsize=12)
for i, p in enumerate(PROBS.tolist()):
    ax.text(i, p + 0.03, f"{p:.3f}", ha="center", fontsize=9)
ax.text(
    1.5,
    0.68,
    "range: (0, 1)\nsum: exactly 1",
    ha="center",
    fontsize=9.5,
    color="#7a7a7a",
)
ax.tick_params(labelsize=9)

# 中间的箭头
fig.add_artist(
    FancyArrowPatch(
        (0.482, 0.52),
        (0.545, 0.52),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=22,
        color="#2c3e50",
        lw=1.8,
    )
)

plt.tight_layout()
plt.savefig("notes/figs/b07-softmax.png", bbox_inches="tight", facecolor="white")
print("已生成 notes/figs/b07-softmax.png")
print("logits =", LOGITS.tolist())
print("probs  =", [round(v, 4) for v in PROBS.tolist()], "和 =", round(PROBS.sum().item(), 6))
