"""生成 B03 讲义用的降维求和示意图。

在开发机上跑（那里有 matplotlib）：
    python3 tools/figs/b03_reduction.py
输出：notes/figs/b03-reduction.png

图里只用英文与数学符号，避免字体缺失。
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

CELL = 0.46
EDGE = "#5b7fa6"
SRC = "#dce8f7"
KEPT = "#e8f5e9"
GONE = "#f3f4f6"


def grid(ax, x0, y_top, rows, cols, fc_fn):
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(
                Rectangle(
                    (x0 + c * CELL, y_top - (r + 1) * CELL),
                    CELL,
                    CELL,
                    fc=fc_fn(r, c),
                    ec=EDGE,
                    lw=1.0,
                )
            )


def arrow(ax, x1, y1, x2, y2, color="#c0392b"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, color=color, lw=1.6
        )
    )


fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), dpi=150)
for ax in axes:
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

# ---------- 左：沿 dim=0 ----------
ax = axes[0]
ax.text(0.35, 5.0, "sum(dim=0)", fontsize=12.5, color="#2c3e50")
ax.text(0.35, 4.62, "A : (3, 4)", fontsize=10.5, color="#7a7a7a")
grid(ax, 0.35, 4.45, 3, 4, lambda r, c: SRC)

# 每一列被压成一个数
arrow(ax, 1.6, 2.8, 1.6, 2.15)
ax.text(1.75, 2.4, "each column\ncollapses", fontsize=9.5, color="#7a7a7a", va="center")

ax.text(0.35, 1.55, "result : (4,)", fontsize=10.5, color="#2c3e50")
grid(ax, 0.35, 1.4, 1, 4, lambda r, c: KEPT)

ax.text(0.35, 0.65, "the axis you sum over disappears", fontsize=9.5, color="#c0392b")

# ---------- 右：沿 dim=1 ----------
ax = axes[1]
ax.text(0.35, 5.0, "sum(dim=1)", fontsize=12.5, color="#2c3e50")
ax.text(0.35, 4.62, "A : (3, 4)", fontsize=10.5, color="#7a7a7a")
grid(ax, 0.35, 4.45, 3, 4, lambda r, c: SRC)

arrow(ax, 2.65, 3.1, 3.45, 3.1)
ax.text(3.6, 3.1, "each row\ncollapses", fontsize=9.5, color="#7a7a7a", va="center")

ax.text(3.65, 2.35, "result : (3,)", fontsize=10.5, color="#2c3e50")
grid(ax, 3.65, 2.2, 1, 3, lambda r, c: KEPT)

ax.text(0.35, 0.65, "both results are 1-D; keepdim=True keeps the summed axis with length 1",
        fontsize=9.5, color="#c0392b")

plt.tight_layout()
plt.savefig("notes/figs/b03-reduction.png", bbox_inches="tight", facecolor="white")
print("已生成 notes/figs/b03-reduction.png")
