"""生成 B02 讲义用的广播示意图。

在开发机上跑（那里有 matplotlib）：
    python3 tools/figs/b02_broadcast.py
输出：notes/figs/b02-broadcast.png

图里只用英文与数学符号，避免字体缺失。
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

A_COLOR = "#dce8f7"
B_COLOR = "#fdecea"
OUT_COLOR = "#e8f5e9"
EDGE = "#5b7fa6"
CELL = 0.5

fig, ax = plt.subplots(figsize=(9.4, 4.0), dpi=150)
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.6)
ax.axis("off")


def grid(x0, y_top, rows, cols, fc):
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(
                Rectangle(
                    (x0 + c * CELL, y_top - (r + 1) * CELL),
                    CELL,
                    CELL,
                    fc=fc,
                    ec=EDGE,
                    lw=1.1,
                )
            )


def arrow(x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=17, color="#2c3e50", lw=1.6
        )
    )


Y_TOP = 4.15

# A: (3, 4)
ax.text(2.0, 4.45, "A : (3, 4)", ha="center", fontsize=11.5, color="#2c3e50")
grid(0.95, Y_TOP, 3, 4, A_COLOR)

# b: (4,)
ax.text(6.0, 4.45, "b : (4,)", ha="center", fontsize=11.5, color="#2c3e50")
grid(5.6, Y_TOP, 1, 4, B_COLOR)

# 结果
ax.text(10.05, 4.45, "A + b : (3, 4)", ha="center", fontsize=11.5, color="#2c3e50")
grid(9.05, Y_TOP, 3, 4, OUT_COLOR)

# 箭头放在格子下方，避开文字
arrow(3.75, 3.05, 5.45, 3.05)
arrow(7.85, 3.05, 8.9, 3.05)

ax.text(
    6.0,
    2.35,
    "align right: b gets padded on the left to (1, 4),\nso every row of A gets the same b",
    ha="center",
    va="top",
    fontsize=10.5,
    color="#2c3e50",
)

ax.text(
    6.0,
    0.75,
    "axes that do not exist on the right side count as length 1",
    ha="center",
    fontsize=10.5,
    color="#5b7fa6",
)

plt.tight_layout()
plt.savefig("notes/figs/b02-broadcast.png", bbox_inches="tight", facecolor="white")
print("已生成 notes/figs/b02-broadcast.png")
