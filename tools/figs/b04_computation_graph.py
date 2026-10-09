"""生成 B04 讲义用的计算图。

在开发机上跑（那里有 matplotlib）：
    python3 tools/figs/b04_computation_graph.py
输出：notes/figs/b04-computation-graph.png

图里只用英文与数学符号，避免字体缺失。
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FORWARD = "#2c3e50"
BACKWARD = "#c0392b"
LEAF = "#fff3cd"
OP = "#dce8f7"

fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")


def box(x, y, text, fc, w=1.7, h=0.95, fs=12):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.08",
            fc=fc,
            ec="#5b7fa6",
            lw=1.3,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)


def arrow(x1, y1, x2, y2, color, rad=0.0, lw=1.5, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=15,
            color=color,
            lw=lw,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


# ---------- 前向 ----------
Y_FWD = 3.6
xs = [0.9, 3.1, 5.5, 7.9]
box(xs[0], Y_FWD, "x", LEAF)
box(xs[1], Y_FWD, "u = x*x", OP)
box(xs[2], Y_FWD, "s = u.sum()", OP, w=2.1)
box(xs[3], Y_FWD, "y = 2*s", OP, w=1.8)

for a, b in zip(xs, xs[1:]):
    arrow(a + 0.9, Y_FWD, b - (1.05 if b == xs[2] else 0.9), Y_FWD, FORWARD)

ax.text(5.0, 4.75, "forward", ha="center", fontsize=13, color=FORWARD)
ax.text(0.9, 4.2, "leaf", ha="center", fontsize=10, color="#8a6d00")

# ---------- 反向 ----------
Y_BWD = 1.5
bx = [0.9, 3.1, 5.5, 7.9]
for x, label in zip(bx, ["x.grad", "2x", "1", "2"]):
    box(x, Y_BWD, label, "#fdecea" if x != 0.9 else LEAF, w=1.7)

for a, b in zip(bx[::-1], bx[::-1][1:]):
    arrow(a - 0.9, Y_BWD, b + 0.9, Y_BWD, BACKWARD)

ax.text(
    5.0,
    0.45,
    "backward: start from y, walk left, multiply the local derivative at each box, "
    "accumulate the result on x",
    ha="center",
    fontsize=10.5,
    color=BACKWARD,
)

# 上下两排按 x 位置对齐，不画连接线（会与说明文字打架）

ax.text(5.0, 2.5, "y.backward()", ha="center", fontsize=11.5, color=BACKWARD)

plt.tight_layout()
plt.savefig("notes/figs/b04-computation-graph.png", bbox_inches="tight", facecolor="white")
print("已生成 notes/figs/b04-computation-graph.png")
