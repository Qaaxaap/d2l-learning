"""生成 B06 讲义用的训练循环示意图。

在开发机上跑（那里有 matplotlib）：
    python3 tools/figs/b06_training_loop.py
输出：notes/figs/b06-training-loop.png

图里只用英文与数学符号，避免字体缺失。
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BOX = "#dce8f7"
UP = "#c0392b"
DOWN = "#2c3e50"

fig, ax = plt.subplots(figsize=(8.6, 6.4), dpi=150)
ax.set_xlim(0, 13)
ax.set_ylim(0, 11)
ax.axis("off")


def box(x, y, text, fc=BOX, w=5.2, h=1.0, fs=12):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.1",
            fc=fc,
            ec="#5b7fa6",
            lw=1.3,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)


def arrow(x1, y1, x2, y2, color=DOWN, rad=0.0, lw=1.6, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=17,
            color=color,
            lw=lw,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


X = 5.0
ys = [9.9, 8.3, 6.7, 5.1, 3.5, 1.9]
labels = [
    "batch (X, y)",
    "forward:  y_hat = X @ w + b",
    "loss:  l = (y_hat - y)^2 / 2",
    "backward:  grads = d l / d w",
    "update:  w -= lr * grad / |B|",
    "zero_grad",
]
for y, text in zip(ys, labels):
    box(X, y, text)

for a, b in zip(ys, ys[1:]):
    arrow(X, a - 0.55, X, b + 0.55)

# 循环：从最后一个框的右侧绕回第一个框。用直角折线，任何一段都不穿过框
RX = X + 3.6
ax.plot(
    [X + 2.6, RX, RX, X + 2.6],
    [ys[-1], ys[-1], ys[0], ys[0]],
    color=UP,
    lw=1.6,
    ls="--",
)
arrow(RX, ys[0], X + 2.6, ys[0], color=UP)
ax.text(
    RX + 0.5,
    (ys[0] + ys[-1]) / 2,
    "next batch",
    rotation=-90,
    ha="center",
    va="center",
    fontsize=11,
    color=UP,
)

ax.text(
    X,
    0.55,
    "one epoch ends when every sample has been used once",
    ha="center",
    fontsize=10.5,
    color="#5b7fa6",
)

plt.tight_layout()
plt.savefig("notes/figs/b06-training-loop.png", bbox_inches="tight", facecolor="white")
print("已生成 notes/figs/b06-training-loop.png")
