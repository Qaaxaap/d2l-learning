"""生成 B11 讲义用的图：层数越多，连乘的权重量级越失控。

对应讲义 4.1 节。三种权重量级各画一条：0.2（衰减）、1.0（不变）、5.0（爆炸）。
图里文字用英文，避免中文字体缺失。

跑法：
    python3 tools/figs/b11_vanishing.py
输出：
    notes/figs/b11-vanishing.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent.parent / "notes" / "figs" / "b11-vanishing.png"


def main() -> None:
    layers = np.arange(1, 31)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for w, color, label in [
        (0.2, "#c53030", "W = 0.2"),
        (1.0, "#718096", "W = 1.0"),
        (5.0, "#2b6cb0", "W = 5.0"),
    ]:
        magnitude = w**layers
        ax.semilogy(layers, magnitude, "o-", color=color, label=label, markersize=3)

    # 标出两个极端值
    ax.annotate(
        f"0.2^30 = {0.2**30:.1e}",
        xy=(30, 0.2**30),
        xytext=(18, 1e-17),
        fontsize=9,
        color="#c53030",
        arrowprops=dict(arrowstyle="->", color="#c53030", lw=0.8),
    )
    ax.annotate(
        f"5^30 = {5.0**30:.1e}",
        xy=(30, 5.0**30),
        xytext=(14, 1e17),
        fontsize=9,
        color="#2b6cb0",
        arrowprops=dict(arrowstyle="->", color="#2b6cb0", lw=0.8),
    )
    ax.axhline(1.0, color="#cbd5e0", linestyle=":", linewidth=1)
    ax.set_xlabel("number of layers")
    ax.set_ylabel("magnitude after L layers  (log scale)")
    ax.set_title("Multiplying L weight matrices: vanishing or exploding")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"已写出 {OUT}")
    for w in (0.2, 1.0, 5.0):
        print(f"  W={w}: 第 30 层量级 {w**30:.3e}")


if __name__ == "__main__":
    main()
