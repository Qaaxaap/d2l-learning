"""B02 参考实现。做完再看。

对应 exercises/b02/code.md 的 T1-T4。
"""

import pandas as pd
import torch


def tensor_info(x: torch.Tensor) -> dict:
    return {
        "shape": tuple(x.shape),
        "numel": x.numel(),
        "dtype": str(x.dtype).removeprefix("torch."),
        "device": str(x.device),
        "is_contiguous": x.is_contiguous(),
    }


def safe_broadcast_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    sa, sb = tuple(a.shape), tuple(b.shape)
    n = max(len(sa), len(sb))
    # 左对齐：短的那个左边补 1
    pa = (1,) * (n - len(sa)) + sa
    pb = (1,) * (n - len(sb)) + sb
    for i, (da, db) in enumerate(zip(pa, pb)):
        if da != db and da != 1 and db != 1:
            raise ValueError(
                f"形状 {sa} 与 {sb} 无法广播：左对齐后第 {i} 维长度分别为 {da} 和 {db}"
            )
    return a + b


def slice_is_view() -> bool:
    x = torch.arange(6.0)
    y = x[2:4]
    y[0] = 100.0
    return bool(x[2].item() == 100.0)


def clone_is_copy() -> bool:
    x = torch.arange(6.0)
    y = x[2:4].clone()
    y[0] = 100.0
    return bool(x[2].item() == 100.0)


def load_csv(path: str) -> tuple[torch.Tensor, list[str], dict]:
    frame = pd.read_csv(path)
    num_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    cat_cols = [c for c in frame.columns if c not in num_cols]

    columns: dict[str, list[float]] = {}
    stats: dict = {"数值列": num_cols, "类别列": {}}

    for name in num_cols:
        filled = frame[name].fillna(frame[name].mean())
        columns[name] = filled.astype("float32").tolist()

    for name in cat_cols:
        counts = frame[name].value_counts().to_dict()
        stats["类别列"][name] = {str(k): int(v) for k, v in counts.items()}
        for value in sorted(counts):
            columns[f"{name}_{value}"] = (frame[name] == value).astype("float32").tolist()

    names = list(columns)
    # columns 是 列名 -> 一列数据，转成 行 x 列
    feats = torch.tensor([columns[n] for n in names], dtype=torch.float32).T
    return feats, names, stats
