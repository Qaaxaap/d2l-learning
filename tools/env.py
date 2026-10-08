"""打印当前 Python 环境概况：解释器路径、关键包版本、CUDA 可用性。"""

import importlib.metadata as md
import sys

PACKAGES = ["torch", "torchvision", "numpy", "matplotlib", "pandas", "tqdm", "requests"]


def main() -> None:
    print(f"python      : {sys.version.split()[0]}  ({sys.executable})")
    for name in PACKAGES:
        try:
            print(f"{name:12s}: {md.version(name)}")
        except md.PackageNotFoundError:
            print(f"{name:12s}: 未安装")

    try:
        import torch
    except ImportError as exc:
        print(f"导入 torch 失败: {exc}")
        return

    print(f"cuda 可用   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"显卡        : {torch.cuda.get_device_name(0)}")
        print(f"显存        : {props.total_memory / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
