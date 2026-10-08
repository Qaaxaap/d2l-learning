SHELL := bash
PY := python3
.PHONY: help env setup run check clean

help:
	@echo "make env              打印解释器与关键包的版本、CUDA 是否可用"
	@echo "make run F=<脚本>     跑一个脚本，例: make run F=work/a0/hello_tensor.py"
	@echo "make check U=<单元>   跑某单元的验收断言，例: make check U=b02"
	@echo "make setup            需要额外包时：建带系统包可见性的 .venv 并同步"
	@echo "make clean            删掉 .venv 和 __pycache__"

env:
	@$(PY) - <<'PY'
import sys, importlib.metadata as md
print("python      :", sys.version.split()[0], "->", sys.executable)
for name in ["torch", "torchvision", "numpy", "matplotlib", "pandas", "tqdm", "requests"]:
    try:
        print(f"{name:12s}:", md.version(name))
    except Exception:
        print(f"{name:12s}: 未安装")
try:
    import torch
    print("cuda 可用   :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("显卡        :", torch.cuda.get_device_name(0))
except Exception as e:
    print("导入 torch 失败:", e)
PY

setup:
	@test -d .venv || uv venv --system-site-packages --python /usr/bin/python3
	uv sync

run:
	@test -n "$(F)" || { echo "用法: make run F=<脚本路径>"; exit 1; }
	$(PY) "$(F)"

check:
	@test -n "$(U)" || { echo "用法: make check U=<单元号>，例: make check U=b02"; exit 1; }
	$(PY) "tools/check_$(U).py"

clean:
	rm -rf .venv
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
