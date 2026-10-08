set shell := ["bash", "-uc"]

# 列出所有命令
default:
    @just --list

# 打印解释器、关键包版本、CUDA 状态
env:
    python3 tools/env.py

# 跑一个脚本：just run work/a0/hello_tensor.py
run f:
    python3 {{f}}

# 跑某单元的验收断言：just check b02
check u:
    python3 tools/check_{{u}}.py

# 需要额外包时用：建带系统包可见性的 .venv 并同步依赖
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    [ -d .venv ] || uv venv --system-site-packages --python /usr/bin/python3
    uv sync

# 删掉 .venv 与 __pycache__
clean:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf .venv
    find . -name __pycache__ -type d -prune -exec rm -rf {} +
