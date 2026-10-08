SHELL := bash
.PHONY: help setup run check clean

help:
	@echo "make setup            首次/更新依赖（在 nix develop 里跑）"
	@echo "make run F=<脚本>     跑一个脚本，例: make run F=work/a1/hello.py"
	@echo "make check U=<单元>   跑某单元的验收断言，例: make check U=b02"
	@echo "make clean            删掉 .venv 和 __pycache__"

setup:
	uv sync

run:
	@test -n "$(F)" || { echo "用法: make run F=<脚本路径>"; exit 1; }
	python "$(F)"

check:
	@test -n "$(U)" || { echo "用法: make check U=<单元号>，例: make check U=b02"; exit 1; }
	python "tools/check_$(U).py"

clean:
	rm -rf .venv
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
