#!/usr/bin/env bash
# 本地写作区 <-> 开发机 的同步。
#   push  本地 -> 开发机（讲义/题目/工具）。不碰 work/，那是学习者写代码的地方。
#   pull  开发机 -> 本地（拉回学习者的代码和进度，供 AI 批改）。
#
# 机器地址、端口、远端路径从 LOCAL.env 读，本脚本不含任何地址。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LDIR="$(cd "$HERE/.." && pwd)"

# shellcheck source=/dev/null
[ -f "$LDIR/LOCAL.env" ] && . "$LDIR/LOCAL.env"
: "${D2L_REMOTE:?LOCAL.env 里缺少 D2L_REMOTE}"
: "${D2L_PORT:?LOCAL.env 里缺少 D2L_PORT}"
: "${D2L_RDIR:?LOCAL.env 里缺少 D2L_RDIR}"

SSH_OPTS=(-e "ssh -p $D2L_PORT -o BatchMode=yes")
# 这些是本机私有或学习者本地的产物，两头都不同步
SKIP=(--exclude '.git/' --exclude 'work/' --exclude '.venv/' --exclude '__pycache__/'
      --exclude 'data/' --exclude 'LOCAL.md' --exclude 'LOCAL.env')

usage() { echo "用法: $0 {push|pull}"; exit 1; }

case "${1:-}" in
  push)
    # 不用 --delete：远端可能有学习者手工建的文件，宁可不一致也不误删。
    # 本地删掉的文件要在远端用 `git rm <文件>` 单独处理。
    rsync -az --chmod=D755,F644 "${SSH_OPTS[@]}" "${SKIP[@]}" \
      "$LDIR"/ "$D2L_REMOTE:$D2L_RDIR"/
    echo "已推送到开发机的 $D2L_RDIR"
    ;;
  pull)
    mkdir -p "$LDIR/work"
    rsync -az --chmod=D755,F644 "${SSH_OPTS[@]}" --exclude '__pycache__/' \
      "$D2L_REMOTE:$D2L_RDIR/work/" "$LDIR/work/"
    rsync -az --chmod=D755,F644 "${SSH_OPTS[@]}" \
      "$D2L_REMOTE:$D2L_RDIR/PROGRESS.md" "$LDIR/PROGRESS.md" 2>/dev/null || true
    echo "已拉回 work/ 与 PROGRESS.md"
    ;;
  *)
    usage
    ;;
esac
