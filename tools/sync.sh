#!/usr/bin/env bash
# 本地写作区 <-> 远程权威仓库 的同步。
#   push  本地 -> 远程（讲义/题目/工具）。不碰 work/，那是学习者写代码的地方。
#   pull  远程 -> 本地（拉回学习者的代码和进度，供 AI 批改）。
set -euo pipefail

REMOTE="Qaaxaap@192.168.1.155"
PORT=2222
RDIR="Projects/d2l"
LDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_OPTS=(-e "ssh -p $PORT -o BatchMode=yes")

usage() { echo "用法: $0 {push|pull}"; exit 1; }

case "${1:-}" in
  push)
    rsync -az --chmod=D755,F644 "${SSH_OPTS[@]}" \
      --exclude '.git/' --exclude 'work/' --exclude '.venv/' \
      --exclude '__pycache__/' --exclude 'solutions/' \
      "$LDIR"/ "$REMOTE:$RDIR"/
    echo "已推送到 $REMOTE:$RDIR"
    ;;
  pull)
    mkdir -p "$LDIR/work"
    rsync -az --chmod=D755,F644 "${SSH_OPTS[@]}" \
      --exclude '__pycache__/' \
      "$REMOTE:$RDIR/work/" "$LDIR/work/"
    rsync -az --chmod=D755,F644 "${SSH_OPTS[@]}" \
      "$REMOTE:$RDIR/PROGRESS.md" "$LDIR/PROGRESS.md" 2>/dev/null || true
    echo "已从 $REMOTE:$RDIR 拉回 work/ 与 PROGRESS.md"
    ;;
  *)
    usage
    ;;
esac
