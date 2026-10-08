#!/usr/bin/env bash
# 扬声器提醒。只在本地跑，开发机没有音频设备。
#
# 用法：
#   tools/remind.sh                       一声提示音
#   tools/remind.sh 3                     连响三声，溜号了用这个
#   tools/remind.sh 2 "该看进度了"          响两声，再念出来
#
# 语音用 piper（离线，中文自然），模型在 ~/.local/share/piper/。
# piper 走本仓库的 tts devShell，不进 home-manager：它默认开 withTrain，
# 会拖进近 800 MiB 的训练依赖，而这里只用来念一句话。
set -euo pipefail

# AI 的沙盒 bash 里既没有 XDG_RUNTIME_DIR 也没有 DBUS 地址，
# 不补的话 paplay 会报 "pa_context_connect() failed: Connection refused"，
# 而脚本里的 || true 会把它吞掉，看起来像播过了。
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$HOME/.local/share/piper/zh_CN-huayan-medium.onnx"
SOUND=/usr/share/sounds/freedesktop/stereo/complete.oga
WAV=/tmp/d2l-remind.wav

TIMES=1
if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
  TIMES="$1"
  shift
fi
MSG="${1:-}"

for _ in $(seq "$TIMES"); do
  paplay --volume=18000 "$SOUND" 2>/dev/null || true
  sleep 1
done

if [ -n "$MSG" ] && [ -f "$MODEL" ]; then
  # 别把 stderr 丢掉，合成或播放失败时要看得见
  if printf '%s\n' "$MSG" | nix develop "$REPO#tts" --command piper \
      --model "$MODEL" --output_file "$WAV"; then
    paplay --volume=26000 "$WAV" || echo "警告：语音播放失败" >&2
  else
    echo "警告：piper 合成失败" >&2
  fi
fi

notify-send -u normal "d2l" "${MSG:-该看进度了}" 2>/dev/null || true
