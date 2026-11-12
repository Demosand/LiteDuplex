#!/usr/bin/env bash
# Frames from record.cjs -> docs/wave.gif, docs/captions.gif (needs ffmpeg in PATH).
set -euo pipefail
cd "$(dirname "$0")/../.."
F=cache/tmp/frames
mkdir -p docs
gif() {   # gif <clip> <out> <start seconds> <width>
  ffmpeg -loglevel error -y -framerate 30 -i "$F/$1/%04d.png" -ss "$3" \
    -vf "fps=24,scale=$4:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle" \
    -loop 0 "$2"
}
gif wave docs/wave.gif 0 720
gif captions docs/captions.gif 1.4 720
ls -la docs/*.gif
