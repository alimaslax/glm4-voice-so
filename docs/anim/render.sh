#!/usr/bin/env bash
# Render the five scenes, pace each one to its narration, and join them into a
# single narrated video per colour theme:
#
#   docs/media/overview.<theme>.mp4   the film the page plays
#   docs/media/overview.<theme>.jpg   poster frame
#   docs/media/chapters.json          chapter start times, read by index.html
#
# Prerequisites:
#   brew install pkgconf cairo pango ffmpeg && pip install manim
#   docs/anim/narrate.sh              (writes docs/media/narration/*.m4a)
#
# MANIM=/path/to/manim overrides the binary (e.g. a virtualenv).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
out="$here/../media"
narr="$out/narration"
tmp="${TMPDIR:-/tmp}/so-anim-media"
manim_bin="${MANIM:-manim}"
scenes=(S1Pipeline S2Tokenizer S4LoRA S3Interleave S5Voice)
titles=("The chain" "The tokenizer" "The adapter" "The format" "Two tracks")
tail_hold=1.0        # seconds of stillness after the narrator stops
max_gap=1.1          # longest pause we will insert between animations

dur(){ ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }

mkdir -p "$out" "$tmp"

for scene in "${scenes[@]}"; do
  [ -f "$narr/$scene.m4a" ] || { echo "missing $narr/$scene.m4a — run narrate.sh first"; exit 1; }
done

# ---- pass 1: measure each scene at draft quality to work out its pacing ----
# (indexed arrays, not associative ones: macOS still ships bash 3.2)
gaps=()
for i in "${!scenes[@]}"; do
  scene="${scenes[$i]}"
  plays=$(SO_THEME=dark SO_GAP=0 "$manim_bin" -ql --disable_caching -v WARNING --format=mp4 --media_dir "$tmp/measure" \
            "$here/scenes.py" "$scene" 2>&1 | grep -o 'plays=[0-9]*' | cut -d= -f2)
  base=$(dur "$tmp/measure/videos/scenes/480p15/$scene.mp4")
  want=$(dur "$narr/$scene.m4a")
  gaps[$i]=$(python3 -c "print(f'{min($max_gap, max(0.0, ($want + $tail_hold - $base) / $plays)):.3f}')")
  echo "--- $scene: $plays animations, ${base}s -> narration ${want}s, gap ${gaps[$i]}s"
done

# ---- pass 2: render, pace, narrate, join ----
for theme in light dark; do
  list="$tmp/$theme.concat"; : > "$list"
  for i in "${!scenes[@]}"; do
    scene="${scenes[$i]}"
    SO_THEME="$theme" SO_GAP="${gaps[$i]}" "$manim_bin" -r 1920,1080 --fps 30 --disable_caching -v WARNING \
      --format=mp4 --media_dir "$tmp/$theme" "$here/scenes.py" "$scene"
    src="$tmp/$theme/videos/scenes/1080p30/$scene.mp4"
    seg="$tmp/$theme/$scene.seg.mp4"
    # hold the last frame until the narrator finishes, then mux the narration
    pad=$(python3 -c "print(f'{max(0.0, $(dur "$narr/$scene.m4a") + $tail_hold - $(dur "$src")):.3f}')")
    ffmpeg -v error -y -i "$src" -i "$narr/$scene.m4a" \
      -filter_complex "[0:v]tpad=stop_mode=clone:stop_duration=$pad,format=yuv420p[v];[1:a]apad=pad_dur=$tail_hold,aresample=48000[a]" \
      -map '[v]' -map '[a]' -c:v libx264 -crf 26 -preset slow -movflags +faststart \
      -c:a aac -b:a 128k -ac 2 -shortest "$seg"
    printf "file '%s'\n" "$seg" >> "$list"
    echo "==> $scene ($theme): +${pad}s hold"
  done
  ffmpeg -v error -y -f concat -safe 0 -i "$list" -c copy -movflags +faststart \
    "$out/overview.$theme.mp4"
  ffmpeg -v error -y -ss 20 -i "$out/overview.$theme.mp4" -frames:v 1 -q:v 4 \
    "$out/overview.$theme.jpg"
  echo "==> $out/overview.$theme.mp4  $(dur "$out/overview.$theme.mp4")s"
done

# ---- chapter marks (identical for both themes) ----
python3 - "$tmp/dark.concat" "${scenes[@]}" <<'PY' > "$out/chapters.json"
import json, subprocess, sys
listfile, scenes = sys.argv[1], sys.argv[2:]
titles = {"S1Pipeline":"The chain","S2Tokenizer":"The tokenizer","S4LoRA":"The adapter",
          "S3Interleave":"The format","S5Voice":"Two tracks"}
files = [l.split("'")[1] for l in open(listfile)]
t, out = 0.0, []
for scene, f in zip(scenes, files):
    out.append({"scene": scene, "title": titles[scene], "start": round(t, 2)})
    t += float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                               "-of","csv=p=0",f], capture_output=True, text=True).stdout)
print(json.dumps({"total": round(t, 2), "chapters": out}, indent=2))
PY
echo "==> $out/chapters.json"

python3 "$here/inject_chapters.py"
