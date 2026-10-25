#!/bin/bash
# Reuse a finished S0 -> S1 screen run (scripts/cl/screen_s1.sh) as session 1 of the full run of a method, so
# scripts/cl/run_method.sh skips S1 and continues at S2. Only valid for screens with the main runs' setup (the default
# SETUP=main, folder v1_40000_full): that S1 run is then the run run_method.sh would have trained. Idempotent.
#
#   bash scripts/cl/promote_screen_s1.sh <variant> [method=<variant>]
#   e.g. bash scripts/cl/promote_screen_s1.sh tmr_dance_sel && bash scripts/cl/run_tmr_dance_sel.sh
set -euo pipefail
V=${1:?usage: promote_screen_s1.sh <variant> [method]}
M=${2:-$V}
B=aistpp_genre5
TAG=${TAG:-v1_40000_full}
S1=$({ ls -d outputs/screen/$TAG/*_cl_ldf_${B}_s1_${V} 2>/dev/null || true; } | tail -1)
[ -n "$S1" ] && [ -f "$S1/last.ckpt" ] || { echo "no finished screen run of $V under outputs/screen/$TAG"; exit 1; }
for i in 0 1; do
  [ -f "results/$B/screen/$TAG/$V/after_s1_sess$i.json" ] || { echo "screen run of $V is not scored yet (S$i)"; exit 1; }
done
LINK=outputs/$(basename "$S1" | sed "s/_s1_${V}\$/_s1_${M}/")  # the folder name run_method.sh looks for
if [ ! -f "$LINK/last.ckpt" ]; then  # a real folder with a hard link to the checkpoint (a copy across filesystems)
  mkdir -p "$LINK"
  ln "$S1/last.ckpt" "$LINK/last.ckpt" 2>/dev/null || cp "$S1/last.ckpt" "$LINK/last.ckpt"
  echo "$S1" > "$LINK/PROMOTED_FROM_SCREEN"
fi
mkdir -p "buffers/$B/$M" "results/$B/$M"
[ -f "buffers/$B/$M/s0.json" ] || cp "buffers/$B/screen/$TAG/$V/s0.json" "buffers/$B/$M/s0.json"
for i in 0 1; do
  [ -f "results/$B/$M/after_s1_sess$i.json" ] || cp "results/$B/screen/$TAG/$V/after_s1_sess$i.json" "results/$B/$M/"
done
echo "S1 of $M = $S1 (linked as $LINK); buffer s0 and S1 scores copied to buffers/$B/$M and results/$B/$M"
