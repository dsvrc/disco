#!/bin/bash
# Test every finished model of a screen on every session it has seen, then write the table of the whole screen
# (scripts/cl/screen_matrix_table.py): rows = model after session t, with a sub-row per dance seen up to t; columns =
# baselines. Only what is missing is run: a test counts as done when its score and its saved dances (needed for the
# per-dance rows) are there, so a rerun after more runs finished is cheap. A score whose dances were not saved is
# regenerated (same checkpoint and seed), its old json kept as <score>.json.bak. Models a screen is still training are
# skipped: that screen scores them itself. Tests run one per GPU in GPUS; several copies can run at once (a lock per test).
#
#   GPUS="0 1" bash scripts/cl/screen_test_all.sh 2>&1 | tee test_all.log
#   env: TAG (screen folder, default v2_40000_full_s0v2), GPUS, SAMPLES (3), S0_RUN (default: the S0 model the screen
#        scored), VAE_CKPT, STYLE_ENCODER_DEVICE (table only, default cpu)
#   writes: results/<benchmark>/screen/<TAG>/matrix_by_genre.md and matrix_by_session.md
set -euo pipefail
B=aistpp_genre5
TAG=${TAG:-v2_40000_full_s0v2}
read -r -a GPU_LIST <<< "${GPUS:-0}"
SAMPLES=${SAMPLES:-3}
OUT=outputs/screen/$TAG
RES=results/$B/screen/$TAG
LOG=logs/$B/screen/$TAG
[ -d "$RES" ] || { echo "[test-all] no screen results at $RES (run from the repo root, set TAG)"; exit 1; }
mkdir -p "$LOG"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
VAE_CKPT=${VAE_CKPT:-$(ls -d outputs/*_cl_vae_${B}_alltrain | tail -1)/last.ckpt}
S0_CK=${S0_RUN:+$S0_RUN/last.ckpt}
if [ -z "$S0_CK" ] && [ -f "$RES/s0_model/after_s0_sess0.json" ]; then
  S0_CK=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['ckpt'])" "$RES/s0_model/after_s0_sess0.json")
fi
EVAL="--samples $SAMPLES"
MAX_TOKENS=$(echo "$TAG" | cut -d_ -f3)
[ "$MAX_TOKENS" != full ] && EVAL="$EVAL --max_tokens $MAX_TOKENS"

model_of() {  # checkpoint of the variant's model after session $2 (empty when there is no run)
  if [ "$2" -eq 0 ]; then echo "$S0_CK"; return; fi
  local d ; d=$({ ls -d "$OUT"/*_cl_ldf_${B}_s$2_$1 2>/dev/null || true; } | tail -1)
  if [ -n "$d" ]; then echo "$d/last.ckpt"; fi
}

complete() {  # score + saved dances + the names that map the dances to their music
  [ -f "$1.json" ] && [ -f "$1.joints.npz" ] && grep -q '"names"' "$1.json"
}

busy() {  # a screen is still training this variant's session (scripts/cl/screen_s2.sh, screen_sk.sh)
  local lock=$RES/$1/.s$2_lock pid
  [ -d "$lock" ] || return 1
  pid=$(cat "$lock/pid" 2>/dev/null || true)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

todo=()
for d in "$RES"/*/; do
  v=$(basename "$d")
  if [ "$v" = s0_model ]; then sessions=0; else sessions="1 2 3 4"; fi
  for t in $sessions; do
    ck=$(model_of "$v" "$t")
    if [ -z "$ck" ] || [ ! -f "$ck" ]; then
      [ "$t" -eq 0 ] && echo "[test-all] s0_model: no checkpoint ('$ck'), set S0_RUN"
      continue
    fi
    if busy "$v" "$t"; then echo "[test-all] $v after S$t: still training in a screen, skipped"; continue; fi
    for i in $(seq 0 "$t"); do
      complete "$RES/$v/after_s${t}_sess$i" || todo+=("$v $t $i")
    done
  done
done
echo "[test-all] screen $TAG: ${#todo[@]} tests to run on GPUs ${GPU_LIST[*]}"

test_one() {  # <variant> <trained up to session> <session tested> <gpu>
  local v=$1 t=$2 i=$3 gpu=$4 f=$RES/$1/after_s$2_sess$3 lock pid
  lock=$f.test_lock
  if ! mkdir "$lock" 2>/dev/null; then
    [ -s "$lock/pid" ] || sleep 2  # the owner may not have written its pid yet
    pid=$(cat "$lock/pid" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then return 0; fi
    [ -d "$lock" ] || return 0  # the owner has just finished this test
  fi
  echo "$BASHPID" > "$lock/pid" || return 0
  if ! complete "$f"; then
    [ -f "$f.json" ] && cp -n "$f.json" "$f.json.bak"
    echo "[test-all] $v after S$t on S$i: GPU $gpu ($(date +%H:%M))"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/eval_cl_session.py --config configs/cl/ldf_${B}_s$t.yaml \
      --vae_ckpt "$VAE_CKPT" --ckpt "$(model_of "$v" "$t")" --session "$i" $EVAL --out "$f.json" \
      > "$LOG/test_$v.s$t.eval$i.log" 2>&1 || echo "[test-all] $v after S$t on S$i: FAILED, see $LOG/test_$v.s$t.eval$i.log"
  fi
  rm -rf "$lock"
}

worker() {  # every GPU walks the same list; a test is run by whichever GPU takes its lock first
  local gpu=$1 job
  for job in "${todo[@]}"; do
    read -r v t i <<< "$job"
    test_one "$v" "$t" "$i" "$gpu"
  done
}

for g in "${GPU_LIST[@]}"; do worker "$g" & done
wait

STYLE_ENCODER_DEVICE=${STYLE_ENCODER_DEVICE:-cpu} python scripts/cl/screen_matrix_table.py --results "$RES" --benchmark "$B"
