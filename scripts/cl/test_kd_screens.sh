#!/bin/bash
# Test the +KD runs of scripts/cl/run_kd_screens.sh (TMR+KD, GCR+KD, CSReL+KD, MIR+KD and their seeds): every trained
# model after S1..S4 on every session it has seen, then write the tables. Only what is missing is run: a test counts as
# done when its score, its saved dances and their music names are there (the per-genre rows need them); a score without
# dances is regenerated (same checkpoint and seed) and its old json kept as <score>.json.bak. Models still training in a
# screen are skipped. One test per GPU; uses the same per-test locks as scripts/cl/screen_test_all.sh, so both can run
# at once. Run from the repo root:
#
#   GPUS="0 1" bash scripts/cl/test_kd_screens.sh 2>&1 | tee test_kd_screens.log
#   env: VARIANTS (default "tmr_joint_neff_kd gcr_kd csrel_kd mir_kd"; their <variant>_seed<k> runs are tested too),
#        GPUS ("0"), SAMPLES (3), PROTOCOL (v2), STEPS (40000), S0_TAG (s0v2) or TAG (screen folder, default
#        <PROTOCOL>_<STEPS>_full_<S0_TAG> = v2_40000_full_s0v2), VAE_CKPT, STYLE_ENCODER_DEVICE (tables, default cpu)
#   writes: results/<benchmark>/screen/<TAG>/matrix_by_genre.md, matrix_by_session.md (every method, mean +- sd),
#           paper_disco_tables.md and paper_disco_tables.tex (paper Tab. 3 and supplementary per-genre table)
set -euo pipefail
B=aistpp_genre5
VARIANTS=${VARIANTS:-"tmr_joint_neff_kd gcr_kd csrel_kd mir_kd"}
TAG=${TAG:-${PROTOCOL:-v2}_${STEPS:-40000}_full_${S0_TAG:-s0v2}}
read -r -a GPU_LIST <<< "${GPUS:-0}"
SAMPLES=${SAMPLES:-3}
OUT=outputs/screen/$TAG
RES=results/$B/screen/$TAG
LOG=logs/$B/screen/$TAG
[ -d "$RES" ] || { echo "[test-kd] no screen results at $RES (run from the repo root, set TAG)"; exit 1; }
mkdir -p "$LOG"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
VAE_CKPT=${VAE_CKPT:-$(ls -d outputs/*_cl_vae_${B}_alltrain | tail -1)/last.ckpt}
EVAL="--samples $SAMPLES"
MAX_TOKENS=$(echo "$TAG" | cut -d_ -f3)
[ "$MAX_TOKENS" != full ] && EVAL="$EVAL --max_tokens $MAX_TOKENS"

model_of() {  # checkpoint of the variant's model after session $2 (empty when there is no run)
  local d ; d=$({ ls -d "$OUT"/*_cl_ldf_${B}_s$2_$1 2>/dev/null || true; } | tail -1)
  if [ -n "$d" ]; then echo "$d/last.ckpt"; fi
}
complete() {  # score + saved dances + the names that map the dances to their music
  [ -f "$1.json" ] && [ -f "$1.joints.npz" ] && grep -q '"names"' "$1.json"
}
busy() {  # a screen is still training this variant's session (scripts/cl/screen_sk.sh lock)
  local lock=$RES/$1/.s$2_lock pid
  [ -d "$lock" ] || return 1
  pid=$(cat "$lock/pid" 2>/dev/null || true)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

RUNS=()  # every +KD run of the screen: <variant> and <variant>_seed<k>
for v in $VARIANTS; do
  for d in "$RES/$v"/ "$RES/${v}_seed"[0-9]*/; do
    if [ -d "$d" ]; then d=${d%/}; RUNS+=("${d##*/}"); fi
  done
done
todo=()
for v in "${RUNS[@]}"; do
  for t in 1 2 3 4; do
    ck=$(model_of "$v" "$t")
    if [ -z "$ck" ] || [ ! -f "$ck" ]; then continue; fi
    if busy "$v" "$t"; then echo "[test-kd] $v after S$t: still training in a screen, skipped"; continue; fi
    for i in $(seq 0 "$t"); do
      complete "$RES/$v/after_s${t}_sess$i" || todo+=("$v $t $i")
    done
  done
done
echo "[test-kd] screen $TAG | runs: ${RUNS[*]:-none} | ${#todo[@]} tests to run on GPUs ${GPU_LIST[*]}"

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
    if [ -f "$f.json" ]; then cp -n "$f.json" "$f.json.bak"; fi
    echo "[test-kd] $v after S$t on S$i: GPU $gpu ($(date +%H:%M))"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/eval_cl_session.py --config configs/cl/ldf_${B}_s$t.yaml \
      --vae_ckpt "$VAE_CKPT" --ckpt "$(model_of "$v" "$t")" --session "$i" $EVAL --out "$f.json" \
      > "$LOG/test_$v.s$t.eval$i.log" 2>&1 || echo "[test-kd] $v after S$t on S$i: FAILED, see $LOG/test_$v.s$t.eval$i.log"
  fi
  rm -rf "$lock"
}
worker() {  # every GPU walks the same list; a test is run by whichever GPU takes its lock first
  local gpu=$1 job v t i
  for job in "${todo[@]}"; do
    read -r v t i <<< "$job"
    test_one "$v" "$t" "$i" "$gpu"
  done
}
if [ ${#todo[@]} -gt 0 ]; then
  for g in "${GPU_LIST[@]}"; do worker "$g" & done
  wait
fi

export STYLE_ENCODER_DEVICE=${STYLE_ENCODER_DEVICE:-cpu}
python scripts/cl/screen_matrix_table.py --results "$RES" --benchmark "$B" --by genre session
python scripts/cl/disco_paper_tables.py --results "$RES" --benchmark "$B"
