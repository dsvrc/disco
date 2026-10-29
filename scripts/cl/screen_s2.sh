#!/bin/bash
# S1 -> S2 continuation of an S0 -> S1 screen (scripts/cl/screen_s1.sh). Every variant whose S1 screen run is finished
# trains session 2 from its own S1 model, replaying its screen S0 buffer plus an S1 buffer built with its own S1 model
# (same selection method, 10 % of the S1 clips; csrel scores against the S0 model), and is scored on S0, S1 and S2 test
# music. Distillation variants (<variant>_kd) distil from their own S1 model. Variants run in parallel, one per GPU in
# GPUS (in waves). Variants without a finished, scored S1 screen run are skipped; finished variants are skipped on a rerun.
# Writes only into the S1 screen's folders: runs outputs/screen/<tag>/*_cl_ldf_<benchmark>_s2_<variant>, buffers
# buffers/<benchmark>/screen/<tag>/<variant>/s1.json (next to s0.json), scores results/.../<variant>/after_s2_sess{0,1,2}.json.
#
#   S0_RUN=<s0 run> S0_TAG=s0v2 PROTOCOL=v2 GPUS=0 VARIANTS="tmr_joint_neff_kd random_kd" bash scripts/cl/screen_s2.sh
#   env: the S1 screen's setup (SETUP, STEPS or the first argument, PROTOCOL, MAX_TOKENS, SAMPLES, S0_RUN + S0_TAG),
#        which must be the values the S1 screen ran with: they name the folder (<protocol>_<steps>_<tokens>_<S0_TAG>)
#        the S1 runs are read from and the S2 runs go to. VARIANTS: as in screen_s1.sh (default: tmr_dance_sel tmr_dance).
set -euo pipefail
B=aistpp_genre5
read -r -a GPU_LIST <<< "${GPUS:-0}"
VARIANTS=${VARIANTS:-"tmr_dance_sel tmr_dance"}
SETUP=${SETUP:-main}
case $SETUP in
  main) D_STEPS=40000 ; D_PROTOCOL=v1 ; D_TOKENS=full ;;
  fast) D_STEPS=20000 ; D_PROTOCOL=v2 ; D_TOKENS=76 ;;
  *) echo "SETUP must be main or fast"; exit 1 ;;
esac
STEPS=${1:-${STEPS:-$D_STEPS}}
PROTOCOL=${PROTOCOL:-$D_PROTOCOL}
MAX_TOKENS=${MAX_TOKENS:-$D_TOKENS}
SAMPLES=${SAMPLES:-3}
unset PYTHONPATH
export PYTHONNOUSERSITE=1
VAE_CKPT=${VAE_CKPT:-$(ls -d outputs/*_cl_vae_${B}_alltrain | tail -1)/last.ckpt}
[ -n "${S0_TAG:-}" ] && [ -z "${S0_RUN:-}" ] && { echo "[screen-s2] S0_TAG=$S0_TAG needs S0_RUN (the S0 model it names)"; exit 1; }
S0_RUN=${S0_RUN:-$(ls -d outputs/*_cl_ldf_${B}_s0 | tail -1)}
[ -f "$S0_RUN/last.ckpt" ] || { echo "[screen-s2] no finished S0 model: $S0_RUN/last.ckpt is missing"; exit 1; }
TAG=${PROTOCOL}_${STEPS}_${MAX_TOKENS}${S0_TAG:+_$S0_TAG}
OUT=outputs/screen/$TAG
RES=results/$B/screen/$TAG
BUFR=buffers/$B/screen
LOG=logs/$B/screen/$TAG
[ -d "$RES" ] || { echo "[screen-s2] no S1 screen with this setup: $RES is missing"; exit 1; }
PROT=""
[ "$PROTOCOL" = v2 ] && PROT="--override lr_scheduler.params.T_max=$STEPS --override model.ema_decay=0.999"
EVAL="--samples $SAMPLES"
[ "$MAX_TOKENS" != full ] && EVAL="$EVAL --max_tokens $MAX_TOKENS"
echo "[screen-s2] S1 -> S2, $STEPS steps, protocol $PROTOCOL, eval $EVAL, GPUs ${GPU_LIST[*]}, screen $TAG"
echo "[screen-s2] S0 model $S0_RUN | variants: $VARIANTS"

selection() {  # same as screen_s1.sh (keep in sync): variant -> "<selection or -> <distill true|false> [overrides]"
  case $1 in
    *_seed[0-9]*) local s ; s=$(selection "${1%_seed*}") || return 1 ; echo "$s --override seed=${1##*_seed}" ;;
    vanilla) echo "- false" ;;
    random|csrel|tmr|ocs|gcr|mir|tmgm) echo "$1 false" ;;
    random_star) echo "random false --override replay.star.enable=true" ;;
    tmr_mix|tmr_joint|tmr_joint_ess|tmr_joint_neff|tmr_joint_op|tmr_resp_neff) echo "$1 false" ;;
    tmr_dance_sel) echo "tmr_dance false" ;;
    tmr_dance) echo "tmr_dance true" ;;
    random_distill) echo "random true" ;;
    tmr_dance_gt1) echo "tmr_dance true --override replay.distill.gt_weight=1.0" ;;
    tmr_dance_lam05) echo "tmr_dance true --override replay.weight=0.5" ;;
    tmr_dance_kd) echo "tmr_dance true --override replay.distill.gt_weight=0.0" ;;
    *_kd) local s sel rest ; s=$(selection "${1%_kd}") || return 1 ; read -r sel _ rest <<< "$s"
          [ "$sel" = "-" ] && { echo "$1: no buffer to distil" >&2 ; return 1 ; }
          echo "$sel true $rest" ;;
    *) echo "unknown variant $1" >&2; return 1 ;;
  esac
}

for v in $VARIANTS; do selection "$v" > /dev/null || exit 1; done

s1_run() {  # the variant's finished S1 screen run (empty when there is none)
  { ls -d "$OUT"/*_cl_ldf_${B}_s1_$1 2>/dev/null || true; } | tail -1
}

# 1. S1 buffer (built with the variant's own S1 model) + S2 training + scoring on S0, S1, S2
run_variant() {
  local v=$1 gpu=$2 SEL DIST VX REPLAY S1 ck PREV_ARG
  read -r SEL DIST VX <<< "$(selection "$v")"
  S1=$(s1_run "$v")
  REPLAY=${VX:-}
  if [ "$SEL" != "-" ]; then
    if [ ! -f "$BUFR/$TAG/$v/s1.json" ]; then
      PREV_ARG=""
      [ "$SEL" = csrel ] && PREV_ARG="--prev_ckpt $S0_RUN/last.ckpt"  # csrel: the model before S1
      echo "[screen-s2] $v: building the S1 buffer ($SEL) on GPU $gpu"
      CUDA_VISIBLE_DEVICES=$gpu python scripts/build_replay_buffer.py --method "$SEL" \
        --config configs/cl/ldf_${B}_s1.yaml --ckpt "$S1/last.ckpt" $PREV_ARG --session 1 \
        --out "$BUFR/$TAG/$v/.tmp_s1.json" > "$LOG/buffer_s1_$v.log" 2>&1 \
        || { echo "[screen-s2] $v: S1 buffer FAILED, see $LOG/buffer_s1_$v.log"; return 0; }
      mv -f "$BUFR/$TAG/$v/.tmp_s1.json" "$BUFR/$TAG/$v/s1.json"  # only a complete buffer gets the name train_ldf reads
      grep -E "^\[|^saved" "$LOG/buffer_s1_$v.log" || true
    fi
    REPLAY="--override replay.buffer_dir=$BUFR/$TAG/$v --override replay.distill.enable=$DIST $REPLAY"
  fi
  echo "[screen-s2] $v: training S2 on GPU $gpu from $S1 ($(date +%H:%M))"
  CUDA_VISIBLE_DEVICES=$gpu python train_ldf.py --config configs/cl/ldf_${B}_s2.yaml --override train=True \
    --override exp_name=cl_ldf_${B}_s2_${v} --override save_dir=./$OUT --override test_vae_ckpt="$VAE_CKPT" \
    --override init_ckpt="$S1/last.ckpt" --override trainer.max_steps="$STEPS" \
    --override validation.validation_steps=1000000 --override validation.save_every_n_steps=1000000 \
    $PROT $REPLAY > "$LOG/$v.s2.train.log" 2>&1 || { echo "[screen-s2] $v: training FAILED, see $LOG/$v.s2.train.log"; return 0; }
  ck=$(ls -d "$OUT"/*_cl_ldf_${B}_s2_${v} | tail -1)/last.ckpt
  for i in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$gpu python scripts/eval_cl_session.py --config configs/cl/ldf_${B}_s2.yaml \
      --vae_ckpt "$VAE_CKPT" --ckpt "$ck" --session $i $EVAL --out "$RES/$v/after_s2_sess$i.json" \
      > "$LOG/$v.s2.eval$i.log" 2>&1 || echo "[screen-s2] $v: scoring S$i FAILED, see $LOG/$v.s2.eval$i.log"
  done
  echo "[screen-s2] $v: done ($(date +%H:%M))"
}

s2_done() { local i ; for i in 0 1 2; do [ -f "$RES/$1/after_s2_sess$i.json" ] || return 1; done; }

# several screen_s2.sh may run at once (e.g. one started early on a free GPU, one queued after the S1 screens): a
# variant is trained by whichever takes its lock first; a lock whose process is gone (killed run) is taken over
run_locked() {
  local v=$1 lock=$RES/$1/.s2_lock pid
  if ! mkdir "$lock" 2>/dev/null; then
    [ -s "$lock/pid" ] || sleep 2  # the owner may not have written its pid yet
    pid=$(cat "$lock/pid" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "[screen-s2] $v: skipped, running in another screen_s2.sh (pid $pid)"; return 0
    fi
    echo "[screen-s2] $v: taking over the lock of a stopped run"
  fi
  echo "$BASHPID" > "$lock/pid"
  if s2_done "$v"; then echo "[screen-s2] $v: done meanwhile by another screen_s2.sh"; else run_variant "$@"; fi
  rm -rf "$lock"
}

todo=()
for v in $VARIANTS; do
  s2_done "$v" && { echo "[screen-s2] $v: already done"; continue; }
  S1=$(s1_run "$v")
  if [ -z "$S1" ] || [ ! -f "$S1/last.ckpt" ] || [ ! -f "$RES/$v/after_s1_sess0.json" ] \
      || [ ! -f "$RES/$v/after_s1_sess1.json" ]; then
    echo "[screen-s2] $v: skipped, no finished and scored S1 screen run under $OUT"
    continue
  fi
  todo+=("$v")
done
n=${#GPU_LIST[@]}
for ((w = 0; w < ${#todo[@]}; w += n)); do  # waves of one variant per GPU
  for ((g = 0; g < n && w + g < ${#todo[@]}; g++)); do
    run_locked "${todo[$((w + g))]}" "${GPU_LIST[$g]}" &
  done
  wait
done

python scripts/cl/screen_summary_s2.py --results "$RES" --buffers "$BUFR/$TAG" | tee "$RES/.summary_s2_$$.md"
mv -f "$RES/.summary_s2_$$.md" "$RES/summary_s2.md"  # the last screen to finish leaves the table with every variant
