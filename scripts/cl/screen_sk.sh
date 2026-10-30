#!/bin/bash
# Session k-1 -> k continuation of an S0 -> S1 screen (scripts/cl/screen_s1.sh), for any k >= 2 (SESSION=k). Every
# variant whose screen run of session k-1 is finished and scored trains session k from that model, replaying its screen
# buffers of sessions 0..k-2 plus a session k-1 buffer built with its own model after k-1 (same selection method, 10 %
# of the clips; csrel scores against its model after k-2), and is scored on sessions 0..k. Distillation variants
# (<variant>_kd) distil from their own model after k-1. Variants run one per GPU in GPUS (in waves). SESSION=2 does
# exactly what scripts/cl/screen_s2.sh does (same folders, same locks), so the two can share a screen.
# Several screens may run at once, also with the same variant list (e.g. one per GPU): a variant is trained by whichever
# takes its lock first, so the variant list works as a shared queue. Finished variants are skipped on a rerun; a
# variant whose session k failed is not retried by the same run.
# Writes only into the S1 screen's folders: runs outputs/screen/<tag>/*_cl_ldf_<benchmark>_s<k>_<variant>, buffers
# buffers/<benchmark>/screen/<tag>/<variant>/s<k-1>.json, scores results/.../<variant>/after_s<k>_sess{0..k}.json.
#
#   SESSION=3 S0_RUN=<s0 run> S0_TAG=s0v2 PROTOCOL=v2 GPUS=0 VARIANTS="tmr_joint_neff_kd random_kd" bash scripts/cl/screen_sk.sh
#   env: SESSION   the session to train (2..4)
#        WAIT      hours to keep waiting for variants whose session k-1 run is not finished yet (polled every POLL
#                  seconds, default 600); the wait restarts whenever a variant became ready. 0 (default): no waiting.
#        the S1 screen's setup (SETUP, STEPS or the first argument, PROTOCOL, MAX_TOKENS, SAMPLES, S0_RUN + S0_TAG),
#        which must be the values the S1 screen ran with: they name the folder (<protocol>_<steps>_<tokens>_<S0_TAG>)
#        the earlier runs are read from and the new ones go to. VARIANTS: as in screen_s1.sh.
set -euo pipefail
B=aistpp_genre5
K=${SESSION:?set SESSION to the session to train (2..4)}
[ "$K" -ge 2 ] && [ -f "configs/cl/ldf_${B}_s$K.yaml" ] || { echo "SESSION must be 2..4"; exit 1; }
J=$((K - 1))
read -r -a GPU_LIST <<< "${GPUS:-0}"
VARIANTS=${VARIANTS:-"tmr_dance_sel tmr_dance"}
WAIT=${WAIT:-0}
POLL=${POLL:-600}
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
[ -n "${S0_TAG:-}" ] && [ -z "${S0_RUN:-}" ] && { echo "[screen-s$K] S0_TAG=$S0_TAG needs S0_RUN (the S0 model it names)"; exit 1; }
S0_RUN=${S0_RUN:-$(ls -d outputs/*_cl_ldf_${B}_s0 | tail -1)}
[ -f "$S0_RUN/last.ckpt" ] || { echo "[screen-s$K] no finished S0 model: $S0_RUN/last.ckpt is missing"; exit 1; }
TAG=${PROTOCOL}_${STEPS}_${MAX_TOKENS}${S0_TAG:+_$S0_TAG}
OUT=outputs/screen/$TAG
RES=results/$B/screen/$TAG
BUFR=buffers/$B/screen
LOG=logs/$B/screen/$TAG
[ -d "$RES" ] || { echo "[screen-s$K] no S1 screen with this setup: $RES is missing"; exit 1; }
PROT=""
[ "$PROTOCOL" = v2 ] && PROT="--override lr_scheduler.params.T_max=$STEPS --override model.ema_decay=0.999"
EVAL="--samples $SAMPLES"
[ "$MAX_TOKENS" != full ] && EVAL="$EVAL --max_tokens $MAX_TOKENS"
echo "[screen-s$K] S$J -> S$K, $STEPS steps, protocol $PROTOCOL, eval $EVAL, GPUs ${GPU_LIST[*]}, screen $TAG, wait ${WAIT} h"
echo "[screen-s$K] S0 model $S0_RUN | variants: $VARIANTS"

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

run_dir() {  # the variant's screen run after session $2 (session 0: the shared S0 model); empty when there is none
  local v=$1 s=$2
  if [ "$s" -eq 0 ]; then echo "$S0_RUN"; else { ls -d "$OUT"/*_cl_ldf_${B}_s${s}_${v} 2>/dev/null || true; } | tail -1; fi
}

scored() {  # the variant's model after session $2 is scored on every session 0..$2
  local i ; for i in $(seq 0 "$2"); do [ -f "$RES/$1/after_s$2_sess$i.json" ] || return 1; done
}

state() {  # done | ready | wait (session k-1 not finished yet) | broken (an earlier buffer is missing)
  local v=$1 d SEL i
  scored "$v" "$K" && { echo done; return; }
  d=$(run_dir "$v" "$J")
  { [ -n "$d" ] && [ -f "$d/last.ckpt" ] && scored "$v" "$J"; } || { echo wait; return; }
  read -r SEL _ <<< "$(selection "$v")"
  if [ "$SEL" != "-" ]; then
    for i in $(seq 0 $((J - 1))); do [ -f "$BUFR/$TAG/$v/s$i.json" ] || { echo broken; return; }; done
  fi
  echo ready
}

# 1. session k-1 buffer (built with the variant's own model after k-1) + session k training + scoring on 0..k
run_variant() {
  local v=$1 gpu=$2 SEL DIST VX REPLAY PREV ck PREV_ARG i
  read -r SEL DIST VX <<< "$(selection "$v")"
  PREV=$(run_dir "$v" "$J")
  REPLAY=${VX:-}
  if [ "$SEL" != "-" ]; then
    if [ ! -f "$BUFR/$TAG/$v/s$J.json" ]; then
      PREV_ARG=""
      [ "$SEL" = csrel ] && PREV_ARG="--prev_ckpt $(run_dir "$v" $((J - 1)))/last.ckpt"  # csrel: the model before S$J
      echo "[screen-s$K] $v: building the S$J buffer ($SEL) on GPU $gpu"
      CUDA_VISIBLE_DEVICES=$gpu python scripts/build_replay_buffer.py --method "$SEL" \
        --config configs/cl/ldf_${B}_s$J.yaml --ckpt "$PREV/last.ckpt" $PREV_ARG --session $J \
        --out "$BUFR/$TAG/$v/.tmp_s$J.json" > "$LOG/buffer_s${J}_$v.log" 2>&1 \
        || { echo "[screen-s$K] $v: S$J buffer FAILED, see $LOG/buffer_s${J}_$v.log"; return 0; }
      mv -f "$BUFR/$TAG/$v/.tmp_s$J.json" "$BUFR/$TAG/$v/s$J.json"  # only a complete buffer gets the name train_ldf reads
      grep -E "^\[|^saved" "$LOG/buffer_s${J}_$v.log" || true
    fi
    REPLAY="--override replay.buffer_dir=$BUFR/$TAG/$v --override replay.distill.enable=$DIST $REPLAY"
  fi
  echo "[screen-s$K] $v: training S$K on GPU $gpu from $PREV ($(date +%H:%M))"
  CUDA_VISIBLE_DEVICES=$gpu python train_ldf.py --config configs/cl/ldf_${B}_s$K.yaml --override train=True \
    --override exp_name=cl_ldf_${B}_s${K}_${v} --override save_dir=./$OUT --override test_vae_ckpt="$VAE_CKPT" \
    --override init_ckpt="$PREV/last.ckpt" --override trainer.max_steps="$STEPS" \
    --override validation.validation_steps=1000000 --override validation.save_every_n_steps=1000000 \
    $PROT $REPLAY > "$LOG/$v.s$K.train.log" 2>&1 || { echo "[screen-s$K] $v: training FAILED, see $LOG/$v.s$K.train.log"; return 0; }
  ck=$(run_dir "$v" "$K")/last.ckpt
  for i in $(seq 0 "$K"); do
    CUDA_VISIBLE_DEVICES=$gpu python scripts/eval_cl_session.py --config configs/cl/ldf_${B}_s$K.yaml \
      --vae_ckpt "$VAE_CKPT" --ckpt "$ck" --session $i $EVAL --out "$RES/$v/after_s${K}_sess$i.json" \
      > "$LOG/$v.s$K.eval$i.log" 2>&1 || echo "[screen-s$K] $v: scoring S$i FAILED, see $LOG/$v.s$K.eval$i.log"
  done
  echo "[screen-s$K] $v: done ($(date +%H:%M))"
}

# a variant is trained by whichever screen takes its lock first; a lock whose process is gone (killed run) is taken over
run_locked() {
  local v=$1 lock=$RES/$1/.s${K}_lock pid
  if ! mkdir "$lock" 2>/dev/null; then
    [ -s "$lock/pid" ] || sleep 2  # the owner may not have written its pid yet
    pid=$(cat "$lock/pid" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "[screen-s$K] $v: skipped, running in another screen (pid $pid)"; return 0
    fi
    echo "[screen-s$K] $v: taking over the lock of a stopped run"
  fi
  echo "$BASHPID" > "$lock/pid"
  if scored "$v" "$K"; then echo "[screen-s$K] $v: done meanwhile by another screen"; else run_variant "$@"; fi
  rm -rf "$lock"
}

declare -A SEEN=()  # variants this run has finished with (done, trained, failed, broken)
first=true waited=0 limit=$(awk "BEGIN { print int(${WAIT} * 3600) }")
while :; do
  todo=() waiting=()
  for v in $VARIANTS; do
    [ -n "${SEEN[$v]:-}" ] && continue
    case $(state "$v") in
      done) echo "[screen-s$K] $v: already done"; SEEN[$v]=1 ;;
      broken) echo "[screen-s$K] $v: skipped, its buffer of an earlier session is missing in $BUFR/$TAG/$v"; SEEN[$v]=1 ;;
      wait) $first && echo "[screen-s$K] $v: not ready, no finished and scored S$J screen run under $OUT"; waiting+=("$v") ;;
      ready) todo+=("$v"); SEEN[$v]=1 ;;
    esac
  done
  first=false
  n=${#GPU_LIST[@]}
  for ((w = 0; w < ${#todo[@]}; w += n)); do  # waves of one variant per GPU
    for ((g = 0; g < n && w + g < ${#todo[@]}; g++)); do
      run_locked "${todo[$((w + g))]}" "${GPU_LIST[$g]}" &
    done
    wait
  done
  [ ${#waiting[@]} -eq 0 ] && break
  [ ${#todo[@]} -gt 0 ] && waited=0
  if [ "$waited" -ge "$limit" ]; then
    echo "[screen-s$K] not trained, their S$J screen run is not finished: ${waiting[*]}"
    break
  fi
  sleep "$POLL"
  waited=$((waited + POLL))
done

python scripts/cl/screen_summary_sk.py --session "$K" --results "$RES" --buffers "$BUFR/$TAG" | tee "$RES/.summary_s${K}_$$.md"
mv -f "$RES/.summary_s${K}_$$.md" "$RES/summary_s$K.md"  # the last screen to finish leaves the table with every variant
