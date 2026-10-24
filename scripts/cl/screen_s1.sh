#!/bin/bash
# S0 -> S1 screen of replay variants. Every variant trains session 1 from the shared S0 model and is scored on S0 (old)
# and S1 (new) test music. Variants run in parallel, one per GPU in GPUS (in waves when there are more variants than
# GPUs). Finished variants are skipped on a rerun.
# Writes only under outputs/screen/, buffers/<benchmark>/screen/, results/<benchmark>/screen/, logs/<benchmark>/screen/.
#
#   GPUS="0 1" bash scripts/cl/screen_s1.sh
#   env: VARIANTS   (default: tmr_dance_sel tmr_dance)
#        SETUP      main (default) = exactly the main runs' setup: 40k steps, the configs' lr schedule, whole test
#                   dances, 3 samples. The baselines' existing S1 scores (results/<benchmark>/<method>/after_s1_*.json)
#                   and the S0 model's scores are reused as comparison rows, so no baseline is retrained.
#                   fast = 20k steps, one cosine decay + EMA 0.999 (protocol v2), first 10 s of every test dance;
#                   nothing from the main runs is comparable, so put the baselines in VARIANTS too.
#        STEPS (also the first argument), PROTOCOL (v1 | v2), MAX_TOKENS (a number | full), SAMPLES override the
#        setup's values; the baselines are only reused when the result still matches the main runs.
#   variants: vanilla | random | csrel | tmr | ocs | gcr | mir      (existing selections, ground-truth replay)
#             tmr_dance_sel   TMR-dance selection, ground-truth replay
#             tmr_dance       TMR-dance selection + distillation replay
#             random_distill  random selection + distillation replay
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
S0_RUN=${S0_RUN:-$(ls -d outputs/*_cl_ldf_${B}_s0 | tail -1)}
TAG=${PROTOCOL}_${STEPS}_${MAX_TOKENS}
OUT=outputs/screen/$TAG
RES=results/$B/screen/$TAG
BUFR=buffers/$B/screen
LOG=logs/$B/screen/$TAG
mkdir -p "$OUT" "$RES" "$BUFR" "$LOG"
PROT=""
[ "$PROTOCOL" = v2 ] && PROT="--override lr_scheduler.params.T_max=$STEPS --override model.ema_decay=0.999"
EVAL="--samples $SAMPLES"
[ "$MAX_TOKENS" != full ] && EVAL="$EVAL --max_tokens $MAX_TOKENS"
MAIN_MATCH=false  # same steps, schedule and scoring as the main runs: their S0 / S1 scores are comparable
[ "$PROTOCOL" = v1 ] && [ "$STEPS" = 40000 ] && [ "$MAX_TOKENS" = full ] && [ "$SAMPLES" = 3 ] && MAIN_MATCH=true
echo "[screen] S0 -> S1, $STEPS steps, protocol $PROTOCOL, eval $EVAL, GPUs ${GPU_LIST[*]}, matches the main runs: $MAIN_MATCH"
echo "[screen] S0 model $S0_RUN | variants: $VARIANTS"

selection() {  # variant -> "<selection method or -> <distill true|false>"
  case $1 in
    vanilla) echo "- false" ;;
    random|csrel|tmr|ocs|gcr|mir) echo "$1 false" ;;
    tmr_dance_sel) echo "tmr_dance false" ;;
    tmr_dance) echo "tmr_dance true" ;;
    random_distill) echo "random true" ;;
    *) echo "unknown variant $1" >&2; return 1 ;;
  esac
}

for v in $VARIANTS; do selection "$v" > /dev/null || exit 1; done

# 1. S0 buffers (one per selection method; reuses the real run's buffer of the same S0 model when there is one)
for v in $VARIANTS; do
  read -r SEL _ <<< "$(selection "$v")"
  [ "$SEL" = "-" ] && continue
  DST=$BUFR/$TAG/$v/s0.json  # every variant has its own buffer folder: parallel screens never write the same file
  [ -f "$DST" ] && continue
  mkdir -p "$BUFR/$TAG/$v"
  if [ -f "buffers/$B/$SEL/s0.json" ]; then
    cp "buffers/$B/$SEL/s0.json" "$DST"  # the main run's buffer of the same S0 model
  else
    echo "[screen] building the S0 buffer of $v ($SEL)"
    CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} python scripts/build_replay_buffer.py --method "$SEL" \
      --config configs/cl/ldf_${B}_s0.yaml --ckpt "$S0_RUN/last.ckpt" --session 0 --out "$DST" > "$LOG/buffer_$v.log" 2>&1
    grep -E "^\[|^saved" "$LOG/buffer_$v.log" || true
  fi
done

# 2. the S0 model on the same eval setting (the "before S1" reference; the main runs' scores when they match)
mkdir -p "$RES/s0_model"
for i in 0 1; do
  ref=$RES/s0_model/after_s0_sess$i.json
  [ -f "$ref" ] && continue
  tmp=$RES/s0_model/.tmp_$$_after_s0_sess$i.json  # written under a private name, then renamed (atomic)
  if [ "$MAIN_MATCH" = true ] && [ -f "results/$B/vanilla/after_s0_sess$i.json" ]; then
    cp "results/$B/vanilla/after_s0_sess$i.json" "$tmp"
  else
    CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} python scripts/eval_cl_session.py --config configs/cl/ldf_${B}_s0.yaml \
      --vae_ckpt "$VAE_CKPT" --ckpt "$S0_RUN/last.ckpt" --session $i $EVAL --out "$tmp" > "$LOG/s0_model.eval$i.$$.log" 2>&1
  fi
  mv -f "$tmp" "$ref"
  rm -f "${tmp%.json}.joints.npz"
done

# 3. train + score every variant
run_variant() {
  local v=$1 gpu=$2 SEL DIST REPLAY="" ck
  read -r SEL DIST <<< "$(selection "$v")"
  [ "$SEL" != "-" ] && REPLAY="--override replay.buffer_dir=$BUFR/$TAG/$v --override replay.distill.enable=$DIST"
  echo "[screen] $v: training on GPU $gpu ($(date +%H:%M))"
  CUDA_VISIBLE_DEVICES=$gpu python train_ldf.py --config configs/cl/ldf_${B}_s1.yaml --override train=True \
    --override exp_name=cl_ldf_${B}_s1_${v} --override save_dir=./$OUT --override test_vae_ckpt="$VAE_CKPT" \
    --override init_ckpt="$S0_RUN/last.ckpt" --override trainer.max_steps="$STEPS" \
    --override validation.validation_steps=1000000 --override validation.save_every_n_steps=1000000 \
    $PROT $REPLAY > "$LOG/$v.train.log" 2>&1 || { echo "[screen] $v: training FAILED, see $LOG/$v.train.log"; return 0; }
  ck=$(ls -d "$OUT"/*_cl_ldf_${B}_s1_${v} | tail -1)/last.ckpt
  mkdir -p "$RES/$v"
  for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$gpu python scripts/eval_cl_session.py --config configs/cl/ldf_${B}_s1.yaml \
      --vae_ckpt "$VAE_CKPT" --ckpt "$ck" --session $i $EVAL --out "$RES/$v/after_s1_sess$i.json" \
      > "$LOG/$v.eval$i.log" 2>&1 || echo "[screen] $v: scoring S$i FAILED, see $LOG/$v.eval$i.log"
  done
  echo "[screen] $v: done ($(date +%H:%M))"
}

todo=()
for v in $VARIANTS; do
  [ -f "$RES/$v/after_s1_sess1.json" ] && [ -f "$RES/$v/after_s1_sess0.json" ] && { echo "[screen] $v: already done"; continue; }
  todo+=("$v")
done
n=${#GPU_LIST[@]}
for ((w = 0; w < ${#todo[@]}; w += n)); do  # waves of one variant per GPU
  for ((g = 0; g < n && w + g < ${#todo[@]}; g++)); do
    run_variant "${todo[$((w + g))]}" "${GPU_LIST[$g]}" &
  done
  wait
done

BASE=""
[ "$MAIN_MATCH" = true ] && BASE="--baselines results/$B --baseline_buffers buffers/$B"
python scripts/cl/screen_summary.py --results "$RES" --logs "$LOG" --buffers "$BUFR/$TAG" $BASE | tee "$RES/.summary_$$.md"
mv -f "$RES/.summary_$$.md" "$RES/summary.md"  # the last screen to finish leaves the table with every variant
