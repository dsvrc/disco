#!/bin/bash
# Continual-learning run of one method on aistpp_genre5 (TMR benchmark), from session 1 to session 4:
#   S0 (shared, already trained) ; S1 + buffer(S0) ; S2 + buffer(S0) + buffer(S1) ; ... ; S4 + buffers(S0..S3)
# For every session k: build the buffer of session k-1 with the model after k-1 (10 % of its training clips),
# train session k warm-started from k-1 (fresh optimizer / LR schedule) with replay, score every seen session with
# the style metric, and print the forgetting table. Finished steps are skipped, so a crashed run can be restarted
# with the same command.
#
#   bash scripts/cl/run_method.sh <method> [first_session=1] [last_session=4]
#   methods: vanilla | random | random_star | random_distill | ocs | gcr | tmr | tmr_mix | tmr_joint | tmr_joint_ess |
#            tmr_joint_neff | tmr_joint_op | tmr_dance | tmr_dance_sel | tmgm | mir | csrel
#   (tmr_mix = the paper's TMR per genre with same-music dances pooled and drawn at random; tmr_joint / tmr_joint_ess =
#   the paper's TMR on the joint music + dance trajectory; tmr_joint_neff = with the effective sample size in its BIC;
#   tmr_joint_op = tmr_joint_neff matching the temporal operator as well as the mean; all with ground-truth replay)
#   (tmr_dance = TMR-dance selection + distillation replay; tmr_dance_sel = TMR-dance selection + ground-truth
#   replay like every baseline; random_distill = random selection + distillation)
#
# Environment overrides: VAE_CKPT, S0_RUN (output folder of the shared session-0 run), RATIO (0.10), SAMPLES (3),
# MAX_TOKENS (empty = full-length evaluation), EXTRA (more train_ldf.py overrides, e.g. "--override trainer.max_steps=20000").
set -euo pipefail
METHOD=${1:?usage: run_method.sh <method> [first_session] [last_session]}
FIRST=${2:-1}
LAST=${3:-4}
B=aistpp_genre5
RATIO=${RATIO:-0.10}
SAMPLES=${SAMPLES:-3}
MAX_TOKENS=${MAX_TOKENS:-}
EXTRA=${EXTRA:-}

unset PYTHONPATH
export PYTHONNOUSERSITE=1
VAE_CKPT=${VAE_CKPT:-$(ls -d outputs/*_cl_vae_${B}_alltrain | tail -1)/last.ckpt}
S0_RUN=${S0_RUN:-$(ls -d outputs/*_cl_ldf_${B}_s0 | tail -1)}

case $METHOD in
  vanilla) SELECT= ; STAR=false ; DISTILL=false ;;
  random) SELECT=random ; STAR=false ; DISTILL=false ;;
  random_star) SELECT=random ; STAR=true ; DISTILL=false ;;
  random_distill) SELECT=random ; STAR=false ; DISTILL=true ;;
  tmr_dance) SELECT=tmr_dance ; STAR=false ; DISTILL=true ;;
  tmr_dance_sel) SELECT=tmr_dance ; STAR=false ; DISTILL=false ;;
  tmr_mix|tmr_joint|tmr_joint_ess|tmr_joint_neff|tmr_joint_op) SELECT=$METHOD ; STAR=false ; DISTILL=false ;;
  ocs|gcr|tmr|tmgm|mir|csrel) SELECT=$METHOD ; STAR=false ; DISTILL=false ;;
  *) echo "unknown method $METHOD"; exit 1 ;;
esac

SUFFIX=_$METHOD
[ "$METHOD" = vanilla ] && SUFFIX=""  # same run names as the earlier vanilla runs (run_vanilla.sh), which are reused
R=results/$B/$METHOD
BUF=buffers/$B/$METHOD
LOG=logs/$B/$METHOD
mkdir -p "$R" "$BUF" "$LOG"
EVAL_ARGS="--samples $SAMPLES"
[ -n "$MAX_TOKENS" ] && EVAL_ARGS="$EVAL_ARGS --max_tokens $MAX_TOKENS"
echo "[$METHOD] sessions $FIRST..$LAST | VAE $VAE_CKPT | S0 $S0_RUN | buffer ${RATIO} per session | eval $EVAL_ARGS"
[ -f "$VAE_CKPT" ] && [ -f "$S0_RUN/last.ckpt" ] || { echo "missing VAE or S0 checkpoint"; exit 1; }

run_dir() {  # output folder of the model after session $1 for this method
  if [ "$1" -eq 0 ]; then echo "$S0_RUN"; else { ls -d outputs/*_cl_ldf_${B}_s$1${SUFFIX} 2>/dev/null || true; } | tail -1; fi
}

evaluate() {  # score every seen session 0..$1 with the model after session $1
  local k=$1 ck
  ck=$(run_dir "$k")/last.ckpt
  for i in $(seq 0 "$k"); do
    local out=$R/after_s${k}_sess$i.json
    [ -f "$out" ] && continue
    if [ "$k" -eq 0 ] && [ -f "results/$B/vanilla/after_s0_sess$i.json" ] && [ -z "$MAX_TOKENS" ]; then
      cp "results/$B/vanilla/after_s0_sess$i.json" "$out"  # shared session-0 model, already scored
      continue
    fi
    python scripts/eval_cl_session.py --config configs/cl/ldf_${B}_s$k.yaml --vae_ckpt "$VAE_CKPT" --ckpt "$ck" \
      --session "$i" $EVAL_ARGS --out "$out" > "$LOG/eval_after_s${k}_sess$i.log" 2>&1
    echo "[$METHOD] scored session $i after S$k -> $out"
  done
}

evaluate 0
for k in $(seq "$FIRST" "$LAST"); do
  j=$((k - 1))
  PREV=$(run_dir "$j")
  [ -f "$PREV/last.ckpt" ] || { echo "no model after session $j for $METHOD ($PREV)"; exit 1; }

  REPLAY=""
  if [ -n "$SELECT" ]; then
    if [ ! -f "$BUF/s$j.json" ]; then
      PREV_ARG=""
      if [ "$SELECT" = csrel ] && [ "$j" -gt 0 ]; then PREV_ARG="--prev_ckpt $(run_dir $((j - 1)))/last.ckpt"; fi
      echo "[$METHOD] building buffer of S$j"
      python scripts/build_replay_buffer.py --method "$SELECT" --config configs/cl/ldf_${B}_s$j.yaml \
        --ckpt "$PREV/last.ckpt" $PREV_ARG --session "$j" --ratio "$RATIO" --out "$BUF/s$j.json" \
        > "$LOG/buffer_s$j.log" 2>&1
      tail -1 "$LOG/buffer_s$j.log"
    fi
    cp "$BUF/s$j.json" "$R/buffer_s$j.json"  # keep a copy next to the results (the buffer itself lives in $BUF)
    REPLAY="--override replay.buffer_dir=$BUF --override replay.star.enable=$STAR --override replay.distill.enable=$DISTILL"
  fi

  # a session counts as trained once its own test set has been scored (also covers earlier runs of this method)
  if [ ! -f "$R/.trained_s$k" ] && [ ! -f "$R/after_s${k}_sess$k.json" ]; then
    echo "[$METHOD] training S$k from $PREV ($(date))"
    python train_ldf.py --config configs/cl/ldf_${B}_s$k.yaml --override train=True \
      --override exp_name=cl_ldf_${B}_s${k}${SUFFIX} --override test_vae_ckpt="$VAE_CKPT" \
      --override init_ckpt="$PREV/last.ckpt" $REPLAY $EXTRA > "$LOG/train_s$k.log" 2>&1
    [ -f "$(run_dir "$k")/last.ckpt" ] || { echo "training S$k produced no checkpoint, see $LOG/train_s$k.log"; exit 1; }
    touch "$R/.trained_s$k"
  fi

  evaluate "$k"
  python scripts/cl_forgetting_table.py --results_dir "$R" | tee "$R/table_after_s$k.txt"
done
echo "[$METHOD] done ($(date))"
