#!/bin/bash
# Diagnostic run of one replay method: audit its buffers, then train session K for a few thousand steps from that
# method's model after K-1 with dense [replay-diag] logs (batch mix, cumulative samples per session, loss and gradient
# shares of the current vs replay terms, buffer coverage, fixed-noise probe losses on buffered vs non-buffered clips
# of every previous session). Writes only to outputs/diag/ (one level below the real runs, and excluded from the code snapshot of
# train_ldf.py) and logs/<benchmark>/<method>/diag_*.log, never to the
# real results or checkpoints.
#
#   bash scripts/cl/diag_replay.sh <method> [session=4] [steps=2000]
#   env: DIAG_EVERY (100), BALANCE (uniform | session), RATIO (0.10, the budget the audit checks against),
#        EXTRA (more train_ldf.py overrides)
set -euo pipefail
METHOD=${1:?usage: diag_replay.sh <method> [session] [steps]}
K=${2:-4}
STEPS=${3:-2000}
B=aistpp_genre5
DIAG_EVERY=${DIAG_EVERY:-100}
BALANCE=${BALANCE:-uniform}
RATIO=${RATIO:-0.10}
EXTRA=${EXTRA:-}
unset PYTHONPATH
export PYTHONNOUSERSITE=1
VAE_CKPT=${VAE_CKPT:-$(ls -d outputs/*_cl_vae_${B}_alltrain | tail -1)/last.ckpt}
S0_RUN=${S0_RUN:-$(ls -d outputs/*_cl_ldf_${B}_s0 | tail -1)}
case $METHOD in
  random_star) STAR=true ; DISTILL=false ;;
  random_distill|tmr_dance) STAR=false ; DISTILL=true ;;
  random|ocs|gcr|tmr|tmr_mix|tmr_joint|tmr_joint_ess|tmr_joint_neff|tmr_joint_op|tmr_dance_sel|tmgm|mir|csrel) STAR=false ; DISTILL=false ;;
  *) echo "diag_replay.sh needs a replay method (random, random_star, random_distill, ocs, gcr, tmr, tmr_mix, tmr_joint, tmr_joint_ess, tmr_joint_neff, tmr_joint_op, tmr_dance, tmr_dance_sel, tmgm, mir, csrel)"; exit 1 ;;
esac
BUF=buffers/$B/$METHOD
LOG=logs/$B/$METHOD
mkdir -p "$LOG"
if [ "$K" -eq 1 ]; then PREV=$S0_RUN; else PREV=$({ ls -d outputs/*_cl_ldf_${B}_s$((K - 1))_${METHOD} 2>/dev/null || true; } | tail -1); fi
[ -f "$PREV/last.ckpt" ] || { echo "no model after S$((K - 1)) for $METHOD"; exit 1; }

echo "=== buffer audit ($METHOD) -> $LOG/diag_audit.log"
python scripts/cl/audit_buffers.py --benchmark $B --methods "$METHOD" --ratio "$RATIO" | tee "$LOG/diag_audit.log" || true

OUT=$LOG/diag_s${K}_${BALANCE}.log
echo "=== $STEPS-step diagnostic run of S$K from $PREV (balance=$BALANCE, block every $DIAG_EVERY steps) -> $OUT"
if ! python train_ldf.py --config configs/cl/ldf_${B}_s$K.yaml --override train=True \
  --override exp_name=cl_ldf_${B}_s${K}_${METHOD}_diag --override save_dir=./outputs/diag \
  --override test_vae_ckpt="$VAE_CKPT" --override init_ckpt="$PREV/last.ckpt" \
  --override replay.buffer_dir="$BUF" --override replay.star.enable=$STAR --override replay.distill.enable=$DISTILL --override replay.balance=$BALANCE \
  --override replay.diag_every="$DIAG_EVERY" --override replay.diag_first=3 \
  --override trainer.max_steps="$STEPS" --override validation.validation_steps=1000000 \
  --override validation.save_every_n_steps=1000000 $EXTRA > "$OUT" 2>&1; then
  echo "=== the diagnostic run FAILED; last lines of $OUT:"
  tail -25 "$OUT"
  exit 1
fi
grep -a "\[replay-diag\]" "$OUT" > "$LOG/diag_s${K}_${BALANCE}.replay_diag.txt" || true
echo "=== startup and the last [replay-diag] block (all blocks: $LOG/diag_s${K}_${BALANCE}.replay_diag.txt)"
grep -a "\[replay-diag\]" "$OUT" | awk '/replay-diag\] step/{exit} {print}'
grep -a "\[replay-diag\]" "$OUT" | awk '/replay-diag\] step/{blk=""} {blk=blk $0 "\n"} END{printf "%s", blk}'
echo "=== every WARN / FAIL line"
grep -a "\[replay-diag\]" "$OUT" | grep -E "WARN|FAIL|MEMORISING|FORGETTING" | sort | uniq -c | sort -rn | head -10 || true
