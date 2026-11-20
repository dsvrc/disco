#!/bin/bash
# Everything the TMR supplementary still needs from the DiscoForcing runs (seeds/variance and the user studies are not
# covered), in one folder and one archive to bring back to the laptop:
#   results/aistpp_genre5/supp/SUPP_FILL.md          numbers per red item of sec/X_suppl.tex (+ supp_collect.json)
#   results/aistpp_genre5/supp/theory/THEORY.md      fast checks of the analysis (+ theory.json)
#   results/aistpp_genre5/supp_collect.tar.gz        both folders without the generated joints
#
# Steps (each is skipped when its output exists, so the script can be rerun after a failure):
#   1. sanity scores on the final pairing (scripts/eval_cl_session.py, one job per GPU at a time): real dance (--gt) and
#      autoencoder reconstructions (--vae_recon) of every session, first 10 s and full length; an untrained generator
#      (--untrained) on every session; the shared S0 model after 2,500 of its steps on S0 and S4 music (first 10 s)
#   2. python -m metrics.style_encoder validate --dataset aistpp (CPU; the untrained generator's output is its
#      "untrained generator" reject probe); the previous validation.json is kept as supp/validation_prev.json
#   3. scripts/supp/theory_fast.py on the first GPU (in parallel with 2)
#   4. scripts/supp/collect_disco_supp.py (CPU): sessions, protocol, buffers, Tab. sr_snr, Tab. sr_valid, tables
#   5. the archive
#
#   GPUS="0 1" bash scripts/supp/run_supp_collect.sh 2>&1 | tee supp_collect.log
#   env: GPUS ("0"), TAG (v2_40000_full_s0v2, the screen of the reported runs), OUT (results/<benchmark>/supp),
#        SAMPLES (3), S0_STEP (2500), SKIP (any of: evals validate theory collect), THEORY_ARGS (extra
#        theory_fast.py arguments, e.g. "--replay gt --steps 200"), COLLECT_ARGS, VALIDATE_WORKERS (6), VAE_CKPT, S0_RUN
set -euo pipefail
B=aistpp_genre5
TAG=${TAG:-v2_40000_full_s0v2}
OUT=${OUT:-results/$B/supp}
RES=results/$B/screen/$TAG
SAMPLES=${SAMPLES:-3}
S0_STEP=${S0_STEP:-2500}
SKIP=${SKIP:-}
read -r -a GPU_LIST <<< "${GPUS:-0}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
mkdir -p "$OUT/sanity" "$OUT/logs"
skip() { [[ " $SKIP " == *" $1 "* ]]; }

[ -d "$RES" ] || { echo "[supp] no screen at $RES (run from the repo root, set TAG)"; exit 1; }
grep -q dances_session_majority_correct metrics/style_encoder.py \
  || { echo "[supp] metrics/style_encoder.py is the old version: sync the laptop's copy (validate() per-genre / majority-vote patch)"; exit 1; }
for f in scripts/supp/collect_disco_supp.py scripts/supp/theory_fast.py scripts/evidence/tmr_select.py; do
  [ -f "$f" ] || { echo "[supp] missing $f: sync scripts/supp/ from the laptop"; exit 1; }
done
VAE_CKPT=${VAE_CKPT:-$(ls -d outputs/*_cl_vae_${B}_alltrain | tail -1)/last.ckpt}
if [ -z "${S0_RUN:-}" ]; then
  S0_RUN=$(python -c "import json, os, sys; print(os.path.dirname(json.load(open(sys.argv[1]))['ckpt']))" \
    "$RES/s0_model/after_s0_sess0.json")
fi
export S0_RUN
[ -f "$VAE_CKPT" ] && [ -f "$S0_RUN/last.ckpt" ] || { echo "[supp] missing VAE ($VAE_CKPT) or S0 model ($S0_RUN)"; exit 1; }
S0_PART=$(ls "$S0_RUN"/*step=${S0_STEP}.ckpt 2>/dev/null | head -1 || true)
if [ -z "$S0_PART" ]; then  # the earliest stored step instead
  S0_PART=$(ls "$S0_RUN"/*step=*.ckpt 2>/dev/null | sed -E 's/.*step=([0-9]+)\.ckpt/\1 &/' | sort -n | head -1 | cut -d' ' -f2- || true)
fi
PART_TAG=$(echo "${S0_PART:-none}" | sed -E 's/.*step=([0-9]+)\.ckpt/\1/')
echo "[supp] screen $TAG | S0 model $S0_RUN | partly trained S0 model: ${S0_PART:-none} | VAE $VAE_CKPT | GPUs ${GPU_LIST[*]}"

# ---- 1. sanity scores -------------------------------------------------------------------------------------
JOBS=()
for k in 0 1 2 3 4; do
  JOBS+=("gt_s${k}_t76|$k|--gt --max_tokens 76" "gt_s${k}_full|$k|--gt"
         "vae_s${k}_t76|$k|--vae_recon --max_tokens 76" "vae_s${k}_full|$k|--vae_recon"
         "untrained_s${k}_t76|$k|--untrained --max_tokens 76 --samples $SAMPLES")
done
if [ -n "${S0_PART:-}" ]; then
  for k in 0 4; do JOBS+=("s0step${PART_TAG}_s${k}_t76|$k|--ckpt $S0_PART --max_tokens 76 --samples $SAMPLES"); done
fi
eval_job() {  # <name> <session> <args> <gpu>
  local name=$1 k=$2 args=$3 gpu=$4 f=$OUT/sanity/$1
  [ -f "$f.json" ] && return 0
  mkdir "$f.lock" 2>/dev/null || return 0
  echo "[supp] $name on GPU $gpu ($(date +%H:%M))"
  # shellcheck disable=SC2086
  if CUDA_VISIBLE_DEVICES=$gpu python scripts/eval_cl_session.py --config configs/cl/ldf_${B}_s$k.yaml \
      --vae_ckpt "$VAE_CKPT" --session "$k" $args --out "$f.tmp.json" > "$OUT/logs/$name.log" 2>&1; then
    [ -f "$f.tmp.joints.npz" ] && mv -f "$f.tmp.joints.npz" "$f.joints.npz"
    mv -f "$f.tmp.json" "$f.json"
  else
    echo "[supp] $name FAILED, see $OUT/logs/$name.log"
  fi
  rmdir "$f.lock" 2>/dev/null || true
}
eval_worker() {
  local gpu=$1 job name k args
  for job in "${JOBS[@]}"; do
    IFS='|' read -r name k args <<< "$job"
    eval_job "$name" "$k" "$args" "$gpu"
  done
}
if ! skip evals; then
  for g in "${GPU_LIST[@]}"; do eval_worker "$g" & done
  wait
  # every session's untrained-generator output in one file: validate's "untrained generator" reject probe
  python - "$OUT/sanity" <<'PY' || echo "[supp] could not merge the untrained generator's dances (validate runs without them)"
import sys, numpy as np
from pathlib import Path
d = Path(sys.argv[1]); out = {}
for k in range(5):
    p = d / f"untrained_s{k}_t76.joints.npz"
    if p.exists():
        z = np.load(p)
        for key in sorted(z.files, key=int):
            out[str(len(out))] = z[key]
if out:
    np.savez_compressed(d / "untrained_all.joints.npz", **out)
    print(f"[supp] untrained generator: {len(out)} dances -> {d / 'untrained_all.joints.npz'}")
PY
fi

# ---- 2. validation (CPU) and 3. theory (GPU), in parallel -------------------------------------------------
VAL=raw_data/cl_sessions/_style_encoder/aistpp/validation.json
pids=()
if ! skip validate && [ ! -f "$OUT/validation.json" ]; then
  [ -f "$VAL" ] && [ ! -f "$OUT/validation_prev.json" ] && cp "$VAL" "$OUT/validation_prev.json"
  OOD=""
  [ -f "$OUT/sanity/untrained_all.joints.npz" ] && OOD="--ood_joints $OUT/sanity/untrained_all.joints.npz"
  echo "[supp] style encoder validation on the final pairing (CPU, log $OUT/logs/validate.log)"
  # shellcheck disable=SC2086
  ( VALIDATE_WORKERS=${VALIDATE_WORKERS:-6} python -m metrics.style_encoder validate --dataset aistpp $OOD \
      > "$OUT/logs/validate.log" 2>&1 && cp "$VAL" "$OUT/validation.json" \
      || echo "[supp] validation FAILED, see $OUT/logs/validate.log" ) &
  pids+=($!)
fi
if ! skip theory && [ ! -f "$OUT/theory/.done" ]; then
  echo "[supp] fast theory checks on GPU ${GPU_LIST[0]} (log $OUT/logs/theory.log)"
  # shellcheck disable=SC2086
  ( CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} python scripts/supp/theory_fast.py --tag "$TAG" --out "$OUT/theory" \
      ${THEORY_ARGS:-} > "$OUT/logs/theory.log" 2>&1 && touch "$OUT/theory/.done" \
      || echo "[supp] theory checks FAILED, see $OUT/logs/theory.log" ) &
  pids+=($!)
fi
for p in "${pids[@]}"; do wait "$p" || true; done

# ---- 4. collect ---------------------------------------------------------------------------------------------
if ! skip collect; then
  # shellcheck disable=SC2086
  python scripts/supp/collect_disco_supp.py --tag "$TAG" --out "$OUT" ${COLLECT_ARGS:-} 2>&1 | tee "$OUT/logs/collect.log"
fi

# ---- 5. archive ---------------------------------------------------------------------------------------------
tar czf "results/$B/supp_collect.tar.gz" -C "results/$B" --exclude='*.joints.npz' --exclude='*.lock' "$(basename "$OUT")"
echo "[supp] done ($(date +%H:%M)): $OUT/SUPP_FILL.md, $OUT/theory/THEORY.md -> results/$B/supp_collect.tar.gz"
