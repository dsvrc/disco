#!/usr/bin/env bash
# Evidence for the TMR paper on DiscoForcing: checks of the analysis (Thm 1-2, Prop. fw, Lemma modal, Prop. 1 /
# kappa, Eq. 4), mechanisms of forgetting (prior vs audio grounding, blocks, noise levels, dynamics) and behaviour of
# the generated dances, consolidated into $OUT/EVIDENCE.md + evidence_tables.tex. Run from the repo root on the GPU
# machine; re-run after S3 / S4 finish (phi and per-checkpoint losses are cached, finished boundaries are recomputed
# only if their checkpoint files changed).
#
#   R=results/aistpp_genre5/vanilla bash scripts/evidence/run_all.sh
#
# R must hold the scripts/eval_cl_session.py outputs after_s{t}_sess{i}.json + .joints.npz (E5 and the
# cross-analyses use them). Optional knobs: B (benchmark), OUT, EXTRA (extra args for e1-e4, e.g. "--max_windows 200").
set -euo pipefail
B=${B:-aistpp_genre5}
CFG=${CFG:-configs/cl/ldf_${B}_s0.yaml}
OUT=${OUT:-results/evidence/$B}
R=${R:-results/$B/vanilla}
EXTRA=${EXTRA:-}
COMMON="--config $CFG --benchmark $B --out_dir $OUT $EXTRA"

python scripts/evidence/e1_spectral.py   $COMMON
python scripts/evidence/e2_gradients.py  $COMMON
python scripts/evidence/e3_probe_step.py $COMMON
python scripts/evidence/e4_trajectory.py $COMMON
python scripts/evidence/e5_behavior.py   --config "$CFG" --benchmark "$B" --out_dir "$OUT" --results_dir "$R"
python scripts/evidence/consolidate.py   --config "$CFG" --benchmark "$B" --out_dir "$OUT" --results_dir "$R"

# Optional: fill the whole m_{i,t} matrix (every finished session t on every seen session i) so E5 and the
# cross-analyses see all pairs; new runs also store the sequence names E5 uses for the beat-alignment score.
#   for t in 0 1 2 3 4; do D=$(ls -d outputs/*_cl_ldf_${B}_s$t 2>/dev/null | tail -1); [ -n "$D" ] || continue
#     for i in $(seq 0 $t); do [ -f $R/after_s${t}_sess$i.json ] && continue
#       python scripts/eval_cl_session.py --config configs/cl/ldf_${B}_s$t.yaml --vae_ckpt $VAE_CKPT \
#         --ckpt $D/last.ckpt --session $i --samples 3 --out $R/after_s${t}_sess$i.json; done; done
