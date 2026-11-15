#!/bin/bash
# Distillation-replay variants (+KD) of the aistpp_genre5 screen through every session, S1 -> S4:
#   TMR+KD = tmr_joint_neff_kd, GCR+KD = gcr_kd, CSReL+KD = csrel_kd, MIR+KD = mir_kd   (and their training seeds)
# Only what is missing is done: a session that is trained and scored is skipped, so the script can be rerun at any time
# (e.g. after TMR+KD and GCR+KD are already finished, it only trains CSReL+KD and MIR+KD).
#
# A +KD variant is the generic <variant>_kd of scripts/cl/screen_s1.sh / screen_sk.sh: the selection method's buffers,
# replayed by distillation from the variant's own model after the previous session (replay.distill: 0.25 x ground-truth
# loss + 1.0 x distillation loss). Its Session-0 buffer is copied from the base method's screen buffer (csrel -> csrel_kd,
# ...) so +KD replays exactly the same S0 buffer (built with screen_s1.sh's command if the base method has none); its
# seeds <variant>_seed<k> share it. Buffers of later sessions are built by every run with its own model, as for GCR+KD
# and TMR+KD. Each GPU trains one variant chain at a time (S1 with screen_s1.sh, then S2..S4 with screen_sk.sh), taken
# from a shared queue. Run from the repo root:
#
#   GPUS="0 1" bash scripts/cl/run_kd_screens.sh 2>&1 | tee run_kd_screens.log
#   then: GPUS="0 1" bash scripts/cl/test_kd_screens.sh 2>&1 | tee test_kd_screens.log
#
#   env: VARIANTS  +KD variants (default "tmr_joint_neff_kd gcr_kd csrel_kd mir_kd")
#        SEEDS     extra training seeds, e.g. "2 3" -> <variant>_seed2, <variant>_seed3. Default: every seed that existing
#                  +KD runs of this screen were trained with (<x>_kd_seed<k> folders), so new methods match the others.
#                  SEEDS=none: the default seed only.
#        GPUS ("0"), S0_RUN (default: the S0 model the screen scored), S0_TAG (s0v2), PROTOCOL (v2), STEPS (40000),
#        LAST (4: the last session to train), DRY=1 (print the plan and exit)
set -euo pipefail
B=aistpp_genre5
VARIANTS=${VARIANTS:-"tmr_joint_neff_kd gcr_kd csrel_kd mir_kd"}
read -r -a GPU_LIST <<< "${GPUS:-0}"
export SETUP=main PROTOCOL=${PROTOCOL:-v2} STEPS=${STEPS:-40000} S0_TAG=${S0_TAG:-s0v2} SAMPLES=${SAMPLES:-3}
LAST=${LAST:-4}
TAG=${PROTOCOL}_${STEPS}_full_${S0_TAG}  # the folder name screen_s1.sh / screen_sk.sh derive from the same settings
RES=results/$B/screen/$TAG
BUFR=buffers/$B/screen/$TAG
LOG=logs/$B/screen/$TAG
[ -d "$RES" ] || { echo "[kd] no screen at $RES (run from the repo root; PROTOCOL / STEPS / S0_TAG as in the screen)"; exit 1; }
mkdir -p "$LOG"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
if [ -z "${S0_RUN:-}" ] && [ -f "$RES/s0_model/after_s0_sess0.json" ]; then
  S0_RUN=$(python -c "import json, os, sys; print(os.path.dirname(json.load(open(sys.argv[1]))['ckpt']))" \
    "$RES/s0_model/after_s0_sess0.json")
fi
[ -n "${S0_RUN:-}" ] && [ -f "$S0_RUN/last.ckpt" ] || { echo "[kd] set S0_RUN: no S0 model at '${S0_RUN:-}'"; exit 1; }
export S0_RUN

# the +KD variants this script knows: <selection method>_kd, where the selection method alone is a screen variant
for v in $VARIANTS; do
  case ${v%_kd} in
    random|csrel|tmr|ocs|gcr|mir|tmgm|tmr_mix|tmr_joint|tmr_joint_ess|tmr_joint_neff|tmr_joint_op|tmr_resp_neff)
      [ "$v" != "${v%_kd}" ] || { echo "[kd] $v is not a +KD variant (<method>_kd)"; exit 1; } ;;
    *) echo "[kd] unknown +KD variant $v"; exit 1 ;;
  esac
done

# seeds: those the existing runs of these +KD variants used (e.g. gcr_kd_seed2 -> 2), unless given
if [ -z "${SEEDS+x}" ]; then
  SEEDS=$(for v in $VARIANTS; do for d in "$RES/${v}_seed"[0-9]*/; do
            if [ -d "$d" ]; then d=${d%/}; echo "${d##*_seed}"; fi
          done; done | sort -un | tr '\n' ' ')
fi
[ "$SEEDS" = none ] && SEEDS=""
CHAINS=()
for v in $VARIANTS; do
  CHAINS+=("$v")
  for s in $SEEDS; do CHAINS+=("${v}_seed$s"); done
done

scored() {  # <variant> <session t>: the model after session t is scored on every session 0..t
  local i ; for i in $(seq 0 "$2"); do [ -f "$RES/$1/after_s$2_sess$i.json" ] || return 1; done
}
progress() {  # last session whose model is trained and scored (0: none)
  local t=0 k ; for k in $(seq 1 "$LAST"); do scored "$1" "$k" && t=$k || break; done ; echo "$t"
}

echo "[kd] screen $TAG | S0 model $S0_RUN | GPUs ${GPU_LIST[*]} | sessions 1..$LAST | seeds: ${SEEDS:-default only}"
for v in "${CHAINS[@]}"; do echo "[kd]   $v: done up to S$(progress "$v")"; done
[ "${DRY:-0}" = 1 ] && exit 0

# 1. Session-0 buffers of the +KD variants: the base method's screen buffer (the same S0 model, so the same selection)
for v in $VARIANTS; do
  dst=$BUFR/$v/s0.json src=$BUFR/${v%_kd}/s0.json
  [ -f "$dst" ] && continue
  mkdir -p "$BUFR/$v"
  if [ -f "$src" ]; then
    cp "$src" "$dst"
    echo "[kd] $v: S0 buffer copied from ${v%_kd}"
  else
    echo "[kd] $v: building the S0 buffer (${v%_kd}; the base method has no screen buffer)"
    CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} python scripts/build_replay_buffer.py --method "${v%_kd}" \
      --config configs/cl/ldf_${B}_s0.yaml --ckpt "$S0_RUN/last.ckpt" --session 0 --out "$BUFR/$v/.tmp_s0.json" \
      > "$LOG/buffer_s0_$v.log" 2>&1 || { echo "[kd] $v: S0 buffer FAILED, see $LOG/buffer_s0_$v.log"; continue; }
    mv -f "$BUFR/$v/.tmp_s0.json" "$dst"
  fi
done

# 2. variant chains S1 -> S4, one per GPU at a time; a chain is taken by whichever GPU claims it first
chain() {  # <variant> <gpu>
  local v=$1 gpu=$2 k
  if ! scored "$v" 1; then
    echo "[kd] $v: S1 on GPU $gpu ($(date +%H:%M))"
    GPUS=$gpu VARIANTS=$v bash scripts/cl/screen_s1.sh || echo "[kd] $v: screen_s1.sh exited with an error"
  fi
  for k in $(seq 2 "$LAST"); do
    scored "$v" "$k" && continue
    scored "$v" $((k - 1)) || { echo "[kd] $v: stopped before S$k (S$((k - 1)) is not trained and scored)"; return 0; }
    echo "[kd] $v: S$k on GPU $gpu ($(date +%H:%M))"
    SESSION=$k GPUS=$gpu VARIANTS=$v bash scripts/cl/screen_sk.sh || echo "[kd] $v: screen_sk.sh S$k exited with an error"
  done
}
QUEUE=$RES/.kd_queue_$$
mkdir -p "$QUEUE"
trap 'rm -rf "$QUEUE"' EXIT
worker() {
  local gpu=$1 v
  for v in "${CHAINS[@]}"; do
    [ "$(progress "$v")" -ge "$LAST" ] && continue
    mkdir "$QUEUE/$v" 2>/dev/null || continue  # another GPU has it
    chain "$v" "$gpu"
  done
}
for g in "${GPU_LIST[@]}"; do worker "$g" & done
wait

echo "[kd] finished ($(date +%H:%M)):"
for v in "${CHAINS[@]}"; do echo "[kd]   $v: done up to S$(progress "$v")"; done
echo "[kd] next: GPUS=\"${GPU_LIST[*]}\" bash scripts/cl/test_kd_screens.sh  (scores with saved dances + paper tables)"
