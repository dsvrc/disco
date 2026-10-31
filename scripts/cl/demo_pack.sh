#!/bin/bash
# Pack what the forgetting demo needs into demo_pack.zip; the video is rendered elsewhere (a machine with a display):
# scripts/cl/demo_render.py. Packs the dances generated for the session 0 test music that scripts/eval_cl_session.py
# saved next to its scores (<score>.joints.npz): the S0 model right after S0, every screen variant after S3, and
# vanilla after its last session. Adds the S0 test music and the style encoder's probabilities per 5 s window
# (1 s stride), which the video shows as "how much of it is still S0 style". Dances that were not saved (older
# scoring runs) are regenerated with the scored checkpoint on GPU $GPU, into the pack only (results/ is not touched).
#
#   bash scripts/cl/demo_pack.sh                      # -> demo_pack.zip in the repo root
#   env: TAG (screen folder, default v2_40000_full_s0v2), GPU (only for regeneration, default 0)
set -euo pipefail
B=aistpp_genre5
TAG=${TAG:-v2_40000_full_s0v2}
GPU=${GPU:-0}
RES=results/$B/screen/$TAG
STAGE=demo_pack
[ -d "$RES" ] || { echo "[demo] no screen results at $RES (run from the repo root, set TAG)"; exit 1; }
unset PYTHONPATH
export PYTHONNOUSERSITE=1
rm -rf "$STAGE" demo_pack.zip
mkdir -p "$STAGE/dances" "$STAGE/wavs"

jget() { python -c "import json, sys; v = json.load(open(sys.argv[1])).get(sys.argv[2]); print('' if v is None else v)" "$1" "$2"; }

# <variant> <session the model was trained up to>
items=("s0_model 0")
for d in "$RES"/*/; do
  v=$(basename "$d")
  [ "$v" = s0_model ] && continue
  if [ -f "$d/after_s3_sess0.json" ]; then
    items+=("$v 3")
  elif [ "$v" = vanilla ]; then  # no replay: the plain forgetting reference, whatever session it reached
    t=$(ls "$d" | sed -n 's/^after_s\([0-9]\)_sess0\.json$/\1/p' | sort -n | tail -1)
    [ -n "$t" ] && items+=("$v $t")
  fi
done

for it in "${items[@]}"; do
  read -r v t <<< "$it"
  src=$RES/$v/after_s${t}_sess0.json
  dst=$STAGE/dances/$v/after_s${t}_sess0.json
  mkdir -p "$(dirname "$dst")"
  if [ -f "${src%.json}.joints.npz" ] && [ -n "$(jget "$src" names)" ]; then
    cp "$src" "$dst"
    cp "${src%.json}.joints.npz" "${dst%.json}.joints.npz"
    echo "[demo] $v after S$t: StyleRetention $(jget "$src" StyleRetention)"
    continue
  fi
  ck=$(jget "$src" ckpt) ; vae=$(jget "$src" vae_ckpt) ; n=$(jget "$src" samples) ; mt=$(jget "$src" max_tokens)
  if [ ! -f "$ck" ] || [ ! -f "$vae" ]; then
    echo "[demo] $v after S$t: dances not saved and checkpoint '$ck' or VAE '$vae' is missing, skipped"
    rm -rf "$STAGE/dances/$v"
    continue
  fi
  echo "[demo] $v after S$t: dances not saved, regenerating on GPU $GPU ($(date +%H:%M))"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_cl_session.py --config configs/cl/ldf_${B}_s$t.yaml --vae_ckpt "$vae" \
    --ckpt "$ck" --session 0 --samples "${n:-3}" ${mt:+--max_tokens $mt} --out "$dst" > "$STAGE/regen_$v.log" 2>&1 \
    || { echo "[demo] $v: regeneration FAILED, see $STAGE/regen_$v.log"; rm -rf "$STAGE/dances/$v"; }
done

# the S0 test music (one wav per test sequence, aligned with its dance at frame 0)
python - "$STAGE" <<'EOF'
import json, shutil, sys
from pathlib import Path
stage = Path(sys.argv[1])
names = sorted({n for p in stage.glob("dances/*/*.json") for n in json.loads(p.read_text())["names"]})
for n in names:
    src = Path("raw_data/aistpp/wavs") / f"{n}.wav"
    if src.exists():
        shutil.copy(src, stage / "wavs" / src.name)
    else:
        print(f"[demo] no music for {n} at {src}")
print(f"[demo] {len(names)} S0 test music")
EOF

# style probabilities per window, as the scores use them: <dance>.style.npz (prob: windows x classes, owner: dance)
STYLE_ENCODER_DEVICE=${STYLE_ENCODER_DEVICE:-cpu} python - "$STAGE" "$B" "$TAG" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from metrics.style_encoder import StyleEncoder, benchmark_session_of_class, ckpt_path

stage, bench, tag = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
man = json.loads((Path("raw_data/cl_sessions") / bench / "manifest.json").read_text())
enc = StyleEncoder(ckpt_path(man["dataset"], -1))
soc = np.asarray(benchmark_session_of_class(enc.classes, man))
dances = []
for p in sorted(stage.glob("dances/*/*.joints.npz")):
    z = np.load(p)
    joints = [z[str(i)] for i in range(len(z.files))]
    e = enc.encode(joints)
    np.savez_compressed(p.with_name(p.name.replace(".joints.npz", ".style.npz")),
                        prob=e["prob"].astype(np.float32), owner=e["owner"])
    score = json.loads(p.with_name(p.name.replace(".joints.npz", ".json")).read_text())
    ret = float(e["prob"][:, soc == 0].sum(1).mean())
    dances.append({"variant": p.parent.name, "after": int(p.name.split("_")[1][1:]), "file": str(p.relative_to(stage)),
                   "names": score["names"], "samples": len(joints) // len(score["names"]),
                   "StyleRetention": score.get("StyleRetention"), "StyleRetention_recomputed": ret})
    print(f"[demo] {p.parent.name} {p.name}: {len(joints)} dances, StyleRetention {score.get('StyleRetention')} "
          f"(recomputed {ret:.4f})")
info = {"benchmark": bench, "screen": tag, "fps": 30, "classes": list(enc.classes),
        "session_of_class": soc.tolist(), "sessions": [s["groups"] for s in man["sessions"]], "dances": dances}
(stage / "info.json").write_text(json.dumps(info, indent=1))
EOF

python -m zipfile -c demo_pack.zip "$STAGE"
echo "[demo] wrote demo_pack.zip ($(du -h demo_pack.zip | cut -f1)), from $STAGE/"
