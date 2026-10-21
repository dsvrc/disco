import argparse
import os
import subprocess
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf


def _infer_dataset_tag(ds_conf) -> str:
    """Best-effort: map a dataset config entry to a stable tag used in test_setting.*."""
    target = str(ds_conf.get("target", "")).lower()
    if "finedance" in target:
        return "finedance"
    if "aistpp" in target:
        return "aistpp"
    if "babel" in target:
        return "babel"

    # fallback: inspect meta paths
    for k in ("train_meta_paths", "val_meta_paths", "test_meta_paths"):
        paths = ds_conf.get(k, None)
        if not paths:
            continue
        # paths can be list-like; convert to str
        p0 = str(paths[0]).lower()
        if "/finedance/" in p0:
            return "finedance"
        if "/aistpp/" in p0:
            return "aistpp"
        if "/babel" in p0:
            return "babel"
    return "unknown"


def _find_sanity_multi_yaml(run_dir: Path) -> Path:
    """
    Find a suitable base config under <run_dir>/sanity_check.

    Preference order:
    1) any *.yaml / *.yml whose filename contains "multi" (case-insensitive), pick newest mtime
    2) if no "multi" matches, but exactly one yaml exists, use it
    3) otherwise raise with a helpful message listing candidates
    """
    sanity_dir = run_dir / "sanity_check"
    if not sanity_dir.exists():
        raise FileNotFoundError(f"sanity_check folder not found: {sanity_dir}")

    yamls = sorted(list(sanity_dir.glob("*.yml")) + list(sanity_dir.glob("*.yaml")))
    if len(yamls) == 0:
        raise FileNotFoundError(f"No yaml configs found under: {sanity_dir}")

    multi = [p for p in yamls if "multi" in p.name.lower()]
    if len(multi) > 0:
        # pick newest to better match the run that produced the folder
        return max(multi, key=lambda p: p.stat().st_mtime)

    if len(yamls) == 1:
        return yamls[0]

    candidates = "\n".join([f"  - {p.name}" for p in yamls])
    raise FileNotFoundError(
        "Could not infer base config: no '*multi*' yaml found and multiple yaml files exist.\n"
        f"Folder: {sanity_dir}\n"
        f"Candidates:\n{candidates}\n"
        "Pass --config explicitly to disambiguate."
    )


def _map_test_to_val_paths(cfg):
    """
    Make validation split load the test split, so that metrics computed in validation loop
    correspond to the test set.
    """
    if "data" not in cfg:
        return cfg

    # Multi-dataset config: cfg.data.datasets = [ ... ]
    if "datasets" in cfg.data:
        for ds_conf in cfg.data.datasets:
            if "test_meta_paths" in ds_conf and ds_conf.get("test_meta_paths"):
                ds_conf["val_meta_paths"] = ds_conf["test_meta_paths"]
        return cfg

    # Single-dataset config: cfg.data.{val_meta_paths,test_meta_paths}
    if "test_meta_paths" in cfg.data and cfg.data.get("test_meta_paths"):
        cfg.data.val_meta_paths = cfg.data.test_meta_paths
    return cfg


def _disable_test_paths(cfg):
    """
    Best-effort: make the "test" dataloader empty.
    This matters because train_ldf.py calls validate() with [val_dataloader, test_dataloader].
    """
    if "data" not in cfg:
        return cfg

    # Multi-dataset
    if "datasets" in cfg.data:
        for ds_conf in cfg.data.datasets:
            # empty list keeps type stable and avoids None-handling surprises in datasets
            ds_conf["test_meta_paths"] = []
        return cfg

    # Single-dataset
    cfg.data.test_meta_paths = []
    return cfg


def make_eval_cfg(
    base_cfg,
    dataset_tag: str,
    ckpt_path: Path,
    out_root: Path,
    device_ids=None,
    eval_split: str = "val",  # "val" or "test"
):
    # NOTE: do NOT resolve interpolations yet. Some older configs don't define `dirs.*`
    # (e.g. `${dirs.raw_data}`), which would crash here with InterpolationKeyError.
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))

    # Backward-compatible defaults for common interpolation roots.
    # These match the typical repo layout: raw_data/, deps/, outputs/.
    if "dirs" not in cfg or cfg.dirs is None:
        cfg.dirs = {}
    for k, default in {"raw_data": "raw_data", "deps": "deps", "outputs": "outputs"}.items():
        if k not in cfg.dirs or cfg.dirs.get(k) in (None, ""):
            cfg.dirs[k] = default

    # Some configs interpolate from `wandb_info.*` even when wandb is disabled.
    if "wandb_info" not in cfg or cfg.wandb_info is None:
        cfg.wandb_info = {}
    # Best-effort defaults: prefer logger.wandb values if present.
    logger_project = None
    logger_entity = None
    try:
        if "logger" in cfg and "wandb" in cfg.logger:
            logger_project = cfg.logger.wandb.get("project", None)
            logger_entity = cfg.logger.wandb.get("entity", None)
    except Exception:
        pass
    if "project" not in cfg.wandb_info or cfg.wandb_info.get("project") in (None, ""):
        cfg.wandb_info.project = logger_project or "m2d"
    if "key" not in cfg.wandb_info or cfg.wandb_info.get("key") in (None, ""):
        cfg.wandb_info.key = ""
    if "entity" not in cfg.wandb_info:
        cfg.wandb_info.entity = logger_entity

    # eval settings
    cfg.train = False
    cfg.resume_ckpt = None
    cfg.test_ckpt = str(ckpt_path)

    # Avoid wandb + rendering (faster, no side-effects)
    if "logger" in cfg and "wandb" in cfg.logger:
        cfg.logger.wandb.wandb_key = ""
    if "test_setting" in cfg:
        cfg.test_setting.render = False

    # single GPU by default (avoid DDP + duplicated work)
    if device_ids is None:
        device_ids = [0]
    cfg.trainer.devices = device_ids

    # make output directory deterministic
    cfg.save_dir = str(out_root)
    cfg.exp_name = f"eval_{dataset_tag}_{ckpt_path.stem}"

    # enable only the selected dataset for val/test; keep train irrelevant here
    if "data" in cfg and "datasets" in cfg.data:
        for ds_conf in cfg.data.datasets:
            tag = _infer_dataset_tag(ds_conf)
            if eval_split == "test":
                # We'll run "test metrics" via the validation loop by mapping test->val paths
                ds_conf["enable_val"] = (tag == dataset_tag)
                ds_conf["enable_test"] = False
            else:
                ds_conf["enable_val"] = (tag == dataset_tag)
                # Even when evaluating val split, train_ldf.py will also build a test dataloader.
                # We disable it here so the second dataloader becomes empty.
                ds_conf["enable_test"] = False

    if eval_split == "test":
        cfg = _map_test_to_val_paths(cfg)
        cfg = _disable_test_paths(cfg)
    else:
        # Make test split empty (best-effort) to avoid doing extra work when we only want val.
        cfg = _disable_test_paths(cfg)

    OmegaConf.resolve(cfg)
    return cfg


def run_one(cfg_path: Path, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    train_py = repo_root / "train_ldf.py"
    cmd = [str(train_py), "--config", str(cfg_path)]
    env = os.environ.copy()
    # Use the same interpreter that's running this script (typically the correct conda env).
    py = env.get("PYTHON", None) or env.get("PYTHON_EXECUTABLE", None) or "python"
    # If user runs this script with a specific interpreter, prefer that.
    try:
        import sys
        py = sys.executable
    except Exception:
        pass
    with open(log_path, "w") as f:
        p = subprocess.Popen(
            [py] + cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(repo_root),
        )
        return p.wait()


def _resolve_ckpt_list(ckpt_arg: str, run_dir: Optional[Path]) -> List[Path]:
    """
    Support:
    - absolute path to a ckpt
    - filename inside run_dir (e.g. last.ckpt)
    - comma-separated list
    - glob pattern (e.g. step_step=*.ckpt)
    """
    raw = ckpt_arg.strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: List[Path] = []

    for p in parts:
        pp = Path(p)
        if any(ch in p for ch in ["*", "?", "["]):
            if run_dir is None:
                raise ValueError("Glob ckpt pattern requires --run_dir")
            matches = sorted(run_dir.glob(p))
            out.extend([m.resolve() for m in matches])
            continue

        if not pp.is_absolute():
            if run_dir is None:
                raise ValueError("Relative --ckpt requires --run_dir")
            pp = run_dir / pp
        out.append(pp.resolve())

    # de-dup, keep order
    seen = set()
    uniq = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a config yaml. If omitted, auto-detects a '*multi*' yaml under "
            "<run_dir>/sanity_check/ (or the only yaml if exactly one exists)."
        ),
    )
    ap.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help=(
            "outputs/<run> folder (contains sanity_check/*.yaml). Used when --config is omitted "
            "or when --ckpt is relative."
        ),
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default="last.ckpt",
        help=(
            "Checkpoint path(s). Supports absolute path, filename inside --run_dir, "
            "comma-separated list, or glob pattern (e.g. 'step_step=*.ckpt')."
        ),
    )
    ap.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset tags to evaluate (e.g. finedance,aistpp). Empty = auto-detect from config.",
    )
    ap.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU index inside the process (default: 0). For a specific physical GPU, set CUDA_VISIBLE_DEVICES externally.",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to evaluate. 'test' runs metrics on test set (implemented by mapping test->val paths).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for logs/configs. Default: <run_dir>/eval_metrics or ./outputs/eval_metrics when run_dir is absent.",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve() if args.run_dir else None

    if args.config:
        base_cfg_path = Path(args.config).resolve()
    else:
        if run_dir is None:
            raise ValueError("Provide either --config or --run_dir")
        base_cfg_path = _find_sanity_multi_yaml(run_dir)
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg_path}")

    ckpt_paths = _resolve_ckpt_list(args.ckpt, run_dir)
    if len(ckpt_paths) == 0:
        raise FileNotFoundError(f"No checkpoints matched: {args.ckpt}")
    for p in ckpt_paths:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    base_cfg = OmegaConf.load(str(base_cfg_path))

    if args.out_dir:
        eval_root = Path(args.out_dir).resolve()
    else:
        eval_root = (run_dir / "eval_metrics") if run_dir is not None else (Path("outputs") / "eval_metrics")
    cfg_root = eval_root / "configs"
    log_root = eval_root / "logs"
    cfg_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    if args.datasets.strip():
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        # auto-detect from config
        found = []
        if "data" in base_cfg and "datasets" in base_cfg.data:
            for ds_conf in base_cfg.data.datasets:
                tag = _infer_dataset_tag(ds_conf)
                if tag in ("finedance", "aistpp", "babel") and tag not in found:
                    found.append(tag)
        datasets = found or ["finedance", "aistpp"]

    # Only run validation metrics; avoid test-step rendering/saving in validate().
    # - For split=test we map test->val paths and disable test dataloader entirely.
    # - For split=val we still want to avoid test_step during validate.
    def _apply_fast_eval_flags(cfg):
        try:
            cfg.validation.test_steps = int(10**18)
        except Exception:
            pass
        try:
            cfg.val_repeat = 1
        except Exception:
            pass
        return cfg

    for ckpt_path in ckpt_paths:
        for ds in datasets:
            out_root = eval_root / "outputs" / ds / ckpt_path.stem
            cfg = make_eval_cfg(
                base_cfg,
                ds,
                ckpt_path,
                out_root,
                device_ids=[args.device],
                eval_split=args.split,
            )
            cfg = _apply_fast_eval_flags(cfg)

            cfg_path = cfg_root / f"{ds}_{ckpt_path.stem}_{args.split}.yaml"
            log_path = log_root / f"{ds}_{ckpt_path.stem}_{args.split}.log"
            OmegaConf.save(cfg, str(cfg_path))

            rc = run_one(cfg_path, log_path)
            if rc != 0:
                raise RuntimeError(f"Eval failed for {ds} @ {ckpt_path.name}. See log: {log_path}")

    print(f"Done. Logs in: {log_root}")


if __name__ == "__main__":
    main()


# python scripts/eval_ckpt_metrics_per_dataset.py \
#   --config configs/ldf_multi.yaml \
#   --run_dir outputs/20260125_075530_ldf_multi \
#   --ckpt step_step=20000.ckpt \
#   --device 0

# python scripts/eval_ckpt_metrics_per_dataset.py \
#   --run_dir outputs/20260112_064127_ldf \
#   --ckpt "step_step=240000.ckpt" \
#   --config configs/aistpp/ldf_aistpp_272d_stream_pae.yaml \
#   --datasets aistpp \
#   --split val \
#   --device 1

# python scripts/eval_ckpt_metrics_per_dataset.py \
#   --run_dir outputs/20260125_213807_ldf_multi \
#   --ckpt "step_step=140000.ckpt,step_step=150000.ckpt,step_step=160000.ckpt" \
#   --datasets aistpp \
#   --split val \
#   --device 7