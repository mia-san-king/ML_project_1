#!/usr/bin/env python3
"""Run isolated T0/T1 retraining, parent-aware T3 retrieval, and full-OLS T4.

Inputs are an independent augmented data root with splits/main.csv.gz. Original
IDs are positive; a training-only variant has ID=-parent_id and inherited GPS/date.
Each model writes under output-root; input metadata, features and geo are never
modified. T0/T1 always start from packaged pretrained GeoCLIP weights, not an old
trained checkpoint. Fifteen augmented epochs use more steps than original epochs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {"T0": "T0_no_time.yaml", "T1": "T1_dual_time.yaml"}


def _paths(data_root, output_root):
    data_root, output_root = (Path(path).expanduser().resolve() for path in (data_root, output_root))
    if output_root == data_root or data_root in output_root.parents:
        raise ValueError("output-root must be outside the augmented input data-root")
    return data_root, output_root


def training_config(model, data_root, output_root):
    """Use the original 15-epoch protocol, changing only paths/run provenance."""
    import yaml

    if model not in CONFIGS:
        raise ValueError("training model must be T0 or T1")
    data_root, output_root = _paths(data_root, output_root)
    cfg = yaml.safe_load((ROOT / "models/geoclip_time/configs" / CONFIGS[model]).read_text())
    cfg.update({"run_name": f"{model}_augmented_s0", "initialization": "pretrained",
                "data_root": str(output_root / model / "data"),
                "split_csv": str(data_root / "splits/main.csv.gz"),
                "allow_partial_features": True, "query_chunk": 64, "gallery_chunk": 2048,
                "augmentation": {"data_root": str(data_root), "variant_id": "-parent_id",
                                 "training": "originals plus one cached variant per augmented parent",
                                 "evaluation": "original positive IDs only"}})
    return cfg


def prepare_model_view(model, data_root, output_root):
    """Link read-only input directories; checkpoints/results stay in this run."""
    cfg = training_config(model, data_root, output_root)
    data_root = Path(data_root).expanduser().resolve()
    view = Path(cfg["data_root"])
    if view.is_symlink():
        raise ValueError("model data view must not be a symlink")
    view.mkdir(parents=True, exist_ok=True)
    for name in ("meta", "features", "geo"):
        target, link = data_root / name, view / name
        if not target.is_dir():
            raise FileNotFoundError(f"missing augmented input directory: {target}")
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise FileExistsError(f"input link points to another dataset: {link}")
        elif link.exists():
            raise FileExistsError(f"refusing to replace existing input path: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
    for name in ("ckpt", "results"):
        if (view / name).is_symlink():
            raise ValueError(f"output directory must not be a symlink: {view / name}")
    return view


def check_inputs(data_root, output_root):
    """Validate data and gallery without loading a model or creating outputs."""
    from models.geoclip_time.lib.temporal_data import load_temporal_data
    from models.geoclip_time.lib.temporal_training import EVAL_SPLITS, _gallery, _require_coverage
    from scripts.augmentation.parent_retrieval import validate_augmented_rows

    data_root, _ = _paths(data_root, output_root)
    cfg = training_config("T1", data_root, output_root)
    data = load_temporal_data(data_root, cfg["split_csv"], time_config=cfg["time"])
    _require_coverage(data, ("train", *EVAL_SPLITS), True)
    augmentation = validate_augmented_rows(data.table)
    grid = _gallery(data_root, cfg)
    return {"data": data.audit, "augmentation": augmentation, "gallery_rows": len(grid),
            "split_csv": cfg["split_csv"], "initialization": "packaged GeoCLIP pretrained weights",
            "training_protocol": {key: cfg[key] for key in ("epochs", "seed", "batch_size", "n_neg", "val_limit", "lr_mlp", "lr_loc")}}


def _save_config(cfg, output_root, model):
    import yaml

    path = Path(output_root).expanduser().resolve() / model / "config.yaml"
    if path.exists() and yaml.safe_load(path.read_text()) != cfg:
        raise FileExistsError(f"existing config describes another run: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def run_action(action, *, data_root, output_root, model=None, device="cpu", threads=8,
               query_chunk=128, encode_batch=2048):
    """Dispatch one explicit stage; never automatically run the other models."""
    import torch

    if action not in ("check", "train", "predict", "t3", "t4"):
        raise ValueError("action must be check, train, predict, t3, or t4")
    if action in ("train", "predict") and model not in CONFIGS:
        raise ValueError("train/predict require --model T0 or --model T1")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    torch.set_num_threads(threads)
    data_root, output_root = _paths(data_root, output_root)
    audit = check_inputs(data_root, output_root)
    print(json.dumps(audit, indent=2), flush=True)
    if action == "check":
        return audit
    split_csv = data_root / "splits/main.csv.gz"
    if action in ("train", "predict"):
        from models.geoclip_time.lib.temporal_training import run_prediction, run_training

        cfg = training_config(model, data_root, output_root)
        view = prepare_model_view(model, data_root, output_root)
        _save_config(cfg, output_root, model)
        if action == "train":
            # run_training always constructs with pretrained=True and seeds 0.
            return run_training(cfg, data_root=view, split_csv=split_csv, device=device)
        checkpoint = view / "ckpt" / f"{cfg['run_name']}.pt"
        return run_prediction(checkpoint, data_root=view, split_csv=split_csv, device=device,
                              allow_partial_features=True)
    if action == "t3":
        from models.geoclip_time.scripts.rerank_references import run_retrieval

        cfg = training_config("T0", data_root, output_root)
        checkpoint = Path(cfg["data_root"]) / "ckpt" / f"{cfg['run_name']}.pt"
        return run_retrieval(data_root=data_root, split_csv=split_csv, checkpoint=checkpoint,
                             output_dir=output_root / "T3", threads=threads, query_chunk=query_chunk,
                             encode_batch=encode_batch, parent_grouping=True, device=device)
    from models.clip_regression.scripts.train import fit_and_predict

    # Existing T4 runs full training-set float64 OLS and all four original splits.
    return fit_and_predict(data_root, split_csv, output_root / "T4")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("check", "train", "predict", "t3", "t4"))
    parser.add_argument("--data-root", type=Path, required=True, help="independent augmented input root")
    parser.add_argument("--output-root", type=Path, required=True, help="separate results root, outside data-root")
    parser.add_argument("--model", choices=tuple(CONFIGS), help="required for train/predict only")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N; T4 always uses CPU OLS")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--query-chunk", type=int, default=128, help="T3 query rows per exhaustive score matrix")
    parser.add_argument("--encode-batch", type=int, default=2048, help="T3 frozen image projection batch size")
    args = parser.parse_args(argv)
    if args.action in ("train", "predict") and args.model is None:
        parser.error("train/predict require --model T0 or --model T1")
    if args.action not in ("train", "predict") and args.model is not None:
        parser.error("--model applies only to train/predict")
    sys.path.insert(0, str(ROOT))
    try:
        result = run_action(**vars(args))
        if args.action in ("train", "predict"):
            print(json.dumps({key: str(value) for key, value in result.items() if isinstance(value, Path)}, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
