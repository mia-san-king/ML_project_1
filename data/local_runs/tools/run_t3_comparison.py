"""Run the two local T0 fits and their validation-tuned T3 photo retrievals."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from experiment_paths import VOLUME
from time import perf_counter

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import yaml

from run_original_t1 import ARCHIVE, ROOT, RUNS, SPLIT, read_json, verify

sys.path.insert(0, str(ROOT))
from models.geoclip_time.lib.temporal_training import run_training
from models.geoclip_time.scripts.rerank_references import run_retrieval
from scripts.augmentation.build_dataset import _atomic_json, _sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=("original", "augmented"))
    parser.add_argument("stage", choices=("train", "retrieve"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    aug = args.condition == "augmented"
    outputs = RUNS / "t3_t4_8192" / args.condition
    view = outputs / "T0"
    if aug:
        source = VOLUME / "aug8192/t1_dataset"
        manifest = read_json(source / "build_manifest.json")
        if (manifest["archived_parents"], manifest["new_parents"], manifest["eligible_train"]) != (512, 7680, 176308):
            raise ValueError("Unexpected augmentation dataset")
        hashes = manifest["artifact_sha256"]
        split = source / "splits/main.csv.gz"
        cfg_path = ROOT / "report/experiments/2026-11-20-diffusion-augmentation/models/T0/config.yaml"
    else:
        source = ROOT / "data/t1_original"
        hashes = {"/".join(row["path"].split("/")[-2:]): row["sha256"]
                  for row in read_json(ARCHIVE / "run/input_manifest.json")["inputs"]}
        split = SPLIT
        cfg_path = ARCHIVE / "run/T0_no_time_full.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg.update(run_name=f"T0_t3_{args.condition}_s0", data_root=str(view), split_csv=str(split))
    if aug:
        cfg["augmentation"]["data_root"] = str(source)
    checkpoint = view / "ckpt" / (cfg["run_name"] + ".pt")
    if args.stage == "retrieve":
        result = run_retrieval(data_root=source, split_csv=split, checkpoint=checkpoint,
                               output_dir=outputs / "T3", threads=4, query_chunk=256,
                               encode_batch=2048, parent_grouping=aug, device="cpu")
        print(f"T3 finished, gamma={result['selected_gamma']}, retrieval_seconds={result['elapsed_seconds']:.1f}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA is required")
    view.mkdir(parents=True, exist_ok=True)
    for name, digest in hashes.items():
        if name == "splits/main.csv.gz":
            if _sha256(split) != digest:
                raise ValueError(f"Split digest differs for {args.condition}")
            continue
        dest = view / name
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            try:
                os.link(source / name, dest)
            except OSError:
                shutil.copyfile(source / name, dest)
        verify(dest, {"bytes": (source / name).stat().st_size, "sha256": digest})
    (view / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    torch.cuda.synchronize()
    start = perf_counter()
    fit = run_training(cfg, device="cuda")
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    history = read_json(fit["log"])
    remote_name = "zhang7_T0_no_time_available_s0_20261118_train_log.json"
    expected_val = read_json(ARCHIVE / "metrics" / remote_name)["validation_ids"]
    expected_examples = 184500 if aug else 176308
    if history["validation_ids"] != expected_val or len(history["history"]) != 15:
        raise ValueError("Checkpoint selection protocol differs from the archive")
    if any(epoch["train_examples"] != expected_examples for epoch in history["history"]):
        raise ValueError("Unexpected training set size")
    _atomic_json(view / "training.json", {
        "condition": args.condition, "training_seconds": elapsed, "device": torch.cuda.get_device_name(),
        "torch": str(torch.__version__), "epochs": len(history["history"]), "best_epoch": history["epoch"],
        "examples_per_epoch": expected_examples, "optimizer_steps_per_epoch": history["history"][0]["optimizer_steps"],
        "selection_ids_match_archive": True, "checkpoint_sha256": _sha256(checkpoint),
        "config_source": str(cfg_path), "input_sha256": hashes,
    })
    print(f"{args.condition} T0 completed in {elapsed:.1f}s, best epoch {history['epoch']}", flush=True)


if __name__ == "__main__":
    main()
