"""Reproduce the archived T1 CUDA run on the original feature cache."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from time import perf_counter


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "data/local_runs"
ARCHIVE = ROOT / "report/experiments/2026-11-18-geoclip-time"
CONFIG = ARCHIVE / "run/T1_dual_time_full.yaml"
SPLIT = ROOT / "data/splits/main.csv.gz"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(path, entry):
    if path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Size differs from archived manifest: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"SHA-256 differs from archived manifest: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/t1_original")
    parser.add_argument("--verify-only", action="store_true", help="Verify all five input hashes and exit.")
    args = parser.parse_args()

    inputs = []
    manifest = read_json(ARCHIVE / "run/input_manifest.json")
    for entry in manifest["inputs"]:
        parts = PurePosixPath(entry["path"]).parts
        relative = Path(*parts[-2:])
        path = SPLIT if relative == Path("splits/main.csv.gz") else args.data_root / relative
        verify(path, entry)
        if path != SPLIT:
            inputs.append((path, relative, entry))
    print("All four data files and the fixed split match the remote SHA-256 manifest.", flush=True)
    if args.verify_only:
        return

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import yaml
    sys.path.insert(0, str(ROOT))
    from models.geoclip_time.lib.temporal_training import run_prediction, run_training

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment.")
    torch.set_num_threads(4)
    run = RUNS / datetime.now(timezone.utc).strftime("t1_full_%Y%m%dT%H%M%S_%fZ")
    run.mkdir(exist_ok=False)
    print(f"Run directory: {run}", flush=True)
    for source, relative, entry in inputs:
        dest = run / relative
        dest.parent.mkdir(exist_ok=True)
        try:
            os.link(source, dest)
        except OSError:
            shutil.copyfile(source, dest)
            verify(dest, entry)

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    name = cfg["run_name"]
    cfg.update(data_root=str(run), split_csv=str(SPLIT))
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    fit = run_training(cfg, device="cuda")
    torch.cuda.synchronize()
    train_seconds = perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 2**20
    checkpoint, training_log = fit["checkpoint"], fit["log"]
    del fit
    torch.cuda.empty_cache()
    print(f"Training complete in {train_seconds:.2f} s. Starting full evaluation.", flush=True)
    start = perf_counter()
    predictions = run_prediction(checkpoint, device="cuda", allow_partial_features=cfg["allow_partial_features"])
    torch.cuda.synchronize()
    eval_seconds = perf_counter() - start
    train = read_json(training_log)
    remote_train = read_json(ARCHIVE / "metrics" / f"{name}_train_log.json")
    reference = read_json(ARCHIVE / "run/pipeline_status.json")["results"]["T1_dual_time"]
    remote_metrics = read_json(ARCHIVE / "metrics" / f"{name}_summary.json")
    local_metrics = {row["split"]: row for row in read_json(predictions["metrics"])["splits"]}
    splits = {}
    for remote in remote_metrics["splits"]:
        split = remote["split"]
        local = local_metrics[split]
        splits[split] = {
            "n": local["n"], "remote_n": remote["n"],
            "median_km": local["median_km"], "remote_median_km": remote["median_km"],
            "delta_median_km": local["median_km"] - remote["median_km"],
            "acc25_percent": local["acc@25km"], "remote_acc25_percent": remote["acc@25km"],
            "delta_acc25_pp": local["acc@25km"] - remote["acc@25km"],
        }
    summary = {
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "remote_code_commit": manifest["code_commit"], "all_inputs_sha256_match": True,
        "train_examples_per_epoch": train["history"][0]["train_examples"], "epochs": len(train["history"]),
        "selected_validation_ids_match": train["validation_ids"] == remote_train["validation_ids"],
        "best_epoch": train["epoch"], "remote_best_epoch": remote_train["epoch"],
        "training_seconds": train_seconds, "remote_training_seconds": reference["training_seconds"],
        "evaluation_seconds": eval_seconds, "remote_evaluation_seconds": reference["evaluation_seconds"],
        "training_peak_mib": peak, "remote_training_peak_mib": reference["training_peak_mib"],
        "splits": splits,
    }
    text = json.dumps(summary, indent=2)
    (run / "comparison.json").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
