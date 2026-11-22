"""Image-only temporal ablation using an isolated source snapshot and run."""
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
RUN = Path(__file__).resolve().parent
SOURCE = RUN / "source"
BASE = RUN.parents[1]
PARENT = BASE / "runs/2026-11-18-geoclip-time"
NAME = "zhang7_T1_image_only_available_s0_20261118"
SPLITS = ("val", "test_in", "test_city", "test_region")


def load(path):
    return json.loads(Path(path).read_text())


def save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def record(stage, **extra):
    status = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), **extra}
    save(RUN / "status.json", status)
    print(json.dumps(status), flush=True)


def verify_source():
    # Paths in this index are relative to RUN: source/... plus both entrypoints.
    seen = set()
    for line in (RUN / "source.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = (RUN / relative).resolve()
        require(path.is_relative_to(RUN), f"source index escapes run directory: {relative}")
        require(checksum(path) == expected, f"source changed: {relative}")
        seen.add(relative)
    require({"run_image_only.py", "launch.sh"}.issubset(seen), "source.sha256 must include both entrypoints")
    require("source/models/geoclip_time/lib/temporal_training.py" in seen, "source snapshot is missing from checksum index")


def verify_inputs(manifest):
    for entry in manifest["inputs"]:
        path = Path(entry["path"])
        before = path.stat()
        require(checksum(path) == entry["sha256"], f"input changed: {path}")
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f"input changed while reading: {path}")


def artifacts(cfg):
    root, name = Path(cfg["data_root"]), cfg["run_name"]
    return [root / "ckpt" / f"{name}.pt", root / "results/metrics" / f"{name}_train_log.json",
            root / "results/preds" / f"{name}.parquet", root / "results/metrics" / f"{name}_summary.json"]


def main():
    import yaml
    verify_source()
    manifest = load(PARENT / "input_manifest.json")
    parent_status = load(PARENT / "pipeline_status.json")
    require(parent_status["state"] == "completed", "parent T0/T1 training and evaluation must complete first")
    require(parent_status["code_commit"] == manifest["code_commit"], "parent provenance differs")
    old_cfg = yaml.safe_load((PARENT / "T1_dual_time_full.yaml").read_text())
    cfg = deepcopy(old_cfg)
    cfg["run_name"], cfg["model"]["mode"] = NAME, "image_only"
    comparable = deepcopy(cfg)
    comparable["run_name"], comparable["model"]["mode"] = old_cfg["run_name"], "both"
    require(comparable == old_cfg, "image-only training settings differ from T1")
    require(cfg["seed"] == 0 and cfg["epochs"] == 15, "expected the same seed 0 and 15 epochs")
    controls = {"T0": parent_status["results"]["T0_no_time"], "T1": parent_status["results"]["T1_dual_time"]}
    control_logs = {label: load(info["training_log"]) for label, info in controls.items()}
    require(control_logs["T1"]["config"] == old_cfg, "T1 saved configuration differs from its YAML")
    selection = control_logs["T1"]["validation_ids"]
    require(selection == control_logs["T0"]["validation_ids"] and len(selection) == 256, "parent validation subsets differ")
    for path in artifacts(cfg):
        require(not path.exists(), f"output already exists; no overwrite/resume: {path}")
    verify_inputs(manifest)
    (RUN / "T1_image_only_full.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    smoke_root = RUN / "smoke_data"
    smoke_root.mkdir(exist_ok=True)
    for name in ("meta", "features", "geo"):
        source, link = PARENT / "smoke_data" / name, smoke_root / name
        require(source.is_dir(), f"missing parent smoke inputs: {source}")
        if not link.exists() and not link.is_symlink():
            link.symlink_to(source, target_is_directory=True)
        require(link.is_symlink() and link.resolve() == source.resolve(), f"unexpected smoke input path: {link}")
    smoke_cfg = deepcopy(cfg)
    smoke_cfg.update(epochs=1, val_limit=8, allow_partial_features=False,
                     run_name="zhang7_T1_image_only_smoke_s0_20261118", data_root=str(smoke_root),
                     split_csv=str(PARENT / "smoke_splits.csv"))
    for path in artifacts(smoke_cfg):
        require(not path.exists(), f"smoke output already exists: {path}")
    (RUN / "T1_image_only_smoke.yaml").write_text(yaml.safe_dump(smoke_cfg, sort_keys=False))

    # This snapshot's existing runner owns the unchanged optimizer groups/rates.
    sys.path.insert(0, str(SOURCE))
    import pandas as pd
    import torch
    from models.geoclip_time.lib.temporal_training import run_prediction, run_training
    require(bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "caller must explicitly select the available GPU")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    record("smoke", gpu=torch.cuda.get_device_name(0), visible_gpu=os.environ["CUDA_VISIBLE_DEVICES"])
    phase = time.monotonic()
    smoke = run_training(smoke_cfg, device="cuda")
    smoke_pred = run_prediction(smoke["checkpoint"], device="cuda", splits=("val",))
    smoke_log = load(smoke["log"])
    require(smoke_log["history"][0]["train_examples"] == 64, "expected 64 smoke training examples")
    require(len(smoke["validation_ids"]) == 8 and len(pd.read_parquet(smoke_pred["predictions"])) == 8, "expected eight smoke validation queries")
    del smoke, smoke_pred
    torch.cuda.empty_cache()
    verify_inputs(manifest)
    record("full_training", smoke_seconds=time.monotonic() - phase)
    phase = time.monotonic()
    trained = run_training(cfg, device="cuda")
    torch.cuda.synchronize()
    require(trained["validation_ids"] == selection, "image-only validation IDs differ from T1")
    checkpoint, training_log = trained["checkpoint"], trained["log"]
    best_epoch = load(training_log)["epoch"]
    training_seconds = time.monotonic() - phase
    del trained
    torch.cuda.empty_cache()
    verify_inputs(manifest)
    record("full_evaluation", selected_epoch=best_epoch, training_seconds=training_seconds)
    phase = time.monotonic()
    output = run_prediction(checkpoint, device="cuda", splits=SPLITS, allow_partial_features=cfg["allow_partial_features"])
    torch.cuda.synchronize()
    evaluation_seconds = time.monotonic() - phase
    verify_inputs(manifest)

    keys = ["id", "split"]
    current = pd.read_parquet(output["predictions"], columns=keys).sort_values(keys).reset_index(drop=True)
    require(current.id.is_unique and set(current.split) == set(SPLITS), "invalid image-only evaluation IDs")
    for label, control in controls.items():
        reference = pd.read_parquet(control["predictions"], columns=keys).sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(current, reference, check_dtype=False)
    summaries = {"image_only": load(output["metrics"]), **{label: load(info["metrics"]) for label, info in controls.items()}}
    rows = []
    for split in SPLITS:
        row = {"split": split, "n": int(current.split.eq(split).sum())}
        for label, summary in summaries.items():
            matches = [value for value in summary["splits"] if value["split"] == split and value["view_type"] == "all"]
            require(len(matches) == 1 and matches[0]["n"] == row["n"], f"metric count mismatch: {label}/{split}")
            row[label] = {key: matches[0][key] for key in ("median_km", "acc@25km")}
        for control in controls:
            row[f"delta_image_only_minus_{control}"] = {key: row["image_only"][key] - row[control][key] for key in ("median_km", "acc@25km")}
        rows.append(row)
    save(RUN / "comparison.json", {"same_eval_ids": True, "same_ordered_validation_ids": True,
                                  "validation_selection_n": 256, "source_index_sha256": checksum(RUN / "source.sha256"),
                                  "parent_code_commit": manifest["code_commit"], "metrics": rows,
                                  "delta_note": "Image-only minus control: negative median km and positive accuracy percentage points favor image-only."})
    for row in rows:
        print(f"{row['split']}: n={row['n']}; image_only={row['image_only']}; T1={row['T1']}; T0={row['T0']}", flush=True)
    record("completed", selected_epoch=best_epoch, checkpoint=str(checkpoint), training_log=str(training_log),
           predictions=str(output["predictions"]), summary=str(output["metrics"]),
           training_seconds=training_seconds, evaluation_seconds=evaluation_seconds,
           comparison=str(RUN / "comparison.json"))


if __name__ == "__main__":
    with (RUN / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (RUN / "status.json").exists(), "run already attempted; no automatic retry/resume")
        try:
            main()
        except Exception as exc:
            traceback.print_exc()
            record("failed", error=repr(exc))
            raise
