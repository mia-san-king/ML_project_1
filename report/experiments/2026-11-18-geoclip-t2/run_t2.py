"""Run the direct-fusion comparison in a detached DHLAB experiment directory."""
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml

run = Path(__file__).resolve().parent
repo = run / "source"
base = run.parents[1]
parent = base / "runs/2026-11-18-geoclip-time"
sys.path.insert(0, str(repo))
from models.geoclip_time.lib.temporal_training import run_prediction, run_training

def load(path):
    return json.loads(path.read_text())

def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)

def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def record(stage, **extra):
    info = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), **extra}
    save(run / "status.json", info)
    print(json.dumps(info), flush=True)

def main():
    old_cfg = yaml.safe_load((parent / "T1_dual_time_full.yaml").read_text())
    cfg = yaml.safe_load((repo / "models/geoclip_time/configs/T2_direct_time.yaml").read_text())
    for key in ("query_chunk", "gallery_chunk", "allow_partial_features", "data_root", "split_csv"):
        cfg[key] = old_cfg[key]
    cfg["run_name"] = "zhang7_T2_direct_pretrained_s0_20261118"
    comparable = dict(cfg, run_name=old_cfg["run_name"], model=dict(cfg["model"], mode="both"))
    assert comparable == old_cfg, "T2 training settings differ from T1"
    (run / "T2_full.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    for item in load(parent / "input_manifest.json")["inputs"]:
        assert checksum(Path(item["path"])) == item["sha256"], f"Input changed: {item['path']}"
    old_status = load(parent / "pipeline_status.json")
    assert old_status["state"] == "completed"
    control = old_status["results"]["T1_dual_time"]
    reference_ids = load(Path(control["training_log"]))["validation_ids"]
    data = Path(cfg["data_root"])
    for path in (data/"ckpt"/(cfg["run_name"]+".pt"), data/"results/metrics"/(cfg["run_name"]+"_train_log.json"),
                 data/"results/metrics"/(cfg["run_name"]+"_summary.json"), data/"results/preds"/(cfg["run_name"]+".parquet")):
        if path.exists():
            raise FileExistsError(path)

    smoke_root = run / "smoke_data"
    smoke_root.mkdir(exist_ok=True)
    for sub in ("meta","features","geo"):
        link = smoke_root / sub
        if not link.exists():
            link.symlink_to(parent / "smoke_data" / sub, target_is_directory=True)
    smoke_cfg = dict(cfg, epochs=1, val_limit=8, run_name="T2_direct_smoke", data_root=str(smoke_root),
                     split_csv=str(parent / "smoke_splits.csv"))
    record("smoke", gpu=torch.cuda.get_device_name(0), gpu_id=os.environ["CUDA_VISIBLE_DEVICES"])
    start = time.monotonic()
    smoke = run_training(smoke_cfg, device="cuda")
    pred = run_prediction(smoke["checkpoint"], device="cuda", splits=("val",))
    assert len(pd.read_parquet(pred["predictions"])) == 8 and len(smoke["validation_ids"]) == 8
    del smoke, pred
    torch.cuda.empty_cache()
    record("full_training", smoke_seconds=round(time.monotonic()-start, 1))
    start = time.monotonic()
    trained = run_training(cfg, device="cuda")
    assert trained["validation_ids"] == reference_ids
    checkpoint = trained["checkpoint"]
    epoch = load(trained["log"])["epoch"]
    del trained
    torch.cuda.empty_cache()
    record("full_evaluation", selected_epoch=epoch, training_minutes=(time.monotonic()-start)/60)
    output = run_prediction(checkpoint, device="cuda", allow_partial_features=True)
    keys = ["id", "split"]
    t2 = pd.read_parquet(output["predictions"], columns=keys).sort_values(keys).reset_index(drop=True)
    t1 = pd.read_parquet(control["predictions"], columns=keys).sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(t1, t2, check_dtype=False)
    assert len(t2) == 93538 and t2.id.is_unique
    rows = []
    for label, metrics in (("T1", control["metrics"]), ("T2", output["metrics"])):
        for row in load(Path(metrics))["splits"]:
            rows.append(dict(model=label, **row))
    save(run / "comparison.json", {"same_eval_ids": True, "same_validation_subset": True, "metrics": rows})
    record("completed", selected_epoch=epoch, checkpoint=str(checkpoint), predictions=str(output["predictions"]),
           summary=str(output["metrics"]))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        record("failed", error=repr(exc))
        raise
