"""Read-only SDXL image parity, runtime and Windows GPU memory benchmark."""
import argparse
from contextlib import closing
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
from experiment_paths import VOLUME
import shutil
import sqlite3
from statistics import mean, median
import tempfile
from threading import Event, Thread
from time import perf_counter

import numpy as np
from PIL import Image

import generate_more as gen


def samples(store, count):
    # SQLite can write a shared-memory sidecar even in read-only mode. Copy the
    # DB and optional WAL to a temporary reader so production files stay untouched.
    with tempfile.TemporaryDirectory(prefix="cs433-parity-db-") as tmp:
        source = store / "records.sqlite"
        target = Path(tmp) / source.name
        shutil.copyfile(source, target)
        wal = source.with_name(source.name + "-wal")
        if wal.is_file():
            shutil.copyfile(wal, target.with_name(target.name + "-wal"))
        with closing(sqlite3.connect(target)) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(
                "SELECT * FROM samples WHERE status='accepted' "
                "AND COALESCE(json_extract(parameters_json, '$.text_encoders_released'), 0)=0 "
                "ORDER BY selection_rank DESC LIMIT 1024"
            )]
    if len(rows) < count:
        raise ValueError(f"Need {count} accepted original-mode samples, found {len(rows)}.")
    # Include both prompt branches if both occur in recent production.
    chosen = []
    for grey in (True, False):
        match = next((row for row in rows if bool(row["is_grayscale"]) == grey), None)
        if match:
            chosen.append(match)
    chosen.extend(row for row in rows if row not in chosen)
    return chosen[:count]


class WindowsGPUMemory:
    """Sample WDDM process memory counters from the OS, independently of PyTorch's allocator."""
    def __init__(self):
        self.peak = {}
        self.samples = 0
        self.error = None
        self.stop = Event()
        self.thread = Thread(target=self._poll, daemon=True)

    def _poll(self):
        try:
            import win32pdh as pdh
            query = pdh.OpenQuery()
            try:
                counters = {
                    group + "_" + memory: pdh.AddEnglishCounter(query, f"\\{group}(*)\\{memory}")
                    for group in ("GPU Process Memory", "GPU Adapter Memory")
                    for memory in ("Dedicated Usage", "Shared Usage")
                }
                while True:
                    pdh.CollectQueryData(query)
                    for key, counter in counters.items():
                        values = pdh.GetFormattedCounterArray(counter, pdh.PDH_FMT_LARGE)
                        if key.startswith("GPU Process Memory"):
                            values = {k:v for k,v in values.items() if k.lower().startswith(f"pid_{os.getpid()}_")}
                        if values:
                            size = sum(values.values()) / 2**20
                            self.peak[key] = max(self.peak.get(key, 0), size)
                    self.samples += 1
                    if self.stop.wait(0.5):
                        break
            finally:
                pdh.CloseQuery(query)
        except Exception as error:
            self.error = repr(error)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def replay(row, renderer, params, store):
    import torch
    parent = Path(row["parent_path"])
    expected_png = store / row["image_path"]
    raw = parent.read_bytes()
    expected = expected_png.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["parent_sha256"] or hashlib.sha256(expected).hexdigest() != row["image_sha256"]:
        raise ValueError(f"Missing or changed production source for parent {row['parent_id']}.")
    with Image.open(io.BytesIO(raw)) as photo:
        canvas, geom, grey = gen.recipe.prepare_image(photo, max_side=params["max_side"])
    prompt = gen.variant_prompt(params, grey)
    saved_params = json.loads(row["parameters_json"])
    if grey != bool(row["is_grayscale"]) or prompt != saved_params["prompt"] or geom != saved_params["geometry"]:
        raise ValueError(f"Production preprocessing differs for parent {row['parent_id']}.")
    torch.cuda.synchronize()
    start = perf_counter()
    generated = renderer(canvas, seed=row["seed"], prompt=prompt)
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    candidate = gen.recipe.restore_image(generated, geom, grayscale=grey)
    buffer = io.BytesIO()
    candidate.save(buffer, format="PNG")
    same_png = hashlib.sha256(buffer.getbuffer()).hexdigest() == row["image_sha256"]
    with Image.open(io.BytesIO(expected)) as previous:
        raw_equal = previous.mode == candidate.mode and previous.size == candidate.size and previous.tobytes() == candidate.tobytes()
        change = np.abs(np.asarray(previous.convert("RGB"), dtype=np.int16) -
                        np.asarray(candidate.convert("RGB"), dtype=np.int16))
    stamp = (datetime.fromisoformat(row["updated_at"]) - datetime.fromisoformat(row["created_at"])).total_seconds()
    return {"parent_id": row["parent_id"], "selection_rank": row["selection_rank"], "seed": row["seed"],
            "is_grayscale": grey, "canvas": list(canvas.size), "render_seconds": seconds,
            "original_workflow_seconds_between_record_timestamps": stamp,
            "identical_png_sha256": same_png, "identical_actual_pixels": raw_equal,
            "max_absolute_channel_difference": int(change.max()),
            "mean_absolute_channel_difference": float(change.mean()),
            "fraction_pixels_changed": float(np.any(change, axis=2).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=VOLUME / "aug8192")
    parser.add_argument("--model-dir", type=Path, default=VOLUME / "public-assets/sdxl-fp16")
    parser.add_argument("--mode", choices=("cached", "existing"), default="cached")
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--report", type=Path, help="Optionally write a new JSON file outside the production store.")
    args = parser.parse_args()
    if args.images < 2 or args.warmup < 1:
        parser.error("Use --images >=2 and --warmup >=1.")
    if args.report and (args.report.exists() or args.report.resolve().is_relative_to(args.store.resolve())):
        parser.error("The benchmark report must be a new file outside the production store.")
    import torch
    manifest = gen.load_json(args.store / "generation_manifest.json")
    params = manifest["parameters"]
    if gen.runtime_info() != manifest["runtime"]:
        raise RuntimeError("Select the same Python runtime as the production generation.")
    rows = samples(args.store, args.images)
    args.offload = False
    args.cached_text_only = args.mode == "cached"
    renderer = gen.load_renderer(args, params)
    warm = [replay(rows[i % len(rows)], renderer, params, args.store) for i in range(args.warmup)]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    results = []
    with WindowsGPUMemory() as memory:
        for row in rows:
            result = replay(row, renderer, params, args.store)
            results.append(result)
            print(f"parent {row['parent_id']}  {result['render_seconds']:.3f}s  identical pixels {result['identical_actual_pixels']}", flush=True)
    seconds = [row["render_seconds"] for row in results]
    report = {
        "mode": args.mode, "device": torch.cuda.get_device_name(), "python_runtime": manifest["runtime"],
        "recipe": "seed, source JPEG, SDXL conditioning and image geometry from each persisted original-mode sample",
        "warmup": warm, "timed_samples": results, "all_pixels_identical": all(r["identical_actual_pixels"] for r in warm + results),
        "mean_warmed_render_seconds": mean(seconds), "median_warmed_render_seconds": median(seconds),
        "mean_original_record_wall_seconds": mean(r["original_workflow_seconds_between_record_timestamps"] for r in results),
        "timing_scope": "Timed render includes CUDA synchronization; original DB timestamps include preprocessing, writes and possible JPEG waits.",
        "peak_cuda_allocated_mib_after_warmup": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib_after_warmup": torch.cuda.max_memory_reserved() / 2**20,
        "peak_wddm_mib_after_warmup": memory.peak, "wddm_samples": memory.samples, "wddm_counter_error": memory.error,
    }
    body = json.dumps(report, indent=2, allow_nan=False)
    print(body, flush=True)
    if args.report:
        with args.report.open("x", encoding="utf-8") as f:
            f.write(body + "\n")
    if not report["all_pixels_identical"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
