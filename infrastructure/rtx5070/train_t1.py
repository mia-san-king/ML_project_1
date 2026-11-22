"""Run the archived original T1 configuration in an independent checkout.

The five original inputs are verified before training. Each invocation requires
a new run directory; training artifacts use a separately reserved unique name.
No feature files are copied and no existing training artifacts are overwritten.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import socket
import subprocess
import sys
import traceback
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path("report/experiments/2026-11-18-geoclip-time/run")
INPUTS = (
    "meta/smapshot.parquet",
    "features/clip_vitl14.npy",
    "features/clip_vitl14_ids.npy",
    "geo/grid_1km.npy",
    "data/splits/main.csv.gz",
)
EXPECTED_ROWS = {"train": 176308, "val": 29529}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_info():
    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *arguments], check=True,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30,
        ).stdout.strip()

    status = git("status", "--porcelain")
    return {"sha": git("rev-parse", "HEAD"), "dirty": bool(status), "status": status}


def _verify_inputs(data_root, manifest):
    entries = manifest["inputs"]
    verified = []
    for relative in INPUTS:
        suffix = PurePosixPath(relative).parts
        matches = [entry for entry in entries
                   if PurePosixPath(entry["path"]).parts[-len(suffix):] == suffix]
        if len(matches) != 1:
            raise ValueError(f"manifest must contain exactly one entry for {relative}")
        expected = matches[0]
        path = (PROJECT_ROOT if relative.startswith("data/") else data_root) / relative
        before = path.stat()
        if not path.is_file() or before.st_size != expected["bytes"]:
            raise ValueError(f"input size mismatch: {path}; expected {expected['bytes']}, got {before.st_size}")
        print(f"校验 SHA256: {relative} ({before.st_size:,} bytes)", flush=True)
        digest = _sha256(path)
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise ValueError(f"input changed while hashing: {path}")
        if digest != expected["sha256"]:
            raise ValueError(f"input SHA256 mismatch: {path}; expected {expected['sha256']}, got {digest}")
        verified.append({"path": str(path.resolve()), "relative": relative,
                         "bytes": before.st_size, "sha256": digest,
                         "original_path": expected["path"]})
    return verified


def _validate_counts(audit):
    for split, expected in EXPECTED_ROWS.items():
        actual = audit.get("coverage", {}).get(split, {}).get("available")
        if actual != expected:
            raise ValueError(f"original T1 requires {split}={expected} available rows; got {actual}")


class _Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        self.console.write(text)
        self.console.flush()
        return len(text)

    def flush(self):
        self.log.flush()
        self.console.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="original feature cache, metadata and Swiss grid directory")
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="new, nonexistent directory for this invocation's audit and console log")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--check-data", action="store_true",
                        help="verify configuration, file hashes and counts; do not load a model or train")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    data_root = args.data_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"无法创建独立运行目录（已有目录不会修改）: {run_dir}: {error}", file=sys.stderr)
        return 1

    run_name = "T1_original_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_") + uuid.uuid4().hex
    split_path = PROJECT_ROOT / "data/splits/main.csv.gz"
    checkpoint = data_root / "ckpt" / f"{run_name}.pt"
    training_log = data_root / "results/metrics" / f"{run_name}_train_log.json"
    state = {
        "status": "preparing", "started_at": _now(), "run_name": run_name,
        "run_dir": str(run_dir), "data_root": str(data_root),
        "split_csv": str(split_path), "check_data_only": args.check_data,
        "device": {"requested": args.device},
        "process": {"pid": os.getpid(), "parent_pid": os.getppid(),
                    "hostname": socket.gethostname(), "platform": platform.platform(),
                    "python": sys.version, "executable": sys.executable,
                    "cwd": str(Path.cwd()), "argv": list(sys.argv if argv is None else argv)},
        "artifacts": {"checkpoint": str(checkpoint), "training_log": str(training_log),
                      "console_log": str(run_dir / "console.log"),
                      "effective_config": str(run_dir / "effective_config.yaml")},
    }

    def save(status):
        state["status"], state["updated_at"] = status, _now()
        _json_write(run_dir / "status.json", state)

    with (run_dir / "console.log").open("x", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            try:
                save("preparing")
                print(f"运行目录: {run_dir}\n唯一实验名: {run_name}", flush=True)
                state["git"] = _repository_info()
                archive = PROJECT_ROOT / ARCHIVE
                config_source = archive / "T1_dual_time_full.yaml"
                manifest_source = archive / "input_manifest.json"
                state["sources"] = {
                    "config": str(config_source), "config_sha256": _sha256(config_source),
                    "manifest": str(manifest_source), "manifest_sha256": _sha256(manifest_source),
                }
                import yaml

                original = yaml.safe_load(config_source.read_text(encoding="utf-8"))
                if original.get("model", {}).get("mode") != "both":
                    raise ValueError("archived configuration must be original dual-time T1 (mode=both)")
                cfg = {**original, "run_name": run_name, "data_root": str(data_root),
                       "split_csv": str(split_path)}
                state["config"] = cfg
                (run_dir / "archive_config.yaml").write_bytes(config_source.read_bytes())
                (run_dir / "effective_config.yaml").write_text(
                    yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
                manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
                _json_write(run_dir / "input_manifest.json", manifest)
                save("verifying")
                verified = _verify_inputs(data_root, manifest)
                _json_write(run_dir / "verified_inputs.json", verified)
                if str(PROJECT_ROOT) not in sys.path:
                    sys.path.insert(0, str(PROJECT_ROOT))
                from models.geoclip_time.lib.temporal_training import check_training_data, run_training

                audit = check_training_data(cfg, data_root=data_root, split_csv=split_path)
                _validate_counts(audit)
                state["audit"] = audit
                _json_write(run_dir / "data_audit.json", audit)
                print("数据核验通过: train=176308, val=29529；归档 T1 参数保持不变。", flush=True)
                if args.check_data:
                    state["finished_at"] = _now()
                    save("checked")
                    print("仅检查完成，没有加载模型或启动训练。", flush=True)
                    return 0

                import torch

                state["device"].update({"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                                         "cuda_available": torch.cuda.is_available()})
                if args.device == "cuda":
                    if not torch.cuda.is_available():
                        raise RuntimeError("--device cuda requested, but CUDA is unavailable in this Python environment")
                    state["device"].update({"name": torch.cuda.get_device_name(0),
                                             "capability": list(torch.cuda.get_device_capability(0))})
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                reservation = checkpoint.with_name(f".{run_name}.reserved.json")
                # Atomic exclusive creation prevents another invocation from ever
                # using this artifact name, even if a UUID collision occurred.
                with reservation.open("x", encoding="utf-8") as stream:
                    json.dump({"run_dir": str(run_dir), "pid": os.getpid(), "created_at": _now()}, stream,
                              ensure_ascii=False)
                state["reservation"] = str(reservation)
                for path in (checkpoint, training_log, checkpoint.with_name(checkpoint.name + ".tmp"),
                             training_log.with_name(training_log.name + ".tmp")):
                    if path.exists():
                        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
                save("training")
                print(f"开始完整 T1 训练: device={args.device}, epochs={cfg['epochs']}\ncheckpoint: {checkpoint}", flush=True)
                result = run_training(cfg, data_root=data_root, split_csv=split_path,
                                      device=args.device, overwrite=False)
                state["result"] = {"checkpoint": str(result["checkpoint"]), "log": str(result["log"]),
                                   "best_val_median_km": float(result["best_val_median_km"])}
                state["finished_at"] = _now()
                save("completed")
                print(f"训练完成: {checkpoint}", flush=True)
                return 0
            except BaseException as error:
                state["exception"] = {"type": type(error).__name__, "message": str(error),
                                      "traceback": traceback.format_exc()}
                state["finished_at"] = _now()
                save("interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
                traceback.print_exc()
                return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
