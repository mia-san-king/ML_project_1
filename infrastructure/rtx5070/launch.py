"""Fetch a tested GitHub revision and start a detached T1 run on Windows."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile


REPOSITORY = "https://github.com/mia-san-king/ML_project_1.git"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit", help="full Git commit SHA tested on the Mac")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path.home() / "cs433-gpu-runs")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("supply a full lowercase Git commit SHA")
    if os.name != "nt":
        parser.error("run the launcher with the project's Python on the Windows GPU host")
    data = args.data_root.resolve(strict=True)
    if not data.is_dir():
        parser.error("--data-root must be a directory")

    runs = args.runs_root.resolve()
    runs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job = Path(tempfile.mkdtemp(prefix=f"{stamp}-{args.commit[:10]}-", dir=runs))
    print(f"Run folder: {job}", flush=True)
    source = job / "source"
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    git = ["git", "-C", str(source)]
    subprocess.run([*git, "fetch", "--quiet", "--depth=1", "--no-tags", REPOSITORY,
                    args.commit], check=True)
    subprocess.run([*git, "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if actual != args.commit:
        raise RuntimeError(f"Expected {args.commit}, checked out {actual}")

    entry = source / "infrastructure/rtx5070/train_t1.py"
    if not entry.is_file():
        parser.error(f"The selected revision does not contain {entry.name}")
    command = [sys.executable, "-u", "-X", "utf8", str(entry), "--data-root", str(data),
               "--run-dir", str(job / "run"), "--device", "cuda"]
    with (job / "console.log").open("xb") as output:
        process = subprocess.Popen(
            command, cwd=source, stdin=subprocess.DEVNULL, stdout=output,
            stderr=subprocess.STDOUT, close_fds=True,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    print(f"Started PID {process.pid} at {actual}", flush=True)
    print(f"Console: {job / 'console.log'}")


if __name__ == "__main__":
    main()
