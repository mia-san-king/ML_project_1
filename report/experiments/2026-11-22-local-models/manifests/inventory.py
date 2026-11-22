"""Create or verify the SHA-256 inventory of local Swiss geolocation experiment assets."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
NAME = "assets.tsv.gz"
HEADER = ("root", "path", "bytes", "sha256")
# The original checkpoint and corresponding metrics may have a local operator prefix.
LEGACY = "T1_dual_time_available_s0_20261118"
DIRECTORIES = {
    "workspace": ["data/t1_original", "data/local_runs", "data/splits"],
    "store": ["aug8192", "public-assets", "t4_geo4096"],
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(root, kind):
    rows = {}
    for name in DIRECTORIES[kind]:
        for directory, subdirs, files in os.walk(root / name):
            subdirs[:] = [s for s in subdirs if s not in ("__pycache__", "tools") and not s.startswith("transfer_")]
            for filename in files:
                path = Path(directory, filename)
                if filename.endswith("-wal") and path.stat().st_size:
                    raise ValueError(f"Take a consistent SQLite snapshot before indexing {path.name}")
                if filename.startswith(".") or filename.endswith((".lock", "-wal", "-shm", ".pyc", ".py", ".part")):
                    continue
                rel = path.relative_to(root)
                if LEGACY in filename:
                    rel = rel.with_name(filename[filename.index(LEGACY):])
                key = (kind, rel.as_posix())
                if key in rows:
                    raise ValueError(f"Duplicate logical name: {key}")
                rows[key] = path
    return rows


def group(kind, path):
    parts = path.split("/")
    if kind == "workspace":
        return "/".join(parts[:3] if parts[1] == "local_runs" else parts[:2])
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0] + "/(records)"


def digests(paths):
    # Multiple paths can reference the same physical input in different experiments.
    cache = {}
    for (kind, rel), path in sorted(paths.items()):
        st = path.stat()
        key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        if not st.st_ino or key not in cache:
            cache[key] = sha256(path)
        yield kind, rel, st.st_size, cache[key]


def write_inventory(rows):
    size_by_group = defaultdict(lambda: {"paths": 0, "bytes": 0})
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(HEADER)
    for kind, path, size, digest in rows:
        g = size_by_group[kind + "/" + group(kind, path)]
        g["paths"] += 1
        g["bytes"] += size
        writer.writerow((kind, path, size, digest))
    raw = stream.getvalue().encode("utf-8")
    with (HERE / NAME).open("wb") as out, gzip.GzipFile(filename="", fileobj=out, mode="wb", mtime=0) as gz:
        gz.write(raw)
    summary = {"format": "root, path, bytes, sha256; UTF-8 TSV with LF and header",
               "uncompressed_tsv_sha256": hashlib.sha256(raw).hexdigest(),
               "counts_by_input_group": dict(sorted(size_by_group.items()))}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def verify(paths, supplied_roots):
    expected = {}
    with gzip.open(HERE / NAME, "rb") as f:
        raw = f.read()
    summary = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
    if hashlib.sha256(raw).hexdigest() != summary["uncompressed_tsv_sha256"]:
        raise ValueError("TSV differs from the recorded summary")
    rows = csv.reader(io.StringIO(raw.decode("utf-8")), delimiter="\t")
    if tuple(next(rows)) != HEADER:
        raise ValueError("Unexpected manifest header")
    for kind, rel, size, h in rows:
        if kind in supplied_roots:
            expected[kind, rel] = (int(size), h)
    bad = sorted(set(expected) - set(paths))
    for kind, rel, size, h in digests({k: paths[k] for k in expected.keys() & paths.keys()}):
        if (size, h) != expected[kind, rel]:
            bad.append((kind, rel))
    for key in bad:
        print("Missing or mismatched:", "/".join(key))
    print(f"{len(expected) - len(bad)} of {len(expected)} paths verified")
    return int(bool(bad))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for kind in DIRECTORIES:
        parser.add_argument("--" + kind, type=Path)
    parser.add_argument("--write", action="store_true", help="Rebuild using both local roots")
    args = parser.parse_args()
    roots = {k: getattr(args, k) for k in DIRECTORIES if getattr(args, k) is not None}
    if not roots or (args.write and set(roots) != set(DIRECTORIES)):
        parser.error("Supply one or both roots for verification, both for --write")
    for kind, root in roots.items():
        for directory in DIRECTORIES[kind]:
            if not (root / directory).is_dir():
                parser.error(f"Missing {kind} directory: {directory}")
    paths = {k: p for kind, root in roots.items() for k, p in scan(root, kind).items()}
    if args.write:
        write_inventory(digests(paths))
        print(f"Wrote {len(paths)} file hashes")
        return 0
    return verify(paths, roots)


if __name__ == "__main__":
    raise SystemExit(main())
