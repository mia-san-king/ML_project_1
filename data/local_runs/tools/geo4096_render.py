"""Generate or reuse exact SDXL variants for the frozen sparse-block arm on local CUDA."""
import argparse
from contextlib import closing
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from geo4096_sources import COHORT, EARLIER, PUBLIC, STORE, pin, read_json, selection
import generate_more
import sdxl_recipe as recipe


def status(store, results, rank, started, generated_this_run, waiting=None, complete=False):
    passed = sum(row["status"] == "accepted" for row in results)
    new = sum(row["status"] == "accepted" and row["source"] == "new_sdxl" for row in results)
    stats = {"target":4096, "visited":rank+1, "accepted":passed, "new":new, "reused":passed-new,
             "skipped":len(results)-passed, "remaining":4096-passed, "waiting_for_parent":waiting,
             "new_per_second":generated_this_run/max(.01,time.monotonic()-started), "complete":complete, "updated":recipe._now()}
    recipe._atomic_json(store / "status.json", stats)
    print(json.dumps(stats), flush=True)


def generate(store):
    ids, pin_info = selection()
    protocol = read_json(COHORT / "protocol.json")
    order_hash = hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()
    if order_hash != protocol["sparse_priority_ids_sha256"]:
        raise ValueError("Frozen sparse parent array differs from protocol")
    archived = read_json(generate_more.ARCHIVE / "synthetic_store/generation_manifest.json")
    local_run = read_json(EARLIER / "generation_manifest.json")
    if (any(archived[key] != local_run[key] for key in ("parameters","model_sha256","model_revision","input_sha256"))
            or generate_more.runtime_info() != local_run["runtime"]):
        raise ValueError("SDXL configuration/runtime differs from reference")
    args = SimpleNamespace(model_dir=PUBLIC / "sdxl-fp16", offload=False, cached_text_only=False)
    generate_more.verify_weights(args.model_dir, archived)
    params = archived["parameters"]
    old_db = sqlite3.connect((EARLIER / "records.sqlite").as_uri()+"?mode=ro",uri=True)
    old_db.row_factory = sqlite3.Row
    with closing(old_db):
        previous = {r["parent_id"]:dict(r) for r in old_db.execute("SELECT * FROM samples WHERE status='accepted'")}
    historical = set(-np.asarray(read_json(generate_more.ARCHIVE / "augmented/build_manifest.json")["synthetic_ids"],dtype=np.int64))
    old_ids = np.load(EARLIER / "t1_dataset/features/clip_vitl14_ids.npy", allow_pickle=False)
    old_features = np.load(EARLIER / "t1_dataset/features/clip_vitl14.npy", mmap_mode="r")
    old_positions = {int(-i):k for k,i in enumerate(old_ids) if i < 0}
    if set(old_positions) != set(previous) | historical or len(old_positions) != 8192:
        raise ValueError("Existing feature cache and historical provenance differ")
    sources = read_json(PUBLIC / "public_assets_manifest.json")["files"]
    store.mkdir(parents=True, exist_ok=True)
    config = dict(pin_info, model_sha256=archived["model_sha256"], model_revision=archived["model_revision"],
                  parameters=params, runtime=local_run["runtime"], original_generation_sha256=recipe._sha256(EARLIER/"generation_manifest.json"),
                  sparse_order_sha256=order_hash, target=4096)
    with generate_more.writer_lock(store / ".renderer.lock"), closing(sqlite3.connect(store / "variants.sqlite")) as db:
        pin(store / "generation_config.json", config)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS variants(rank INTEGER PRIMARY KEY, parent INTEGER UNIQUE, record TEXT NOT NULL)")
        db.commit()
        recorded = {r:json.loads(j) for r,j in db.execute("SELECT rank,record FROM variants ORDER BY rank")}
        if list(recorded) != list(range(len(recorded))):
            raise ValueError("Invalid gap in previous sparse-cohort records")
        results, created, passed = [], 0, 0
        started = time.monotonic()
        renderer = None
        outputs = store / "images"
        outputs.mkdir(exist_ok=True)
        with closing(sqlite3.connect(store / "sources.sqlite", timeout=75)) as downloads:
            for rank, item in enumerate(ids):
                pid = int(item)
                row = {"rank":rank, "parent_id":pid, "synthetic_id":-pid, "seed":int(np.random.SeedSequence([params["seed"],pid]).generate_state(1)[0]),
                       "status":"accepted", "source":("cached_8192" if pid in old_positions else "new_sdxl"), "parent_path":None,
                       "parent_sha256":None,"image_path":None,"image_sha256":None,"cached_vector_sha256":None,
                       "quality_json":None,"reason":None}
                if rank in recorded:
                    row = recorded[rank]
                    if row["parent_id"] != pid:
                        raise ValueError(f"Resume parent mismatch at {rank}")
                    if row["image_path"] and recipe._sha256(Path(row["image_path"])) != row["image_sha256"]:
                        raise ValueError(f"Previously generated image changed for {pid}")
                elif pid in old_positions:
                    name = f"images/{pid//1000:04d}/{pid}.jpg"
                    jpg = PUBLIC / name
                    if not jpg.is_file() or recipe._sha256(jpg) != sources[name]["sha256"]:
                        raise ValueError(f"Parent JPEG receipt mismatch for cached variant {pid}")
                    row.update(parent_path=str(jpg), parent_sha256=sources[name]["sha256"],
                               cached_vector_sha256=hashlib.sha256(np.ascontiguousarray(old_features[old_positions[pid]]).tobytes()).hexdigest())
                    if pid in previous:
                        old = previous[pid]
                        path = EARLIER / old["image_path"]
                        if (old["model_sha256"] != config["model_sha256"] or old["seed"] != row["seed"]
                                or old["parent_sha256"] != row["parent_sha256"] or recipe._sha256(path) != old["image_sha256"]
                                or not json.loads(old["quality_json"])["passed"]):
                            raise ValueError(f"Invalid stored image provenance for {pid}")
                        row.update(image_path=str(path),image_sha256=old["image_sha256"],quality_json=old["quality_json"])
                    else:
                        # These 512 images came from the original cluster run. Their complete accepted
                        # feature tail is checksum-verified by augment_train.load_original_inputs.
                        row["quality_json"] = json.dumps({"passed":True, "provenance":"original archive accepted feature; PNG not staged"})
                else:
                    waiting_since = 0.
                    while True:
                        found = downloads.execute("SELECT status,path,sha,reason FROM sources WHERE parent=?", (pid,)).fetchone()
                        if found is not None:
                            break
                        now = time.monotonic()
                        if now-waiting_since >= 55:
                            status(store, results, rank-1, started, created, waiting=pid)
                            waiting_since = now
                        time.sleep(2)
                    available, path_str, parent_sha, reason = found
                    if available == "unavailable":
                        row.update(status="source_unavailable", reason=reason)
                    else:
                        path = Path(path_str)
                        raw = path.read_bytes()
                        if hashlib.sha256(raw).hexdigest() != parent_sha:
                            raise ValueError(f"Parent JPEG changed after download: {pid}")
                        with Image.open(io.BytesIO(raw)) as parent:
                            canvas, geometry, gray = recipe.prepare_image(parent, max_side=params["max_side"])
                        if renderer is None:
                            print(f"Starting local CUDA SDXL at training parent {pid}", flush=True)
                            renderer = generate_more.load_renderer(args, params)
                        prompt = generate_more.variant_prompt(params,gray)
                        generated = renderer(canvas, seed=row["seed"], prompt=prompt)
                        pixels = recipe.restore_image(generated, geometry, grayscale=gray)
                        original = recipe.restore_image(canvas, geometry, grayscale=gray)
                        quality = recipe.quality_check(original, pixels, min_structure=params["min_structure"])
                        dest = outputs / f"{pid}.png"
                        temporary = dest.with_suffix(".part")
                        pixels.save(temporary, format="PNG")
                        image_sha = recipe._sha256(temporary)
                        temporary.replace(dest)
                        row.update(parent_path=path_str, parent_sha256=parent_sha, image_path=str(dest),
                                   image_sha256=image_sha,quality_json=recipe._json(quality),
                                   status="accepted" if quality["passed"] else "rejected",
                                   reason=None if quality["passed"] else ",".join(quality["reasons"]))
                        created += int(quality["passed"])
                if rank not in recorded:
                    db.execute("INSERT INTO variants VALUES(?,?,?)", (rank,pid,recipe._json(row)))
                    db.commit()
                results.append(row)
                passed += int(row["status"] == "accepted")
                if (rank+1) % 100 == 0 or passed == 1 or passed == 4096:
                    status(store,results,rank,started,created)
                if passed == 4096:
                    break
        if passed != 4096:
            raise ValueError("Exhausted frozen parent list before reaching 4096 accepted variants")
        output = pd.DataFrame(results)
        for name, frame in (("attempts",output), ("selected",output.loc[output.status.eq("accepted")])):
            dest = store / f"{name}.parquet"
            tmp = dest.with_suffix(".tmp")
            frame.to_parquet(tmp,index=False)
            tmp.replace(dest)
        status(store,results,rank,started,created,complete=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store",type=Path,default=STORE)
    generate(parser.parse_args().store.resolve())
