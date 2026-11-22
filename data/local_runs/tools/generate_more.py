"""Extend the historical SDXL sample on Windows, preserving its original 512 parents."""
import argparse
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import io
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
from experiment_paths import VOLUME
import sqlite3
from time import perf_counter, sleep
from types import SimpleNamespace

import numpy as np
from PIL import Image

import sdxl_recipe as recipe


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "report/experiments/2026-11-20-diffusion-augmentation"
ASSETS = VOLUME / "public-assets"
ARCHIVED_COUNT = 512
TARGET = 7680


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def runtime_info():
    result = {}
    for key in ("diffusers", "transformers", "torch", "accelerate", "numpy", "Pillow"):
        try:
            result[key] = version(key)
        except PackageNotFoundError:
            result[key] = None
    return result


def selection(args, archived):
    recorded = load_json(ARCHIVE / "synthetic_store/generation_summary.json")
    parents, eligible, hashes = recipe._selection(SimpleNamespace(
        data_root=args.data_root, split_csv=args.split_csv, feature_ids=None,
        seed=archived["parameters"]["seed"], count=recorded["eligible_train"],
    ))
    ids = parents.index.to_numpy(dtype="<i8")
    prefix = hashlib.sha256(ids[:ARCHIVED_COUNT].tobytes()).hexdigest()
    existing = load_json(ARCHIVE / "augmented/build_manifest.json")["synthetic_ids"]
    if (hashes != archived["input_sha256"] or len(ids) != eligible or
            prefix != recorded["selection_ids_sha256"] or not np.array_equal(-np.sort(ids[:ARCHIVED_COUNT]), existing)):
        raise ValueError("Original inputs or seeded selection differ from the historical experiment.")
    return parents, hashes, hashlib.sha256(ids.tobytes()).hexdigest()


def verify_weights(directory, archived):
    files = archived["model_files"]
    for item in files:
        path = directory / item["path"]
        if path.stat().st_size != item["bytes"] or recipe._sha256(path) != item["sha256"]:
            raise ValueError(f"SDXL weight or config differs from archived manifest: {path}")
    combined = hashlib.sha256(recipe._json(files).encode()).hexdigest()
    if combined != archived["model_sha256"]:
        raise ValueError("Unrecognized aggregate SDXL weight digest.")


@contextmanager
def writer_lock(path):
    with path.open("a+b") as file:
        if file.tell() == 0:
            file.write(b"0")
            file.flush()
        file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another generator is using this store.") from error
        yield


def allow_unavailable_status(conn):
    """Extend the archived CHECK while preserving existing local pilot rows."""
    sql = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='samples'").fetchone()[0]
    if "'source_unavailable'" in sql:
        return
    old = "'pending','accepted','rejected','error'"
    if old not in sql:
        raise ValueError("Unrecognized samples status constraint.")
    temporary = "samples_v2"
    new = sql.replace("CREATE TABLE samples", f"CREATE TABLE {temporary}", 1).replace(old, old + ",'source_unavailable'", 1)
    if new == sql or f"CREATE TABLE {temporary}" not in new:
        raise ValueError("Cannot migrate samples table.")
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(new)
        conn.execute("INSERT INTO samples_v2 SELECT * FROM samples")
        conn.execute("DROP TABLE samples")
        conn.execute("ALTER TABLE samples_v2 RENAME TO samples")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def downloading(root):
    lock_path = root / "transfer.lock"
    if not lock_path.is_file():
        return False
    with lock_path.open("r+b") as f:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
        except (BlockingIOError, PermissionError):
            return True
        return False


class PublicPhotoSource:
    def __init__(self, images_root, parents):
        self.folder = images_root.parent
        self.path = self.folder / "public_asset_status.json"
        self.parents = parents.index.to_numpy(dtype="<i8")
        self.last_count = None
        self.status = None
        self.last_modified = None

    def read(self):
        modified = self.path.stat().st_mtime_ns
        if modified != self.last_modified:
            status = load_json(self.path)
            count = status["photo_count"]
            if not ARCHIVED_COUNT <= count <= len(self.parents):
                raise ValueError(f"Invalid downloader selection count: {count}")
            if count != self.last_count:
                self.expected_digest = hashlib.sha256(self.parents[:count].tobytes()).hexdigest()
                self.last_count = count
            if status["selection_sha256"] != self.expected_digest:
                raise ValueError("Public download prefix differs from the generation seed-0 selection.")
            self.status, self.last_modified = status, modified
        return self.status

    def await_source(self, path, rank, remaining):
        name = path.relative_to(self.folder).as_posix()
        last_log = None
        while True:
            status = self.read()
            unavailable = status["photo_failures"].get(name)
            if unavailable is not None:
                return f"Public downloader marked {name} unavailable: {unavailable}"
            if path.is_file():
                return None
            count = status["photo_count"]
            if rank >= count:
                raise FileNotFoundError(
                    f"Staged photo prefix of {count} exhausted at rank {rank}; exactly {remaining} more accepted variants "
                    f"are needed. Extend the download to at least --photo-count {rank + remaining}."
                )
            if (status["state"] != "running" or status["scope"] not in ("all", "images") or
                    status["processed_photos"] >= count or not downloading(self.folder)):
                raise FileNotFoundError(
                    f"Downloader is not supplying {name} (state={status['state']}, rank={rank}). "
                    f"Exactly {remaining} further accepted variants needed. Restore/resume the photo transfer, then resume generation."
                )
            now = perf_counter()
            if last_log is None or now - last_log >= 60:
                print(f"Waiting for JPEG {name} at rank {rank}; downloader processed {status['processed_photos']}/{count}.", flush=True)
                last_log = now
            sleep(3)


def load_renderer(args, parameters):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA device is required.")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        str(args.model_dir), local_files_only=True, torch_dtype=torch.float16,
        use_safetensors=True, add_watermarker=False, variant=parameters["variant"],
    )
    pipe.set_progress_bar_config(disable=True)
    embeddings = {}

    @torch.inference_mode()
    def encode(prompt):
        return pipe.encode_prompt(
            prompt=prompt, device=torch.device("cuda"), num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt=parameters["negative_prompt"],
        )

    cached_only = bool(getattr(args, "cached_text_only", False))
    if cached_only:
        import gc
        import weakref
        # Only the encoders are on CUDA while caching the two possible prompts.
        pipe.text_encoder.to("cuda")
        pipe.text_encoder_2.to("cuda")
        refs = [weakref.ref(pipe.text_encoder), weakref.ref(pipe.text_encoder_2)]
        for grayscale in (False, True):
            prompt = variant_prompt(parameters, grayscale)
            tensors = encode(prompt)
            proj_dim = pipe.text_encoder_2.config.projection_dim
            if (any(t.device.type != "cuda" or t.dtype != pipe.unet.dtype for t in tensors)
                    or tensors[2].shape[-1] != proj_dim or tensors[3].shape[-1] != proj_dim):
                raise ValueError("Precomputed CUDA text embeddings do not match the UNet conditioning.")
            embeddings[prompt] = tensors
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        pipe.register_modules(text_encoder=None, text_encoder_2=None)
        gc.collect()
        torch.cuda.empty_cache()
        if any(ref() is not None for ref in refs):
            raise RuntimeError("A text encoder is still referenced after precomputation.")
        print(f"Cached two prompts; released both text encoders, {(before - torch.cuda.memory_allocated()) / 2**20:.1f} MiB.", flush=True)

    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if pipe._execution_device.type != "cuda":
        raise RuntimeError(f"SDXL execution device must be CUDA: {pipe._execution_device}")

    @torch.inference_mode()
    def render(image, *, seed, prompt):
        if prompt not in embeddings:
            if cached_only:
                raise ValueError("Unexpected new prompt after releasing text encoders.")
            embeddings[prompt] = encode(prompt)
        positive, negative, pooled, negative_pooled = embeddings[prompt]
        return pipe(
            image=image, prompt_embeds=positive, negative_prompt_embeds=negative,
            pooled_prompt_embeds=pooled, negative_pooled_prompt_embeds=negative_pooled,
            strength=parameters["strength"], num_inference_steps=parameters["steps"],
            guidance_scale=parameters["guidance_scale"], num_images_per_prompt=1,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            original_size=(image.height, image.width), target_size=(image.height, image.width),
            crops_coords_top_left=(0, 0), output_type="pil",
        ).images[0]
    return render


def variant_prompt(parameters, grayscale):
    suffix = " Preserve black-and-white grayscale photography." if grayscale else " Preserve the original colors."
    return parameters["prompt"] + suffix


def summarize(conn, store, initial_accepted, started, stop_reason):
    counts = dict(conn.execute("SELECT status, count(*) FROM samples GROUP BY status"))
    accepted = counts.get("accepted", 0)
    rate = (accepted - initial_accepted) / max(perf_counter() - started, 1e-6)
    summary = {
        "database": str(store / "records.sqlite"), "archived_accepted": ARCHIVED_COUNT,
        "target_new_accepted": TARGET, "complete": accepted == TARGET,
        **{key: counts.get(key, 0) for key in ("accepted", "rejected", "source_unavailable", "pending", "error")},
        "new_accepted_per_second_this_invocation": rate,
        "eta_seconds": (TARGET - accepted) / rate if rate > 0 else None,
        "reason": stop_reason, "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    recipe._atomic_json(store / "generation_summary.json", summary)
    conn.execute("INSERT OR REPLACE INTO run_metadata VALUES ('complete', ?)", (recipe._json(summary["complete"]),))
    conn.commit()
    print(json.dumps(summary, separators=(",", ":")), flush=True)
    return summary


def run(args, *, renderer_factory=load_renderer, weights_checker=verify_weights):
    archived = load_json(ARCHIVE / "synthetic_store/generation_manifest.json")
    parameters = archived["parameters"]
    parents, hashes, order_hash = selection(args, archived)
    print(f"Verified all input hashes and archived 512-parent prefix; {len(parents)} eligible parents.", flush=True)
    if args.selection_only:
        return
    public_source = PublicPhotoSource(args.images_root, parents) if args.wait_for_photos else None
    if public_source:
        public_source.read()
    weights_checker(args.model_dir, archived)
    runtime = runtime_info()
    config = {
        "recipe": "archived SDXL img2img training-only extension", "schema_version": 1,
        "input_sha256": hashes, "eligible_order_sha256": order_hash, "archived_accepted": ARCHIVED_COUNT,
        "target_new_accepted": TARGET, "model_revision": archived["model_revision"],
        "model_sha256": archived["model_sha256"], "parameters": parameters, "runtime": runtime,
        "images_root": str(args.images_root.resolve()), "helpers_sha256": recipe._sha256(Path(recipe.__file__)),
        "weights_verified": True,
    }
    store = args.store
    store.mkdir(parents=True, exist_ok=True)
    with writer_lock(store / ".generator.lock"), closing(sqlite3.connect(store / "records.sqlite")) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        recipe._initialize(conn, config, archived["model_files"], resume=True, count=len(parents))
        allow_unavailable_status(conn)
        recipe._atomic_json(store / "generation_manifest.json", {**config, "model_files": archived["model_files"]})
        (store / "images").mkdir(exist_ok=True)
        previous = {r["selection_rank"]: r for r in conn.execute("SELECT * FROM samples")}
        base_accepted = sum(r["status"] == "accepted" for r in previous.values())
        if any(rank < ARCHIVED_COUNT or rank >= len(parents) or int(parents.index[rank]) != row["parent_id"]
               for rank, row in previous.items()):
            raise ValueError("Recorded sample IDs differ from the seed-0 selection.")
        started = perf_counter()
        attempts = accepted = 0
        render = None
        try:
            for rank in range(ARCHIVED_COUNT, len(parents)):
                pid = int(parents.index[rank])
                path = args.images_root / f"{pid//1000:04d}" / f"{pid}.jpg"
                relative = f"images/{pid}.png"
                output = store / relative
                row = previous.get(rank)
                if row and row["status"] == "source_unavailable":
                    if row["image_path"] or row["image_sha256"]:
                        raise ValueError(f"Unexpected generated file recorded for unavailable parent {pid}.")
                    continue
                if row and row["status"] in {"accepted", "rejected"}:
                    if row["image_path"] != relative or recipe._sha256(output) != row["image_sha256"] or recipe._sha256(path) != row["parent_sha256"]:
                        raise ValueError(f"Saved PNG or source JPEG changed for parent {pid}.")
                    accepted += row["status"] == "accepted"
                    if accepted == TARGET:
                        break
                    continue
                parent = parents.iloc[rank]
                seed = int(np.random.SeedSequence([parameters["seed"], pid]).generate_state(1)[0])
                sample = {
                    "synthetic_id": -pid, "parent_id": pid, "selection_rank": rank,
                    "parent_split": "train", "is_synthetic": 1, "is_diffusion": 1,
                    "source": "diffusion", "generator": "sdxl_img2img", "status": "pending",
                    "image_path": relative, "latitude": float(parent.latitude), "longitude": float(parent.longitude),
                    "date_shot_min": recipe._text(parent.date_shot_min), "date_shot_max": recipe._text(parent.date_shot_max),
                    "view_type": recipe._text(parent.view_type), "block": recipe._text(parent.block),
                    "seed": seed, "parent_path": str(path), "model_revision": config["model_revision"],
                    "model_sha256": config["model_sha256"], "parameters_json": recipe._json(parameters),
                    "created_at": row["created_at"] if row else recipe._now(), "updated_at": recipe._now(),
                }
                columns = ",".join(sample)
                placeholders = ",".join("?" for _ in sample)
                conn.execute(f"INSERT OR REPLACE INTO samples ({columns}) VALUES ({placeholders})", tuple(sample.values()))
                conn.commit()
                try:
                    if public_source:
                        unavailable = public_source.await_source(path, rank, TARGET - accepted)
                        if unavailable is not None:
                            conn.execute(
                                "UPDATE samples SET status='source_unavailable',image_path='',error=?,updated_at=? WHERE parent_id=?",
                                (unavailable, recipe._now(), pid),
                            )
                            conn.commit()
                            summarize(conn, store, base_accepted, started, f"rank {rank}: {unavailable}")
                            continue
                    if not path.is_file():
                        raise FileNotFoundError(f"Stage missing original JPEG, then resume: rank={rank} file={path}")
                    raw = path.read_bytes()
                    parent_sha = hashlib.sha256(raw).hexdigest()
                    with Image.open(io.BytesIO(raw)) as parent_image:
                        canvas, geometry, grey = recipe.prepare_image(parent_image, max_side=parameters["max_side"])
                    prompt = variant_prompt(parameters, grey)
                    while True:
                        try:
                            if render is None:
                                print(f"Loading local SDXL at {args.model_dir} (CPU offload: {args.offload}).", flush=True)
                                render = renderer_factory(args, parameters)
                            generated = render(canvas, seed=seed, prompt=prompt)
                            break
                        except RuntimeError as error:
                            if args.offload or "out of memory" not in str(error).lower():
                                raise
                            print("CUDA memory exhausted; retrying this seed with model CPU offload.", flush=True)
                            args.offload = True
                            render = None
                        import gc
                        import torch
                        gc.collect()
                        torch.cuda.empty_cache()
                    pixels = recipe.restore_image(generated, geometry, grayscale=grey)
                    original = recipe.restore_image(canvas, geometry, grayscale=grey)
                    quality = recipe.quality_check(original, pixels, min_structure=parameters["min_structure"])
                    partial = output.with_suffix(".png.part")
                    pixels.save(partial, format="PNG")
                    image_sha = recipe._sha256(partial)
                    partial.replace(output)
                    status = "accepted" if quality["passed"] else "rejected"
                    render_info = dict(parameters, prompt=prompt, geometry=geometry, runtime=runtime, cpu_offload=args.offload,
                                       text_encoders_released=bool(getattr(args, "cached_text_only", False)))
                    conn.execute(
                        "UPDATE samples SET status=?,parent_sha256=?,image_sha256=?,width=?,height=?,"
                        "parent_width=?,parent_height=?,is_grayscale=?,quality_json=?,parameters_json=?,updated_at=? WHERE parent_id=?",
                        (status, parent_sha, image_sha, *pixels.size, *geometry["parent_size"], int(grey),
                         recipe._json(quality), recipe._json(render_info), recipe._now(), pid),
                    )
                    conn.commit()
                except Exception as error:
                    conn.execute("UPDATE samples SET status='error',error=?,updated_at=? WHERE parent_id=?",
                                 (repr(error), recipe._now(), pid))
                    conn.commit()
                    raise
                attempts += 1
                accepted += status == "accepted"
                if attempts <= 3 or attempts % args.progress_every == 0:
                    summarize(conn, store, base_accepted, started, f"rank {rank}, parent {pid}, {status}")
                if accepted == TARGET or attempts == args.stop_after:
                    break
        except Exception as error:
            summarize(conn, store, base_accepted, started, repr(error))
            raise
        reason = "target reached" if accepted == TARGET else "pilot limit reached"
        if accepted != TARGET and attempts != args.stop_after:
            raise RuntimeError("The available permutation was exhausted before reaching the accepted target.")
        return summarize(conn, store, base_accepted, started, reason)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/t1_original")
    parser.add_argument("--split-csv", type=Path, default=ROOT / "data/splits/main.csv.gz")
    parser.add_argument("--images-root", type=Path, default=ASSETS / "images")
    parser.add_argument("--model-dir", type=Path, default=ASSETS / "sdxl-fp16")
    parser.add_argument("--store", type=Path, default=VOLUME / "aug8192")
    parser.add_argument("--stop-after", type=int, help="Stop after N new SDXL renders (accepted + QA-rejected) this invocation; skipped sources and prior renders do not count.")
    parser.add_argument("--wait-for-photos", action="store_true", help="Follow the adjacent public_asset_status.json; wait for pending JPEGs and permanently skip its explicit source failures.")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--offload", action="store_true", help="Use Accelerate model CPU offload during CUDA inference.")
    parser.add_argument("--cached-text-only", action="store_true", help="Cache both fixed prompts and release both CUDA text encoders before image generation.")
    parser.add_argument("--selection-only", action="store_true", help="Verify the archived prefix and return without writes.")
    args = parser.parse_args()
    if (args.stop_after is not None and args.stop_after < 1) or args.progress_every < 1:
        parser.error("stop-after and progress-every must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
