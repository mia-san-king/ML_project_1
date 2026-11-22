"""Small, offline checks for synthetic-data isolation and cache calibration."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager

import numpy as np
import pandas as pd
from PIL import Image


def sample(parent_id=10, **changes):
    row = dict(
        synthetic_id=-parent_id, parent_id=parent_id, parent_split="train",
        is_synthetic=1, is_diffusion=1, source="diffusion", generator="sdxl_img2img",
        status="accepted", image_path=f"images/{-parent_id}.png", latitude=46.1,
        longitude=7.1, date_shot_min="1900", date_shot_max="1910", view_type="terrestrial",
        block="11", seed=7, parent_path="", parent_sha256="", image_sha256="",
        width=16, height=16, parent_width=16, parent_height=16, is_grayscale=1,
        quality_json="{}", model_revision="revision", model_sha256="model-sha",
        parameters_json='{"strength":0.15}', created_at="2026-11-20T00:00:00Z",
        updated_at="2026-11-20T00:00:00Z", error=None,
    )
    row.update(changes)
    return row


def write_store(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    columns = list(sample())
    numeric = {"synthetic_id", "parent_id", "is_synthetic", "is_diffusion", "seed", "width", "height", "parent_width", "parent_height", "is_grayscale"}
    with sqlite3.connect(root / "synthetic.sqlite") as db:
        db.execute("CREATE TABLE samples (" + ",".join(f"{name} {'INTEGER' if name in numeric else 'TEXT'}" for name in columns) + ")")
        for row in rows:
            if row["status"] == "accepted" and not Path(row["image_path"]).is_absolute():
                image = root / row["image_path"]
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (16, 16), color=(30, 30, 30)).save(image)
                row["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
            db.execute("INSERT INTO samples VALUES (" + ",".join("?" for _ in columns) + ")", [row[name] for name in columns])


class InputTests(unittest.TestCase):
    def test_only_accepted_diffusion_rows_are_consumed_without_writing_sqlite(self):
        from scripts.augmentation.build_dataset import load_accepted_samples

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_store(root, [sample(20, status="rejected"), sample(10), sample(30, status="error")])
            before = (root / "synthetic.sqlite").read_bytes()
            rows = load_accepted_samples(root)
            self.assertEqual([row["parent_id"] for row in rows], [10])
            self.assertEqual((root / "synthetic.sqlite").read_bytes(), before)

    def test_accepted_id_or_provenance_corruption_is_rejected(self):
        from scripts.augmentation.build_dataset import load_accepted_samples

        for change in ({"synthetic_id": -11}, {"is_diffusion": 0}, {"parent_split": "val"}, {"source": "original"}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_store(root, [sample(**change)])
                with self.assertRaises(ValueError):
                    load_accepted_samples(root)

    def test_modified_accepted_image_fails_its_recorded_hash(self):
        from scripts.augmentation.build_dataset import load_accepted_samples

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_store(root, [sample()])
            (root / "images/-10.png").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "hash|sha256"):
                load_accepted_samples(root)

    def test_parent_comparison_catches_normalized_or_wrong_clip_features(self):
        from scripts.augmentation.build_dataset import validate_parent_features

        original = np.array([[2., 0., 0.], [0., 4., 0.]], dtype=np.float32)
        report = validate_parent_features(original, original * 1.001, [10, 20])
        self.assertGreaterEqual(report["min_cosine"], 0.999)
        for changed in (original / np.linalg.norm(original, axis=1, keepdims=True), original[:, ::-1]):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "parent|CLIP"):
                validate_parent_features(original, changed, [10, 20])

    def test_append_preserves_labels_order_and_validation_ids(self):
        from scripts.augmentation.build_dataset import make_augmented_tables

        metadata, splits = source_tables()
        meta, split, audit = make_augmented_tables(metadata, splits, np.array([30, 10, 20]), [sample()], Path("/store"))
        self.assertEqual(meta.id.tolist(), [20, 30, 10, 40, -10])
        self.assertEqual(split.id.tolist(), [10, 40, 30, 20, -10])
        synthetic = meta.iloc[-1]
        self.assertEqual(synthetic.date_shot_min, pd.Timestamp("1900-01-01"))
        self.assertEqual(synthetic.latitude, 46.1)
        self.assertEqual(synthetic.parent_id, 10)
        self.assertEqual(split.iloc[-1]["split"], "train")
        self.assertEqual(split.iloc[-1]["random_split"], "train")
        self.assertEqual(audit["validation_ids"], [30])
        self.assertEqual(audit["evaluation_splits"]["val"]["count"], 1)
        self.assertEqual(audit["evaluation_splits"]["test_in"]["count"], 1)
        pd.testing.assert_frame_equal(meta.iloc[:4][metadata.columns].reset_index(drop=True), metadata)
        pd.testing.assert_frame_equal(split.iloc[:4].reset_index(drop=True), splits)

    def test_validation_parent_or_uncached_parent_cannot_enter_training(self):
        from scripts.augmentation.build_dataset import make_augmented_tables

        metadata, splits = source_tables()
        for parent_id in (30, 40):
            with self.subTest(parent_id=parent_id), self.assertRaises(ValueError):
                make_augmented_tables(metadata, splits, np.array([30, 10, 20]), [sample(parent_id)], Path("/store"))


def source_tables():
    metadata = pd.DataFrame({
        "id": [20, 30, 10, 40], "latitude": [46.2, 46.3, 46.1, 46.4],
        "longitude": [7.2, 7.3, 7.1, 7.4], "view_type": ["terrestrial"] * 4,
        "date_shot_min": pd.to_datetime(["1920-01-01", "1930-01-01", "1900-01-01", "1940-01-01"]),
        "date_shot_max": pd.to_datetime(["1920-01-01", "1930-01-01", "1910-01-01", "1940-01-01"]),
    })
    splits = pd.DataFrame({"id": [10, 40, 30, 20], "split": ["train", "train", "val", "test_in"], "random_split": ["val", "train", "val", "test_in"], "block": [11, 44, 33, 22]})
    return metadata, splits


@contextmanager
def build_fixture(*, two_parents=False):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, store, output = root / "original", root / "store", root / "augmented"
        for directory in (source / "meta", source / "features", source / "geo", source / "images/0000"):
            directory.mkdir(parents=True, exist_ok=True)
        metadata, splits = source_tables()
        metadata.to_parquet(source / "meta/smapshot.parquet", index=False)
        split_path = source / "split.csv.gz"
        splits.to_csv(split_path, index=False)
        ids = [30, 10, 20, 40] if two_parents else [30, 10, 20]
        features = np.zeros((len(ids), 768), dtype=np.float16)
        features[:, 0] = ids
        np.save(source / "features/clip_vitl14.npy", features)
        np.save(source / "features/clip_vitl14_ids.npy", ids)
        rows = []
        for parent in ([10, 40] if two_parents else [10]):
            path = source / f"images/0000/{parent}.jpg"
            Image.new("RGB", (16, 16), color=(parent, parent, parent)).save(path)
            rows.append(sample(parent, parent_path=str(path), parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        write_store(store, rows)
        yield source, split_path, store, output, features


class PixelEncoder:
    info = {"encoder": "offline test fixture"}

    def encode(self, paths):
        result = np.zeros((len(paths), 768), dtype=np.float32)
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                result[index, 0] = image.getpixel((0, 0))[0]
        return result


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow required for actual parquet integration")
class BuildTests(unittest.TestCase):
    def test_build_is_atomic_resumable_and_preserves_original_cache(self):
        from scripts.augmentation.build_dataset import build_dataset

        with build_fixture() as (source, split_path, store, output, features):
            before = (source / "features/clip_vitl14.npy").read_bytes()
            manifest = build_dataset(source, split_path, store, output, encoder=PixelEncoder(), parent_check_count=1)
            self.assertEqual(manifest["status"], "complete")
            np.testing.assert_array_equal(np.load(output / "features/clip_vitl14.npy")[:3], features)
            np.testing.assert_array_equal(np.load(output / "features/clip_vitl14_ids.npy"), [30, 10, 20, -10])
            self.assertEqual(np.load(output / "features/clip_vitl14.npy").dtype, np.float16)
            self.assertEqual((source / "features/clip_vitl14.npy").read_bytes(), before)
            self.assertEqual((output / "geo").resolve(), (source / "geo").resolve())
            self.assertEqual(json.loads((output / "build_manifest.json").read_text())["validation_ids"], [30])
            self.assertEqual(build_dataset(source, split_path, store, output, encoder=PixelEncoder(), parent_check_count=1), manifest)

    def test_completed_synthetic_batches_survive_an_interrupted_build(self):
        from scripts.augmentation.build_dataset import build_dataset

        class InterruptedEncoder(PixelEncoder):
            def encode(self, paths):
                if any(Path(path).stem == "-40" for path in paths):
                    raise RuntimeError("simulated interruption")
                return super().encode(paths)

        class ResumingEncoder(PixelEncoder):
            def encode(self, paths):
                if any(Path(path).stem == "-10" for path in paths):
                    raise AssertionError("completed synthetic batch must not be re-encoded")
                return super().encode(paths)

        with build_fixture(two_parents=True) as (source, split_path, store, output, features):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                build_dataset(source, split_path, store, output, encoder=InterruptedEncoder(), parent_check_count=1, batch_size=1)
            self.assertFalse(output.exists())
            progress = list(output.parent.glob(".augmented.building-*/progress.json"))
            self.assertEqual(len(progress), 1)
            self.assertEqual(json.loads(progress[0].read_text())["synthetic_completed"], 1)
            build_dataset(source, split_path, store, output, encoder=ResumingEncoder(), parent_check_count=1, batch_size=1)
            np.testing.assert_array_equal(np.load(output / "features/clip_vitl14_ids.npy"), [30, 10, 20, 40, -10, -40])
            np.testing.assert_array_equal(np.load(output / "features/clip_vitl14.npy")[:4], features)

    def test_normalized_parent_features_prevent_publication(self):
        from scripts.augmentation.build_dataset import build_dataset

        class NormalizedEncoder(PixelEncoder):
            def encode(self, paths):
                features = super().encode(paths)
                return features / np.linalg.norm(features, axis=1, keepdims=True)

        with build_fixture() as (source, split_path, store, output, _features):
            with self.assertRaisesRegex(ValueError, "calibration"):
                build_dataset(source, split_path, store, output, encoder=NormalizedEncoder(), parent_check_count=1)
            self.assertFalse(output.exists())

    def test_image_changed_during_encoding_prevents_publication(self):
        from scripts.augmentation.build_dataset import build_dataset

        class MutatingEncoder(PixelEncoder):
            def encode(self, paths):
                features = super().encode(paths)
                for path in paths:
                    if Path(path).stem == "-10":
                        Image.new("RGB", (16, 16), color=(99, 99, 99)).save(path)
                return features

        with build_fixture() as (source, split_path, store, output, _features):
            with self.assertRaisesRegex(ValueError, "hash|sha256"):
                build_dataset(source, split_path, store, output, encoder=MutatingEncoder(), parent_check_count=1)
            self.assertFalse(output.exists())

    def test_incomplete_stage_cannot_be_published(self):
        from scripts.augmentation.build_dataset import _publish

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage, output = root / "stage", root / "output"
            stage.mkdir()
            with self.assertRaises(ValueError):
                _publish(stage, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
