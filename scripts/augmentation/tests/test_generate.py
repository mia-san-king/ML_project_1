"""Generation orchestration tests use a tiny deterministic renderer, no weights."""
import importlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from PIL import Image
import pytest


def module():
    name = "scripts.augmentation.generate"
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, "augmentation generator is not implemented"
    return importlib.import_module(name)


@pytest.fixture
def inputs(tmp_path):
    data, images, model = tmp_path / "data", tmp_path / "original_images", tmp_path / "sdxl"
    (data / "meta").mkdir(parents=True)
    (data / "features").mkdir()
    images.mkdir()
    model.mkdir()
    (model / "model_index.json").write_text('{"_class_name":"StableDiffusionXLPipeline"}')
    (model / "weights.safetensors").write_bytes(b"test fixture, never loaded as a model")
    ids = [1001, 1002, 1003, 1004, 1005]
    metadata = pd.DataFrame({"id": ids, "latitude": [46.] * 5, "longitude": [7.] * 5,
                             "date_shot_min": ["1900-01-01", None, "1950-01-01", None, None],
                             "date_shot_max": ["1901-01-01", None, "1950-01-01", None, None],
                             "view_type": ["oblique"] * 5})
    metadata.to_parquet(data / "meta/smapshot.parquet", index=False)
    split = tmp_path / "split.csv"
    pd.DataFrame({"id": ids, "split": ["train"] * 3 + ["val", "test_in"]}).to_csv(split, index=False)
    np.save(data / "features/clip_vitl14_ids.npy", np.array(ids[::-1]))
    values = np.tile(np.linspace(20, 220, 80, dtype=np.uint8), (50, 1))
    for image_id in ids:
        directory = images / f"{image_id // 1000:04d}"
        directory.mkdir(exist_ok=True)
        Image.fromarray(values).convert("RGB").save(directory / f"{image_id}.jpg")
    return data, images, model, split, tmp_path / "store"


def arguments(mod, inputs, count=1, resume=False):
    data, images, model, split, store = inputs
    values = ["--data-root", str(data), "--images-root", str(images), "--model-dir", str(model),
              "--model-revision", "fixture-revision", "--split-csv", str(split), "--store", str(store),
              "--count", str(count), "--seed", "7", "--device", "cpu", "--max-side", "512"]
    return mod.parse_args(values + (["--resume"] if resume else []))


def renderer(calls, *, copy=False):
    def factory(args):
        def render(image, *, seed, prompt, negative_prompt):
            calls.append(seed)
            if copy:
                return image.copy()
            pixels = np.asarray(image, dtype=np.int16)
            return Image.fromarray(np.clip(pixels + 6, 0, 255).astype(np.uint8))
        return render
    return factory


def rows(store):
    with sqlite3.connect(store / "synthetic.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM samples ORDER BY selection_rank")]


def test_train_only_deterministic_generation_resume_and_count_extension(inputs):
    mod, calls = module(), []
    args = arguments(mod, inputs)
    first = mod.run_generation(args, generator_factory=renderer(calls))
    assert first["accepted"] == 1 and len(calls) == 1
    old = rows(inputs[-1])[0]
    assert old["parent_id"] in {1001, 1002, 1003} and old["synthetic_id"] == -old["parent_id"]
    assert old["status"] == "accepted" and old["parent_split"] == "train"
    assert old["source"] == "diffusion" and old["generator"] == "sdxl_img2img"
    assert old["is_synthetic"] == old["is_diffusion"] == old["is_grayscale"] == 1
    assert json.loads(old["quality_json"])["passed"]
    with Image.open(inputs[-1] / old["image_path"]) as image:
        assert image.mode == "L" and image.size == (old["width"], old["height"])
    second = mod.run_generation(arguments(mod, inputs, count=3, resume=True), generator_factory=renderer(calls))
    assert second["accepted"] == 3 and len(calls) == 3
    current = rows(inputs[-1])
    assert current[0]["synthetic_id"] == old["synthetic_id"] and current[0]["image_sha256"] == old["image_sha256"]
    assert {row["parent_id"] for row in current} == {1001, 1002, 1003}
    assert len({row["seed"] for row in current}) == 3
    mod.run_generation(arguments(mod, inputs, count=3, resume=True), generator_factory=renderer(calls))
    assert len(calls) == 3


def test_a_copied_parent_is_rejected_and_never_marked_accepted(inputs):
    mod = module()
    result = mod.run_generation(arguments(mod, inputs), generator_factory=renderer([], copy=True))
    assert result["accepted"] == 0 and result["rejected"] == 1
    row = rows(inputs[-1])[0]
    assert "difference_too_small" in json.loads(row["quality_json"])["reasons"]


def test_padding_preserves_full_parent_content_and_alignment():
    mod = module()
    image = Image.fromarray(np.arange(17 * 11, dtype=np.uint8).reshape(11, 17)).convert("RGB")
    canvas, geometry, gray = mod.prepare_image(image, max_side=512)
    assert canvas.width % 64 == canvas.height % 64 == 0
    assert min(canvas.size) >= 512
    restored = mod.restore_image(canvas, geometry, grayscale=gray)
    expected = image.resize(tuple(geometry["content_size"]), Image.Resampling.LANCZOS).convert("L")
    np.testing.assert_array_equal(restored, expected)
    assert abs(restored.width / restored.height - image.width / image.height) < .01


@pytest.mark.parametrize("change", ["parameters", "weights", "generated_file"])
def test_resume_rejects_changed_configuration_weights_or_saved_file(inputs, change):
    mod = module()
    mod.run_generation(arguments(mod, inputs), generator_factory=renderer([]))
    args = arguments(mod, inputs, resume=True)
    if change == "parameters":
        args.strength = .2
    elif change == "weights":
        (inputs[2] / "weights.safetensors").write_bytes(b"changed fixture weights")
    else:
        (inputs[-1] / rows(inputs[-1])[0]["image_path"]).write_bytes(b"corrupted output")
    with pytest.raises((ValueError, FileExistsError)):
        mod.run_generation(args, generator_factory=renderer([]))


def test_check_data_does_not_create_store_or_instantiate_model(inputs):
    mod = module()
    args = arguments(mod, inputs)
    args.check_data = True
    def forbidden(_):
        raise AssertionError("must not instantiate a model")
    result = mod.run_generation(args, generator_factory=forbidden)
    assert result["eligible_train"] == 3 and result["selected"] == 1
    assert not inputs[-1].exists()


def test_negative_original_ids_are_rejected_before_store_creation(inputs):
    mod = module()
    np.save(inputs[0] / "features/clip_vitl14_ids.npy", np.array([-1001, 1002]))
    with pytest.raises(ValueError, match="positive"):
        mod.run_generation(arguments(mod, inputs), generator_factory=renderer([]))
    assert not inputs[-1].exists()


def test_model_load_failure_is_recorded_once_and_stops_instead_of_repeating(inputs):
    mod, calls = module(), []
    def unavailable(_):
        calls.append(1)
        raise RuntimeError("local model load failed")
    with pytest.raises(RuntimeError, match="model load"):
        mod.run_generation(arguments(mod, inputs, count=2), generator_factory=unavailable)
    assert len(calls) == 1
    assert len(rows(inputs[-1])) == 1 and rows(inputs[-1])[0]["status"] == "error"


def test_resume_retries_errors_with_the_same_parent_and_seed(inputs):
    mod, calls = module(), []
    def intermittent(args):
        render = renderer(calls)(args)
        def run(image, **kwargs):
            if not calls:
                calls.append(kwargs["seed"])
                raise RuntimeError("temporary per-image failure")
            return render(image, **kwargs)
        return run
    first = mod.run_generation(arguments(mod, inputs, count=2), generator_factory=intermittent)
    assert first["error"] == 1 and first["accepted"] == 1
    before = rows(inputs[-1])
    second = mod.run_generation(arguments(mod, inputs, count=2, resume=True), generator_factory=renderer(calls))
    assert second["error"] == 0 and second["accepted"] == 2
    assert [(row["parent_id"], row["seed"]) for row in rows(inputs[-1])] == [(row["parent_id"], row["seed"]) for row in before]
