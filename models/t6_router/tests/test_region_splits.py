import json

import numpy as np
import pandas as pd
import pytest

from models.t6_router.region_splits import REGIONS, write_region_splits


def make_inputs(tmp_path):
    cores = [(name, x * 100_000 + y) for name, (x, y) in REGIONS.items()]
    rows = [(i, "train", b) for i, (_, b) in enumerate(cores, start=1)]
    rows.extend([(5, "train", cores[0][1] - 100_001), (6, "train", cores[3][1] - 100_001),
                 (7, "train", 25500120), (8, "val", 25500121), (9, "test_in", 25500122),
                 (10, "val", cores[0][1])])
    path = tmp_path / "canonical.csv"
    pd.DataFrame(rows, columns=["id", "split", "block"]).to_csv(path, index=False)
    return path, np.array([i for i, _, _ in rows] + [-i for i in range(1, 8)])


def test_each_region_withholds_its_buffer_and_all_parent_variants(tmp_path):
    source, ids = make_inputs(tmp_path)
    out = tmp_path / "split"
    files = write_region_splits(canonical_split=source, augmented_ids=ids, output_dir=out)
    info = json.loads((out / "manifest.json").read_text())
    assert info["regions"]["dev1"]["canonical_real_queries"] == 2
    for i, (name, file) in enumerate(files.items(), start=1):
        table = pd.read_csv(file).set_index("id")
        assert np.array_equal(table.index.to_numpy(), ids)
        assert table.loc[i, "split"] == "query"
        assert table.loc[-i, "split"] == "ignore"
        assert table.loc[4, "split"] == ("query" if name == "final" else "ignore")
        assert table.loc[[-4, 6, -6, 9], "split"].eq("ignore").all()
        assert table.loc[[7, -7], "split"].eq("train").all()
        assert table.loc[8, "split"] == "val"
        assert table.region.dropna().eq(name).all()
        if name == "dev1":
            assert table.loc[[5, -5], "split"].eq("ignore").all()
            assert table.loc[10, "split"] == "query"


def test_previously_seen_final_region_is_rejected_before_publication(tmp_path):
    path, ids = make_inputs(tmp_path)
    rows = pd.read_csv(path)
    rows.loc[rows.id.eq(6), "split"] = "val"
    rows.to_csv(path, index=False)
    with pytest.raises(ValueError, match="final region"):
        write_region_splits(canonical_split=path, augmented_ids=ids, output_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_variant_needs_cached_real_training_parent(tmp_path):
    source, ids = make_inputs(tmp_path)
    with pytest.raises(ValueError, match="no real cached parent"):
        write_region_splits(canonical_split=source, augmented_ids=ids[ids != 3], output_dir=tmp_path / "bad")
