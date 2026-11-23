import json

import numpy as np
import pandas as pd
import pytest

from models.t6_router.expanded_regions import (
    CORNERS, FINAL, SEVEN, THREE, matched_samples, write_expanded_splits,
)
from models.t6_router.region_splits import REGIONS


def test_expanded_folds_hide_both_reserved_areas_and_all_variants(tmp_path):
    rows = []
    for name, (x, y) in {**CORNERS, "archived": REGIONS["final"]}.items():
        for role, block in (("core", x*100_000+y), ("ring", (x-1)*100_000+y)):
            rows.append((len(rows)+1, "train", block, name, role))
    rows.extend([(19, "val", 24000110, "", ""), (20, "train", 24000112, "", ""),
                 (21, "test_in", 24000113, "", ""), (22, "val", CORNERS[FINAL][0]*100_000+CORNERS[FINAL][1], FINAL, "core")])
    canonical = pd.DataFrame(rows, columns=["id", "split", "block", "group", "role"])
    path = tmp_path / "official.csv"
    canonical.iloc[:, :3].to_csv(path, index=False)
    ids = np.r_[canonical.id.to_numpy(), -canonical.id[canonical.split.eq("train")].to_numpy()]
    output = tmp_path / "folds"
    files = write_expanded_splits(canonical_split=path, augmented_ids=ids, output_dir=output)
    assert set(files) == set((*SEVEN, FINAL))
    manifest = json.loads((output / "manifest.json").read_text())
    for name, filename in files.items():
        table = pd.read_csv(filename).set_index("id")
        assert np.array_equal(table.index, ids)
        for row in canonical.itertuples():
            if row.group in (name, FINAL, "archived"):
                expected = "query" if row.group == name and row.role == "core" else "ignore"
                assert table.at[row.id, "split"] == expected
                if -row.id in table.index:
                    assert table.at[-row.id, "split"] == "ignore"
        assert table.loc[19:21, "split"].tolist() == ["val", "train", "ignore"]
        assert manifest["regions"][name]["query"] == len(table[table.split.eq("query")])


def test_matched_picks_equal_budget_without_input_order_or_label_dependence():
    capacities = [12, 6, 3, 3, 2, 2, 1]
    regions = np.repeat(SEVEN, capacities)
    data = pd.DataFrame({"id": range(1, len(regions)+1), "region": regions, "error": 0.})
    sets = matched_samples(data, budget=20)
    assert sets["three"].region.value_counts().to_dict() == dict(zip(THREE, (11, 6, 3)))
    assert sets["seven"].region.value_counts().sort_index().to_dict() == dict(zip(SEVEN, (5,4,3,3,2,2,1)))
    for name, picked in sets.items():
        assert picked.id.is_unique and len(picked) == 20
        shuffled = data.assign(error=np.arange(len(data))).sample(frac=1, random_state=23)
        assert picked.id.tolist() == matched_samples(shuffled, budget=20)[name].id.tolist()
    assert set(sets["seven"].loc[sets["seven"].region.isin(THREE), "id"]) <= set(sets["three"].id)


def test_sampling_rejects_reserved_query_and_shortfall():
    data = pd.DataFrame({"id": np.arange(1, 8), "region": SEVEN})
    with pytest.raises(ValueError, match="budget"):
        matched_samples(data, budget=8)
    with pytest.raises(ValueError, match="regions"):
        matched_samples(data.assign(region=[*SEVEN[:-1], FINAL]), budget=3)
