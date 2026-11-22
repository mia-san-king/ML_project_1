import json

import pandas as pd
import pytest

from models.t6_router.run_regional_closer import load_expert_rows
from scripts.confidence.evaluate import sha256


def test_loads_exact_expert_predictions_from_the_completed_study(tmp_path):
    folds = {}
    for number, region in enumerate(("dev1", "dev2", "dev3", "final"), 1):
        path = tmp_path / region / "experts/predictions.parquet"
        path.parent.mkdir(parents=True)
        pd.DataFrame({"id": [number], "block": [10 * number], "region": [region]}).to_parquet(path)
        folds[region] = {"query_photos": 1, "predictions_sha256": sha256(path)}
    (tmp_path / "study.json").write_text(json.dumps({"inputs": {"folds": folds}}))

    development, final = load_expert_rows(tmp_path)
    assert development.id.tolist() == [1, 2, 3]
    assert final.id.tolist() == [4]
    path = tmp_path / "final/experts/predictions.parquet"
    pd.DataFrame({"id": [5], "block": [40], "region": ["final"]}).to_parquet(path)
    with pytest.raises(ValueError, match="final"):
        load_expert_rows(tmp_path)
