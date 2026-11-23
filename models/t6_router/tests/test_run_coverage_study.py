import json

import pandas as pd
import pytest

from models.t6_router.run_coverage_study import read_fold
from scripts.confidence.evaluate import sha256


def test_read_fold_rejects_wrong_assignment_even_with_matching_prediction_digest(tmp_path):
    split = tmp_path / "dev1.csv.gz"
    pd.DataFrame({"id": [1, 2], "split": ["train", "query"], "block": [111, 222]}).to_csv(split, index=False)
    experts = tmp_path / "dev1/experts"
    experts.mkdir(parents=True)
    predictions = experts / "predictions.parquet"
    def write_prediction(block):
        pd.DataFrame({"id": [2], "region": ["dev1"], "block": [block]}).to_parquet(predictions, index=False)
        info = {"inputs": {"split_sha256": sha256(split)}, "query_count": 1,
                "predictions_sha256": sha256(predictions)}
        (experts / "summary.json").write_text(json.dumps(info), encoding="utf-8")
    write_prediction(222)
    frame, _ = read_fold(tmp_path, "dev1", split)
    assert frame.id.tolist() == [2]
    write_prediction(333)
    with pytest.raises(ValueError, match="block"):
        read_fold(tmp_path, "dev1", split)
