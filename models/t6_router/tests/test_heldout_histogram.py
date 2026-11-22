import pandas as pd
import pytest

from models.t6_router.heldout_histogram import summarize


def test_frequency_bins_include_upper_edge_and_count_each_photo_once():
    frame = pd.DataFrame({
        "T1_error_km": [0, 25, 25.01, 50, 250, 300],
        "T3_error_km": [10, 26, 20, 26, 80, 150],
        "T6_error_km": [10, 26, 20, 24, 80, 150],
    })
    summary = summarize(frame)
    counts = summary["models"]["T1"]["counts"]
    assert counts[:2] == [2, 2] and counts[-2:] == [1, 1]
    for model in summary["models"].values():
        assert sum(model["counts"]) == summary["n"]
        assert sum(model["frequency_pct"]) == pytest.approx(100)
    assert summary["T6_vs"]["T1"] == {"rescued_25km": 2, "lost_25km": 1, "net_25km": 1}
