import numpy as np
import pandas as pd
import pytest

from models.t6_router import geographic_gate as gate
from models.t6_router import gate_confidence


@pytest.fixture
def examples():
    # All ten blocks contain both types of site. Only predicted location
    # distinguishes them; branch confidence is identical for every photo.
    i = np.arange(1500)
    block, within = np.divmod(i, 150)
    east = within < 50
    both, neither = within % 5 == 3, within % 5 == 4
    hit4 = (east & ~neither) | both
    hit5 = (~east & ~neither) | both
    frame = pd.DataFrame({
        "id": 10000 + i, "block": block.astype(str), "split": "val",
        "T4_lat_pred": 46.5, "T4_lon_pred": np.where(east, 8.9, 6.3),
        "T5_lat_pred": 46.6, "T5_lon_pred": np.where(east, 9.0, 6.4),
        "T4_p25": 0.20, "T5_p25": 0.40,
        "T4_error_km": np.where(hit4, 25.0, 25.01),
        "T5_error_km": np.where(hit5, 25.0, 25.01),
        # Extra columns intentionally reveal the answer.
        "latitude": hit4.astype(float), "expert_gap_km": hit4.astype(float),
    })
    old = frame[["id", "block"]].copy()
    old["fold"] = block % 5 + 1
    old["T4_refit_p25"] = 0.20
    old["T5_refit_p25"] = 0.40
    old["archived_T6_choose_T4"] = False
    old["block_refit_T6_choose_T4"] = False
    return frame, old


def test_gate_uses_only_six_inputs_and_learns_location(examples):
    frame, _ = examples
    assert set(gate.POSITION) == {"T4_lat_pred", "T4_lon_pred", "T5_lat_pred", "T5_lon_pred"}
    assert set(gate.NONPOSITION) == {"T4_p25", "T5_p25"}
    east = frame.T4_lon_pred > 8
    with_location = gate.fit_gate(frame)
    input_only = frame[[*gate.POSITION, *gate.NONPOSITION]]
    np.testing.assert_array_equal(with_location.predict_t4_probability(input_only) > 0.5, east)
    without_location = gate.fit_gate(frame, use_position=False)
    assert not (without_location.predict_t4_probability(frame[list(gate.NONPOSITION)]) > 0.5).any()

    # Adding perfectly informative truth or errors for a query cannot change
    # the prediction made from the six deployment features.
    altered = frame.assign(block="other", latitude=0, expert_gap_km=100, T4_error_km=0, T5_error_km=100)
    np.testing.assert_array_equal(with_location.predict_t4_probability(altered),
                                  with_location.predict_t4_probability(input_only))


def test_fit_ignores_agreements_and_treats_25_km_as_a_hit(examples):
    frame, _ = examples
    disagree = (frame.T4_error_km <= 25) != (frame.T5_error_km <= 25)
    all_rows = gate.fit_gate(frame)
    only_disagreements = gate.fit_gate(frame.loc[disagree])
    np.testing.assert_array_equal(all_rows.predict_t4_probability(frame),
                                  only_disagreements.predict_t4_probability(frame))
    for err4, err5, choose4 in [(25, 25.01, True), (25.01, 25, False), (0, 0, False), (26, 26, False)]:
        fitted = gate.fit_gate(frame.assign(T4_error_km=err4, T5_error_km=err5))
        np.testing.assert_array_equal(fitted.predict_t4_probability(frame) > 0.5,
                                      np.full(len(frame), choose4))


def test_nested_gate_and_paired_counts(examples, monkeypatch):
    frame, old = examples
    # Mark archived confidence values so that passing them as gate inputs
    # would fail. Old T6's own reference decisions still use these values.
    frame = frame.assign(T4_p25=0.77, T5_p25=0.99)
    outer_training_sets = []

    def inner_confidence(train_frame, *, n_splits=4):
        outer_training_sets.append(frozenset(train_frame.id))
        return np.tile([0.20, 0.40], (len(train_frame), 1))

    actual_fit = gate.fit_gate
    actual_predict = gate.GeographicGate.predict_t4_probability

    def check_fit(train_frame, use_position=True):
        assert train_frame.T4_p25.eq(0.20).all()
        assert train_frame.T5_p25.eq(0.40).all()
        return actual_fit(train_frame, use_position=use_position)

    def check_predict(self, held_frame):
        np.testing.assert_array_equal(held_frame.T4_p25, old.set_index("id").loc[held_frame.id, "T4_refit_p25"])
        np.testing.assert_array_equal(held_frame.T5_p25, old.set_index("id").loc[held_frame.id, "T5_refit_p25"])
        return actual_predict(self, held_frame)

    monkeypatch.setattr(gate, "crossfit_confidence", inner_confidence)
    monkeypatch.setattr(gate, "fit_gate", check_fit)
    monkeypatch.setattr(gate.GeographicGate, "predict_t4_probability", check_predict)
    report, rows = gate.evaluate_gate(frame, old)
    expected = {frozenset(old.loc[old.fold != fold, "id"]) for fold in range(1, 6)}
    assert set(outer_training_sets) == expected
    assert rows.id.is_unique and set(rows.id) == set(frame.id)
    assert len(report["folds"]) == 5
    aligned = frame.set_index("id").loc[rows.id]
    np.testing.assert_array_equal(rows.geographic_gate_choose_T4, aligned.T4_lon_pred > 8)
    assert not rows.without_position_choose_T4.any()
    for name in ("vs_archived_T6", "vs_block_refit_T6", "vs_without_position", "vs_always_T5"):
        counts = report["comparisons"][name]
        assert (counts["rescued"], counts["lost"], counts["net"]) == (300, 0, 300)
        assert counts["net"] == counts["candidate_hits"] - counts["baseline_hits"]

    held = old.fold == 1
    swapped = frame.copy()
    swapped.loc[held, ["T4_error_km", "T5_error_km"]] = frame.loc[held, ["T5_error_km", "T4_error_km"]].to_numpy()
    _, repeat = gate.evaluate_gate(swapped, old)
    ids = old.loc[held, "id"]
    np.testing.assert_array_equal(rows.set_index("id").loc[ids, "geographic_gate_p4"],
                                  repeat.set_index("id").loc[ids, "geographic_gate_p4"])


def test_outer_folds_cannot_split_one_geographic_block(examples):
    frame, old = examples
    old.loc[0, "fold"] = 2
    with pytest.raises(ValueError):
        gate.evaluate_gate(frame, old)


def test_each_inner_score_excludes_its_own_block(examples, monkeypatch):
    frame, _ = examples
    visited = []

    def refit(source, train, held, blocks):
        assert set(blocks[train]).isdisjoint(blocks[held])
        assert set(source.iloc[train].id).isdisjoint(source.iloc[held].id)
        visited.extend(source.iloc[held].id)
        # Distinct scores indexed by query expose a row-order mismatch.
        score = (source.iloc[held].id.to_numpy() - 10000) / 2000
        return {"T4": score, "T5": 1 - score}

    monkeypatch.setattr(gate_confidence, "_refit_confidence", refit)
    shuffled = frame.sample(frac=1, random_state=89)
    scores = gate_confidence.crossfit_confidence(shuffled)
    assert sorted(visited) == sorted(frame.id)
    expected = (shuffled.id.to_numpy() - 10000) / 2000
    np.testing.assert_allclose(scores, np.column_stack([expected, 1 - expected]))
