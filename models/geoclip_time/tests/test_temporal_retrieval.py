"""Visual-neighbor retrieval followed by the teammate's capped date penalty."""

import numpy as np
import pandas as pd
import pytest


def test_gamma_zero_preserves_the_visual_nearest_neighbor():
    from models.geoclip_time.lib.temporal_retrieval import rerank_by_date

    indices = np.array([[2, 0, 1], [1, 2, 0]])
    scores = np.array([[0.9, 0.8, 0.7], [0.95, 0.9, 0.7]])
    gps = np.array([[46.0, 7.0], [47.0, 9.0], [45.0, 6.0]])
    actual = rerank_by_date(
        indices, scores, np.array([1900.0, 2000.0]),
        np.array([1900.0, 2000.0, 1850.0]), gps, gamma=0.0,
    )
    np.testing.assert_array_equal(actual["neighbor_indices"], [2, 1])
    np.testing.assert_array_equal(actual["pred_gps"], gps[[2, 1]])
    np.testing.assert_allclose(actual["scores"], [0.9, 0.95])


def test_date_penalty_is_linear_until_fifty_years_then_capped():
    from models.geoclip_time.lib.temporal_retrieval import rerank_by_date

    actual = rerank_by_date(
        np.arange(3)[:, None], np.ones((3, 1)), np.full(3, 1900.0),
        np.array([1925.0, 1950.0, 2000.0]), np.array([[46, 7], [47, 9], [45, 6]]),
        gamma=0.4,
    )
    np.testing.assert_allclose(actual["year_penalties"], [0.5, 1.0, 1.0])
    np.testing.assert_allclose(actual["scores"], [0.8, 0.6, 0.6])


def test_missing_dates_or_unknown_masks_never_add_a_penalty():
    from models.geoclip_time.lib.temporal_retrieval import rerank_by_date

    actual = rerank_by_date(
        np.array([[0], [1], [2], [0]]), np.full((4, 1), 0.8),
        np.array([np.nan, 1900.0, 1900.0, 1900.0]),
        np.array([1950.0, np.nan, 1950.0]), np.array([[46, 7], [47, 9], [45, 6]]),
        gamma=1.0, query_known=np.array([True, True, True, False]),
        reference_known=np.array([True, True, False]),
    )
    np.testing.assert_array_equal(actual["year_penalties"], np.zeros(4))
    np.testing.assert_allclose(actual["scores"], np.full(4, 0.8))


def test_selected_neighbor_id_and_coordinates_refer_to_the_same_middle_candidate():
    from models.geoclip_time.lib.temporal_retrieval import rerank_by_date

    reference_ids = np.array([101, 205, 309])
    gps = np.array([[46.0, 7.0], [47.0, 9.0], [45.0, 6.0]])
    actual = rerank_by_date(
        np.array([[2, 0, 1]]), np.array([[0.95, 0.9, 0.89]]),
        np.array([1900.0]), np.array([1900.0, 1950.0, 1960.0]), gps, gamma=0.2,
    )
    # The winner is visual rank two, not the final candidate visited by a loop.
    np.testing.assert_array_equal(actual["neighbor_indices"], [0])
    np.testing.assert_array_equal(reference_ids[actual["neighbor_indices"]], [101])
    np.testing.assert_array_equal(actual["pred_gps"], [[46.0, 7.0]])
    np.testing.assert_allclose(actual["visual_scores"], [0.9])
    np.testing.assert_allclose(actual["scores"], [0.9])


def test_reranking_ties_retain_visual_order_instead_of_reference_index_order():
    from models.geoclip_time.lib.temporal_retrieval import rerank_by_date

    # Binary-exact scores and penalties give three exact ties at 0.625.
    actual = rerank_by_date(
        np.array([[2, 0, 1]]), np.array([[0.875, 0.75, 0.625]]),
        np.array([1900.0]), np.array([1906.25, 1900.0, 1912.5]),
        np.array([[46, 7], [47, 9], [45, 6]]), gamma=1.0,
    )
    np.testing.assert_array_equal(actual["neighbor_indices"], [2])
    np.testing.assert_array_equal(actual["scores"], [0.625])


def test_dates_cannot_promote_a_reference_outside_visual_top_k():
    from models.geoclip_time.lib.temporal_retrieval import exact_cosine_topk, rerank_by_date

    # Unnormalized magnitudes intentionally disagree with cosine ordering.
    indices, scores = exact_cosine_topk(
        np.array([[7.0, 0.0]]), np.array([[3.0, 0.0], [8.0, 6.0], [60.0, 80.0]]),
        top_k=2, query_chunk=1,
    )
    np.testing.assert_array_equal(indices, [[0, 1]])
    np.testing.assert_allclose(scores, [[1.0, 0.8]], atol=1e-6)
    actual = rerank_by_date(
        indices, scores, np.array([1900.0]), np.array([1800.0, 1800.0, 1900.0]),
        np.array([[46, 7], [47, 9], [45, 6]]), gamma=3.0,
    )
    # Reference 2 has the closest date but was never a visual candidate.
    np.testing.assert_array_equal(actual["neighbor_indices"], [0])
    np.testing.assert_allclose(actual["scores"], [-2.0])


@pytest.mark.parametrize("query_chunk", [1, 4])
def test_cosine_top_k_uses_reference_index_for_exact_boundary_ties(query_chunk):
    from models.geoclip_time.lib.temporal_retrieval import exact_cosine_topk

    indices, scores = exact_cosine_topk(
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 1], [1, 0], [1, 0], [0, -1]]),
        top_k=2, query_chunk=query_chunk,
    )
    np.testing.assert_array_equal(indices, [[0, 2], [1, 0]])
    np.testing.assert_array_equal(scores, [[1, 1], [1, 0]])


def test_cosine_top_k_reports_completed_query_chunks_without_changing_results():
    from models.geoclip_time.lib.temporal_retrieval import exact_cosine_topk

    progress = []
    indices, scores = exact_cosine_topk(
        np.array([[1, 0], [0, 1], [1, 1], [1, -1], [-1, 1]]),
        np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]), top_k=1, query_chunk=2,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )
    assert progress == [(2, 5), (4, 5), (5, 5)]
    np.testing.assert_array_equal(indices, [[0], [1], [0], [0], [1]])
    np.testing.assert_allclose(scores[:, 0], [1, 1, 2 ** -0.5, 2 ** -0.5, 2 ** -0.5], atol=1e-6)


def test_cosine_top_k_accepts_empty_queries_with_explicit_empty_shapes():
    from models.geoclip_time.lib.temporal_retrieval import exact_cosine_topk

    indices, scores = exact_cosine_topk(np.empty((0, 2)), np.array([[1, 0]]), top_k=1)
    assert indices.shape == scores.shape == (0, 1)
    assert np.issubdtype(indices.dtype, np.integer)


@pytest.mark.parametrize("override", [
    {"query_embeddings": np.array([[0, 0]])},
    {"reference_embeddings": np.array([[0, 0], [1, 0]])},
    {"query_embeddings": np.array([[np.nan, 1]])},
    {"reference_embeddings": np.array([[1, np.inf], [1, 0]])},
    {"reference_embeddings": np.empty((0, 2))},
    {"top_k": 3},
    {"query_chunk": 0},
])
def test_cosine_top_k_rejects_invalid_inputs(override):
    from models.geoclip_time.lib.temporal_retrieval import exact_cosine_topk

    arguments = {
        "query_embeddings": np.array([[1, 0]]),
        "reference_embeddings": np.array([[1, 0], [0, 1]]), "top_k": 1,
        **override,
    }
    with pytest.raises(ValueError):
        exact_cosine_topk(**arguments)


@pytest.mark.parametrize("override", [
    {"indices": np.array([[0, 2]])},
    {"indices": np.array([[0.5, 1]])},
    {"scores": np.array([[0.9, np.nan]])},
    {"query_years": np.array([1900, 1910])},
    {"reference_years": np.array([1900])},
    {"query_known": np.array([True, False])},
    {"reference_known": np.array([True])},
    {"reference_gps": np.array([[np.nan, 7], [47, 9]])},
    {"gamma": -0.1},
    {"cap_years": 0},
])
def test_date_reranking_rejects_misaligned_or_invalid_inputs(override):
    from models.geoclip_time.lib.temporal_retrieval import rerank_by_date

    arguments = {
        "indices": np.array([[0, 1]]), "scores": np.array([[0.9, 0.8]]),
        "query_years": np.array([1900]), "reference_years": np.array([1900, 1950]),
        "reference_gps": np.array([[46, 7], [47, 9]]), "gamma": 0.2, **override,
    }
    with pytest.raises(ValueError):
        rerank_by_date(**arguments)


def test_retrieval_years_reproduce_the_teammate_scalar_date_algorithm():
    from models.geoclip_time.scripts.rerank_references import dates_to_retrieval_years

    lower = [
        "1900-01-01", "1900-12-31", "1899-12-31T23:00:00-02:00",
        "1900-07-01T23:00:00Z", "2030-01-01", "2030-01-02", None,
        "not a date", 1900, "1750-01-01",
    ]
    upper = [
        "1900-12-31", "1900-01-01", "1900-01-01T01:00:00+00:00",
        "1900-07-01T23:00:00Z", "2030-01-01", "2030-01-02", "1900-01-01",
        "1900-01-01", 1900, "1850-01-01",
    ]
    years, known = dates_to_retrieval_years(lower, upper)
    # UTC timestamp midpoint, then year + (dayofyear - 1)/366. Hours are
    # discarded; inverse intervals are accepted; numeric inputs remain ns.
    expected = np.array([
        1900 + 182 / 366, 1900 + 182 / 366, 1900.0, 1900 + 181 / 366,
        2030.0, np.nan, np.nan, np.nan, 1970.0, 1800.0,
    ])
    np.testing.assert_allclose(years, expected, rtol=0, atol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(known, np.isfinite(expected))


@pytest.mark.parametrize("gammas,expected", [([0.0, 0.2, 0.4], 0.2), ([0.4, 0.2, 0.0], 0.4)])
def test_gamma_selection_uses_validation_error_and_keeps_first_tied_gamma(gammas, expected):
    from models.geoclip_time.scripts.rerank_references import select_gamma_on_validation

    gamma, curve = select_gamma_on_validation(
        np.array([[0, 1], [0, 1]]), np.array([[0.9, 0.8], [0.9, 0.8]]),
        np.array([1900.0, 1900.0]), np.array([1800.0, 1900.0]),
        np.array([[0.0, 0.0], [0.0, 1.0]]), np.array([[0.0, 1.0], [0.0, 1.0]]),
        np.array(["val", "val"]), gammas=gammas,
    )
    assert gamma == expected
    assert [row["gamma"] for row in curve] == gammas
    assert all(row["n"] == 2 for row in curve)
    assert next(row for row in curve if row["gamma"] == 0.0)["median_km"] > 100
    assert next(row for row in curve if row["gamma"] == 0.2)["median_km"] == pytest.approx(0.0)


@pytest.mark.parametrize("forbidden_split", ["train", "test_in", "test_city", "test_region", "nadir_val"])
def test_gamma_selection_rejects_nonvalidation_rows(forbidden_split):
    from models.geoclip_time.scripts.rerank_references import select_gamma_on_validation

    with pytest.raises(ValueError, match="val|validation"):
        select_gamma_on_validation(
            np.array([[0, 1], [0, 1]]), np.array([[0.9, 0.8], [0.9, 0.8]]),
            np.array([1900.0, 1900.0]), np.array([1800.0, 1900.0]),
            np.array([[0.0, 0.0], [0.0, 1.0]]), np.array([[0.0, 1.0], [0.0, 0.0]]),
            np.array(["val", forbidden_split]), gammas=[0.0, 0.2],
        )


def test_prediction_rows_export_the_selected_reference_id_with_its_coordinates():
    from models.geoclip_time.lib.temporal_retrieval import rerank_by_date
    from models.geoclip_time.scripts.rerank_references import build_prediction_rows

    reference = pd.DataFrame({
        "id": [101, 205, 309], "split": ["train"] * 3,
        "latitude": [46.0, 47.0, 45.0], "longitude": [7.0, 9.0, 6.0],
    }, index=[12, 44, 2])
    query = pd.DataFrame({"id": [800], "split": ["val"]}, index=[9])
    reranked = rerank_by_date(
        np.array([[2, 0, 1]]), np.array([[0.95, 0.9, 0.89]]),
        np.array([1900.0]), np.array([1900.0, 1950.0, 1960.0]),
        reference[["latitude", "longitude"]].to_numpy(), gamma=0.2,
    )
    out = build_prediction_rows(query, reference, reranked, method="date_rerank", gamma=0.2)
    assert out.id.tolist() == [800]
    assert out.neighbor_id.tolist() == [101]
    np.testing.assert_array_equal(out[["lat_pred", "lon_pred"]].to_numpy(), [[46.0, 7.0]])
    reference.loc[44, "split"] = "test_in"
    with pytest.raises(ValueError, match="train"):
        build_prediction_rows(query, reference, reranked, method="date_rerank", gamma=0.2)
