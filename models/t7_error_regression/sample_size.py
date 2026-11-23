"""Compare matched and full T7 training samples on identical held-out development photos."""
import numpy as np
import pandas as pd

from models.t6_router.coverage_gate import ERRORS, _signature
from models.t6_router.expanded_regions import SEVEN
from models.t6_router.regional_closer import distance_metrics, paired_distances
from models.t7_error_regression import model


ARMS = ("matched", "full")
EXPECTED = ["T4_expected_km", "T5_expected_km"]
TRAIN_COLUMNS = [*model.INPUT_COLUMNS, *ERRORS]
MEASURES = ("chosen_mean_km", "chosen_median_km", "closer_accuracy", "forecast_joint_rmse_km",
            "forecast_T4_rmse_km", "forecast_T5_rmse_km")


def _ordered(rows, columns):
    if not rows.columns.is_unique or not {"id", *columns}.issubset(rows.columns):
        raise ValueError("Missing or ambiguous region, frozen expert input or error columns")
    if (not pd.api.types.is_integer_dtype(rows.id) or rows.id.isna().any() or rows.id.duplicated().any()
            or rows.id.le(0).any()):
        raise ValueError("Expected unique positive integer photo IDs")
    if rows.region.isna().any() or set(rows.region) != set(SEVEN):
        raise ValueError("Only the seven development regions are eligible")
    if "block" in columns and (rows.block.isna().any() or rows.groupby("block").region.nunique().gt(1).any()):
        raise ValueError("A geographic block must belong to one region")
    return rows.set_index("id")[columns].sort_index()


def _summarize(rows, predictions):
    observed = model.expert_errors(rows)
    summary = {"n": len(rows)}
    choices = {}
    for arm in ARMS:
        pred = predictions[arm].loc[rows.index]
        choices[arm] = pred.T7_choose_T4.to_numpy()
        metrics = distance_metrics(rows, choices[arm])
        n, correct = metrics["n_non_ties"], metrics["correct_non_ties"]
        mse = ((observed - pred[EXPECTED].to_numpy())**2).mean(axis=0)
        summary[arm] = {
            "chosen_mean_km": metrics["mean_km"], "chosen_median_km": metrics["median_km"],
            "non_ties": n, "closer_correct": correct, "closer_accuracy": correct / n if n else None,
            "choose_T4": metrics["choose_T4"], "forecast_joint_rmse_km": float(np.sqrt(mse.mean())),
            "forecast_T4_rmse_km": float(np.sqrt(mse[0])), "forecast_T5_rmse_km": float(np.sqrt(mse[1])),
        }
    delta = paired_distances(rows, choices["full"], choices["matched"])
    summary["paired"] = {
        "closer": delta["improved"], "farther": delta["worsened"],
        "unchanged": len(rows) - delta["improved"] - delta["worsened"], "switched": delta["switches"],
        "net_closer": delta["net_correct_non_ties"], "total_saved_km": delta["net_saved_km"],
        "mean_saved_km": delta["mean_saved_km"],
    }
    return summary


def _macro(regions):
    result = {"n_regions": len(regions)}
    for arm in ARMS:
        result[arm] = {}
        for key in MEASURES:
            values = [row[arm][key] for row in regions.values() if row[arm][key] is not None]
            result[arm][key] = float(np.mean(values)) if values else None
    result["paired"] = {key + "_fraction": float(np.mean([row["paired"][key] / row["n"] for row in regions.values()]))
                        for key in ("closer", "farther", "unchanged", "switched")}
    result["paired"]["mean_saved_km"] = float(np.mean([row["paired"]["mean_saved_km"] for row in regions.values()]))
    return result


def evaluate_sample_size(matched, full):
    """Train 80-stage T7 independently on nested sources and score every full-region holdout."""
    if model.REGRESSOR_PARAMS["max_iter"] != 80 or model.REGRESSOR_PARAMS["early_stopping"]:
        raise ValueError("This comparison requires the fixed 80-stage T7")
    columns = ["region", *TRAIN_COLUMNS]
    if "block" in matched or "block" in full:
        columns.append("block")
    matched, full = _ordered(matched, columns), _ordered(full, columns)
    if not matched.index.isin(full.index).all():
        raise ValueError("Every matched photo ID must be present in the full sample")
    try:
        pd.testing.assert_frame_equal(matched, full.loc[matched.index], check_exact=True, check_dtype=False)
    except AssertionError as exc:
        raise ValueError("A shared photo has different geography, inputs or targets") from exc

    folds = {}
    predictions = {arm: [] for arm in ARMS}
    for region in SEVEN:
        held = full.loc[full.region.eq(region), list(model.INPUT_COLUMNS)]
        sources = {"matched": matched.loc[matched.region.ne(region)], "full": full.loc[full.region.ne(region)]}
        info = {"held_n": len(held), "matched_held_n": int(matched.region.eq(region).sum()),
                "same_training_ids": sources["matched"].index.equals(sources["full"].index)}
        for arm, source in sources.items():
            estimator = model.fit(source[TRAIN_COLUMNS])
            expected = model.predict(estimator, held)
            if not expected.index.equals(held.index):
                raise ValueError("T7 output is misaligned with the held-out photos")
            predictions[arm].append(expected)
            info[f"{arm}_train_n"] = len(source)
            info[f"{arm}_train_ids_sha256"] = _signature(source.index.to_frame(index=False), ["id"])
        a, b = (predictions[arm][-1] for arm in ARMS)
        info["identical_forecasts"] = np.array_equal(a[EXPECTED], b[EXPECTED])
        info["identical_choices"] = a.T7_choose_T4.equals(b.T7_choose_T4)
        if info["same_training_ids"] and not (info["identical_forecasts"] and info["identical_choices"]):
            raise RuntimeError(f"{region}: identical training rows must reproduce forecasts and decisions")
        folds[region] = info

    predictions = {arm: pd.concat(pieces, verify_integrity=True) for arm, pieces in predictions.items()}
    by_region = {name: _summarize(full.loc[full.region.eq(name)], predictions) for name in SEVEN}
    affected = [name for name in SEVEN if not folds[name]["same_training_ids"]]
    return {
        "matched_n": len(matched), "full_n": len(full), "parameters": dict(model.REGRESSOR_PARAMS),
        "folds": folds, "by_region": by_region, "pooled": _summarize(full, predictions), "macro": _macro(by_region),
        "matched_evaluation": _summarize(matched, predictions), "affected_regions": affected,
        "affected_evaluation": _summarize(full.loc[full.region.isin(affected)], predictions) if affected else None,
        "protocol": "Both arms score every photo in the same full held-out region after fitting on their "
                    "other six regions. Paired counts compare observed errors per photo. Positive saved km "
                    "means the full arm is closer. Closer accuracy excludes observed error ties. Joint RMSE "
                    "weights both expert error forecasts equally. Macro metrics weight regions equally, "
                    "omitting undefined accuracies. Matched evaluation reuses the same forecasts on matched IDs.",
    }
