"""Leave-one-development-region-out learning curves for the frozen T7 expert inputs."""
from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from models.t6_router.coverage_gate import _signature
from models.t6_router.expanded_regions import SEVEN
from models.t7_error_regression import model


def _settings(stages, fractions, seed):
    stages, fractions = tuple(stages), tuple(fractions)
    if (not stages or any(isinstance(s, bool) or not isinstance(s, Integral) or s < 1 for s in stages)
            or len(stages) != len(set(stages))):
        raise ValueError("Stages must be distinct positive integers")
    if (not fractions or any(isinstance(f, bool) or not isinstance(f, Real) or not 0 < f <= 1 for f in fractions)
            or len(fractions) != len(set(fractions))):
        raise ValueError("Fractions must be distinct numbers in (0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("Sampling seed must be a nonnegative integer")
    return sorted(map(int, stages)), sorted(map(float, fractions))


def _metrics(train_n, train_sse, observed, predicted):
    residual = predicted - observed
    squared = residual**2
    t4_wins = predicted[:, 0] < predicted[:, 1]
    e4, e5 = observed.T
    chosen = np.where(t4_wins, e4, e5)
    unequal = e4 != e5
    non_ties = int(unequal.sum())
    correct = int(np.count_nonzero(unequal & (t4_wins == (e4 < e5))))
    return {
        "train_n": train_n, "held_n": len(observed), "non_ties": non_ties, "closer_correct": correct,
        "train_joint_rmse_km": float(np.sqrt(train_sse / (2 * train_n))),
        "held_joint_rmse_km": float(np.sqrt(squared.mean())),
        "held_T4_rmse_km": float(np.sqrt(squared[:, 0].mean())), "held_T5_rmse_km": float(np.sqrt(squared[:, 1].mean())),
        "held_T4_mae_km": float(np.abs(residual[:, 0]).mean()), "held_T5_mae_km": float(np.abs(residual[:, 1]).mean()),
        "chosen_mean_km": float(chosen.mean()), "chosen_median_km": float(np.median(chosen)),
        "closer_accuracy": correct / non_ties if non_ties else None,
    }


def _staged_predictions(features, observed, fit_ids, held_ids, stages):
    parameters = {**model.REGRESSOR_PARAMS, "max_iter": stages[-1], "early_stopping": False}
    source = features.iloc[fit_ids]
    both = features.iloc[np.concatenate((fit_ids, held_ids))]
    generators = []
    for target in observed.T:
        estimator = HistGradientBoostingRegressor(**parameters).fit(source, target[fit_ids])
        generators.append(estimator.staged_predict(both))
    requested = set(stages)
    for stage, pair in enumerate(zip(*generators, strict=True), 1):
        if stage in requested:
            estimates = np.column_stack(pair)
            if not np.isfinite(estimates).all():
                raise ValueError("T7 stage produced a nonfinite expected distance")
            np.maximum(estimates, 0., out=estimates)
            yield stage, estimates[:len(source)], estimates[len(source):]


def _macro(by_region):
    counts = {"train_n", "held_n", "closer_correct", "non_ties"}
    result = {}
    for metric in next(iter(by_region.values())):
        if metric in counts:
            continue
        values = [row[metric] for row in by_region.values() if row[metric] is not None]
        result[metric] = float(np.mean(values)) if values else None
    return result


def evaluate_curve(train, *, stages=(20, 40, 80, 160, 320), fractions=(0.5, 1.0), seed=42):
    """Fit both error regressors to six regions, scoring the seventh at fixed tree counts.

    Each source region contributes max(1, floor(fraction * n)) IDs from a seeded
    permutation reused across fractions and folds. Pooled training RMSE weights
    every in-fold training exposure; each photo contributes once to held metrics.
    """
    stages, fractions = _settings(stages, fractions, seed)
    if not train.columns.is_unique or not {"id", "region"}.issubset(train):
        raise ValueError("Development rows need distinct columns with photo IDs and region labels")
    if (not pd.api.types.is_integer_dtype(train.id) or train.id.isna().any() or train.id.duplicated().any()
            or train.id.le(0).any()):
        raise ValueError("Expected unique positive integer photo IDs")
    if train.region.isna().any() or set(train.region) != set(SEVEN):
        raise ValueError("Expected the seven predeclared development regions")
    if "block" in train and (train.block.isna().any() or train.groupby("block").region.nunique().gt(1).any()):
        raise ValueError("A geographic block must belong to a single development region")

    rows = train.sort_values("id").reset_index(drop=True)
    features, observed = model._inputs(rows), model.expert_errors(rows)
    region_rows = {region: np.flatnonzero(rows.region.eq(region)) for region in SEVEN}
    draws = {region: np.random.default_rng(np.random.SeedSequence([seed, i])).permutation(indices)
             for i, (region, indices) in enumerate(region_rows.items())}
    folds = {region: {"held_n": len(indices), "train_by_fraction": {}} for region, indices in region_rows.items()}
    points = []
    with threadpool_limits(limits=4):
        for fraction in fractions:
            sampled = {name: indices[:max(1, int(len(indices) * fraction))] for name, indices in draws.items()}
            out_of_fold = {s: np.empty_like(observed) for s in stages}
            by_stage = {s: {} for s in stages}
            pooled_train_sse = dict.fromkeys(stages, 0.)
            pooled_train_n = 0
            for region, held_ids in region_rows.items():
                sources = {name: ids for name, ids in sampled.items() if name != region}
                fit_ids = np.sort(np.concatenate(list(sources.values())))
                n = len(fit_ids)
                pooled_train_n += n
                folds[region]["train_by_fraction"][str(fraction)] = {
                    "train_n": n, "photos_by_region": {name: len(ids) for name, ids in sources.items()},
                    "sample_id_sha256": _signature(rows.iloc[fit_ids], ["id"]),
                }
                for s, training, held in _staged_predictions(features, observed, fit_ids, held_ids, stages):
                    sse = float(np.square(training - observed[fit_ids]).sum())
                    pooled_train_sse[s] += sse
                    by_stage[s][region] = _metrics(n, sse, observed[held_ids], held)
                    out_of_fold[s][held_ids] = held
            for s in stages:
                regions = by_stage[s]
                if len(regions) != len(SEVEN):
                    raise RuntimeError("Missing staged predictions for a development region")
                points.append({"fraction": fraction, "stage": s, "by_region": regions, "macro": _macro(regions),
                               "pooled": _metrics(pooled_train_n, pooled_train_sse[s], observed, out_of_fold[s])})
    return {
        "n": len(rows), "seed": int(seed), "stages": stages, "fractions": fractions, "folds": folds, "points": points,
        "protocol": {
            "sampling": "Within each source region use the same seeded ID permutation prefix, taking "
                        "max(1, floor(fraction * n)). Both regressors use the same six source regions and photos.",
            "joint_rmse": "Square root of the mean squared forecast error over both experts and all photos, "
                          "after clipping expected distances to zero from below.",
            "pooling": "Pooled training metrics count each in-fold training exposure. Held photos enter once. "
                       "Macro averages the seven region metrics equally, excluding undefined closer accuracies.",
            "routing": "Choose the smaller expected distance; predicted ties choose T5. Closer accuracy is "
                       "a fraction excluding exact observed-error ties. Geographic distance metrics include all photos.",
        },
    }
