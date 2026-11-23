"""Choose the frozen locator with the smaller estimated mean geodesic error."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from models.clip_regression.scripts.compare_t1 import _errors as geodesic_km, _gps
from models.t6_router.geographic_gate import PARAMETERS, POSITION, _errors as expert_errors
from scripts.confidence.evaluate import FEATURES


DIAGNOSTICS = (*FEATURES["T4"], *FEATURES["T5"])
INPUT_COLUMNS = (*POSITION, *DIAGNOSTICS)
FEATURE_COLUMNS = (*INPUT_COLUMNS, "expert_gap_km")
REGRESSOR_PARAMS = {**PARAMETERS, "loss": "squared_error"}


@dataclass
class ExpectedErrorModel:
    t4: HistGradientBoostingRegressor
    t5: HistGradientBoostingRegressor


def _inputs(frame):
    if not frame.columns.is_unique:
        raise ValueError("T7 needs unique column names")
    missing = set(INPUT_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing T7 inputs: {sorted(missing)}")
    gps4 = _gps(frame, POSITION[:2], "T4")
    gps5 = _gps(frame, POSITION[2:], "T5")
    diagnostics = frame.loc[:, list(DIAGNOSTICS)].to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(diagnostics).any():
        raise ValueError("Expert diagnostics may be missing but cannot be infinite")
    matrix = np.column_stack((gps4, gps5, diagnostics, geodesic_km(gps4, gps5)))
    return pd.DataFrame(matrix, index=frame.index, columns=FEATURE_COLUMNS)


def fit(train_df):
    """Fit one squared-error regression in kilometres per expert on development rows."""
    if train_df.empty:
        raise ValueError("T7 needs development photos for model fitting")
    x = _inputs(train_df)
    e4, e5 = expert_errors(train_df).T
    with threadpool_limits(limits=4):
        return ExpectedErrorModel(HistGradientBoostingRegressor(**REGRESSOR_PARAMS).fit(x, e4),
                                  HistGradientBoostingRegressor(**REGRESSOR_PARAMS).fit(x, e5))


def predict(model, df):
    """Return the two estimated errors and a hard choice, with exact ties going to T5."""
    x = _inputs(df)
    with threadpool_limits(limits=4):
        values = np.column_stack((model.t4.predict(x), model.t5.predict(x))) if len(x) else np.empty((0, 2))
    if not np.isfinite(values).all():
        raise ValueError("T7 must predict finite distances")
    # The squared-error regressor can extrapolate slightly below the physical lower bound.
    e4, e5 = np.maximum(values, 0.).T
    return pd.DataFrame({"T4_expected_km": e4, "T5_expected_km": e5, "T7_choose_T4": e4 < e5},
                        index=df.index)
