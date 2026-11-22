"""Estimate P(location error <= radius) for a single, fixed upstream predictor."""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit


class GeoConfidence:
    """One feature schema and one probability model per locator version and radius.

    Feature columns are chosen by the caller and must be available for a new
    photo. Extra columns are ignored. All fit inputs are aligned by row position.
    Geographic groups are used only to split fitting from sigmoid calibration.
    """

    def __init__(self, feature_names: Sequence[str], threshold_km: float = 25., random_state: int = 42):
        if isinstance(feature_names, str):
            raise ValueError("Pass a sequence of feature names, not a string")
        names = tuple(feature_names)
        if not names or not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names):
            raise ValueError("Feature names must be unique, nonempty strings")
        if not np.isfinite(threshold_km) or threshold_km <= 0:
            raise ValueError("threshold_km must be positive and finite")
        self.feature_names = names
        self.threshold_km = float(threshold_km)
        self.random_state = random_state

    def _features(self, frame: pd.DataFrame) -> np.ndarray:
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
            raise ValueError("Expected a DataFrame with unique column names")
        missing = [name for name in self.feature_names if name not in frame]
        if missing:
            raise ValueError(f"Missing confidence features: {missing}")
        matrix = frame.loc[:, list(self.feature_names)].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(matrix).any():
            raise ValueError("Features may contain NaN but not infinity")
        return matrix

    @staticmethod
    def _ids(values, n: int, name: str) -> np.ndarray:
        ids = np.asarray(values)
        if ids.shape != (n,) or pd.isna(ids).any():
            raise ValueError(f"{name} must contain one nonmissing ID per row")
        return ids.astype(str)

    def fit(self, features: pd.DataFrame, errors_km, *, groups, query_ids=None) -> GeoConfidence:
        """Use 70% of unique blocks for the classifier and 30% for calibration."""
        x = self._features(features)
        errors = np.asarray(errors_km, dtype=float)
        if errors.shape != (len(x),) or not np.isfinite(errors).all() or (errors < 0).any():
            raise ValueError("Provide one finite, nonnegative error in km per row")
        block = self._ids(groups, len(x), "groups")
        if query_ids is not None:
            ids = self._ids(query_ids, len(x), "query_ids")
            pairs = pd.DataFrame({"id": ids, "block": block}).drop_duplicates()
            if pairs.id.duplicated().any():
                raise ValueError("A query ID belongs to multiple geographic blocks")
        blocks = np.unique(block)
        if len(blocks) < 2:
            raise ValueError("Fitting and calibration require at least two distinct geographic blocks")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=self.random_state)
        train_idx, cal_idx = next(splitter.split(blocks, groups=blocks))
        train = np.isin(block, blocks[train_idx])
        y = (errors <= self.threshold_km).astype(int)
        y_cal = y[~train]
        base = sigmoid = None
        # A fold with a single outcome cannot support a fitted sigmoid.
        fallback = (y_cal.sum() + 1.) / (len(y_cal) + 2.)
        if np.unique(y[train]).size == 2 and np.unique(y_cal).size == 2:
            base = HistGradientBoostingClassifier(
                max_iter=80, max_leaf_nodes=7, min_samples_leaf=25, learning_rate=0.08,
                early_stopping=False, random_state=self.random_state,
            ).fit(x[train], y[train])
            raw = base.decision_function(x[~train]).reshape(-1, 1)
            sigmoid = LogisticRegression(random_state=self.random_state).fit(raw, y_cal)

        self.fit_groups_ = tuple(blocks[train_idx].tolist())
        self.calibration_groups_ = tuple(blocks[cal_idx].tolist())
        self._base = base
        self._sigmoid = sigmoid
        self._fallback = float(fallback)
        return self

    def _check_fitted(self):
        if not hasattr(self, "calibration_groups_"):
            raise NotFittedError("Fit or load the confidence estimator first")

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return one estimated hit probability per row in the input order."""
        self._check_fitted()
        x = self._features(features)
        if not len(x) or self._base is None:
            return np.full(len(x), self._fallback)
        raw = self._base.decision_function(x).reshape(-1, 1)
        return self._sigmoid.predict_proba(raw)[:, 1]

    def save(self, path: str | Path) -> None:
        """Atomically replace a joblib file, including the feature schema and radius."""
        self._check_fitted()
        path = Path(path)
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
            temp = Path(f.name)
        try:
            joblib.dump({"format": 1, "model": self}, temp, compress=3)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> GeoConfidence:
        """Load a trusted model file produced by save()."""
        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("format") != 1 or not isinstance(payload.get("model"), cls):
            raise ValueError("Unrecognized GeoConfidence artifact")
        model = payload["model"]
        model._check_fitted()
        return model
