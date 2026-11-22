"""Inference-time geographic support for any point-valued geolocator.

Build GalleryFeatureExtractor once from the real training gallery. Pass WGS84
coordinates as (latitude, longitude) in degrees, one row per training photo.
At inference, transform a batch of model predictions and their query date masks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_KM = 6371.0088
SINGLE_FEATURES = ["nearest_train_km", "log1p_train_5km", "log1p_train_25km", "has_date"]


def _coordinates(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite WGS84 latitude/longitude pairs of shape (N, 2)")
    if np.any(np.abs(values) > [90, 180]):
        raise ValueError("Latitude or longitude outside WGS84 range")
    return values


def _unit_vectors(latlon):
    lat, lon = np.deg2rad(latlon).T
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


class GalleryFeatureExtractor:
    """Distances and photo counts for a fixed real training gallery on a sphere."""

    def __init__(self, train_latlon):
        coordinates = _coordinates(train_latlon)
        if len(coordinates) == 0:
            raise ValueError("Training gallery must contain at least one photo")
        self._tree = cKDTree(_unit_vectors(coordinates))

    def transform(self, predicted_latlon, has_date, *, workers=1):
        """Return one row per prediction; has_date is the query's known-date mask."""
        coordinates = _coordinates(predicted_latlon)
        known = np.asarray(has_date)
        if (known.shape != (len(coordinates),) or known.dtype.kind not in "biuf"
                or not np.isin(known, [0, 1]).all()):
            raise ValueError("Pass one boolean or 0/1 date flag per predicted coordinate")
        unique, inverse = np.unique(coordinates, axis=0, return_inverse=True)
        locations = _unit_vectors(unique)
        nearest, _ = self._tree.query(locations, k=1, workers=workers)
        nearest_km = 2 * EARTH_KM * np.arcsin(np.clip(nearest / 2, 0, 1))
        # Euclidean chord thresholds on the unit sphere preserve spherical-radius
        # membership. The gallery keeps multiple photos at identical coordinates.
        features = {"nearest_train_km": nearest_km}
        for radius in (5, 25):
            chord = 2 * np.sin(radius / (2 * EARTH_KM))
            counts = self._tree.query_ball_point(locations, r=chord, return_length=True, workers=workers)
            features[f"log1p_train_{radius}km"] = np.log1p(counts)
        mapped = pd.DataFrame(features).iloc[inverse.ravel()].reset_index(drop=True)
        mapped["has_date"] = known.astype(bool)
        return mapped
