"""T4: one linear map from cached CLIP plus date features to two coordinates."""
from numbers import Integral

import numpy as np
import torch
from torch import nn


class LinearTimeRegressor(nn.Module):
    """Predict two normalized LV95 coordinates without hidden layers.

    The caller supplies CLIP features (optionally standardized using training
    statistics) and T1's normalized midpoint, half-width, and known-date mask.
    Coordinate normalization and conversion back to meters belong to the caller.
    """

    def __init__(self, feature_dim=768):
        super().__init__()
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, Integral) or feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer")
        self.feature_dim = int(feature_dim)
        self.time_dim = 3
        self.linear = nn.Linear(self.feature_dim + self.time_dim, 2)

    def forward(self, clip_features, date_features):
        if not isinstance(clip_features, torch.Tensor) or clip_features.ndim != 2 or clip_features.shape[1] != self.feature_dim:
            raise ValueError(f"clip_features must be a tensor with shape [N,{self.feature_dim}]")
        if not isinstance(date_features, torch.Tensor) or date_features.shape != (len(clip_features), self.time_dim):
            raise ValueError("date_features must be a tensor with shape [N,3] aligned with CLIP rows")
        return self.linear(torch.cat((clip_features, date_features), dim=1))


def clip_to_switzerland(xy_m, polygon):
    """Project outside points to the nearest Swiss boundary in LV95 meters.

    ``polygon`` is a valid, nonempty Shapely Polygon or MultiPolygon already in
    the same LV95 CRS as ``xy_m``. Inside and boundary points stay unchanged;
    points in holes are outside. Return a separate float64 array of shape [N,2].
    This function does not infer a CRS, convert longitude/latitude, or fetch a
    country outline. Call it after denormalizing the model's coordinates.
    """
    from shapely.geometry import MultiPolygon, Point, Polygon
    from shapely.ops import nearest_points
    from shapely.prepared import prep

    if not isinstance(polygon, (Polygon, MultiPolygon)) or polygon.is_empty or not polygon.is_valid:
        raise ValueError("polygon must be a valid, nonempty Polygon or MultiPolygon in LV95 meters")
    if np.iscomplexobj(xy_m):
        raise ValueError("xy_m must contain real coordinates")
    xy = np.array(xy_m, dtype=np.float64, copy=True)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("xy_m must be a finite array with shape [N,2]")
    prepared, boundary = prep(polygon), polygon.boundary
    for index, (east, north) in enumerate(xy):
        point = Point(east, north)
        if not prepared.covers(point):
            nearest = nearest_points(boundary, point)[0]
            xy[index] = nearest.x, nearest.y
    return xy
