"""Small tests for the T4 linear map and LV95 polygon projection."""
import importlib

import numpy as np
import pytest
import torch
from shapely.geometry import MultiPolygon, Polygon, box


def regression():
    name = "models.clip_regression.lib.regression"
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, "T4 regression module is not implemented"
    return importlib.import_module(name)


def test_single_linear_layer_uses_clip_then_three_date_values_without_activation():
    model = regression().LinearTimeRegressor(feature_dim=2)
    assert [(name, tuple(parameter.shape)) for name, parameter in model.named_parameters()] == [
        ("linear.weight", (2, 5)), ("linear.bias", (2,))]
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1., 2., 3., 4., 5.], [-1., 0., 0., 0., -2.]]))
        model.linear.bias.copy_(torch.tensor([.5, -3.]))
    clip = torch.tensor([[2., 1.], [0., -1.]], requires_grad=True)
    dates = torch.tensor([[.5, .1, 1.], [0., 0., 0.]], requires_grad=True)
    actual = model(clip, dates)
    torch.testing.assert_close(actual, torch.tensor([[11.4, -7.], [-1.5, -3.]]))
    actual.sum().backward()
    torch.testing.assert_close(dates.grad, torch.tensor([[3., 4., 3.], [3., 4., 3.]]))
    assert regression().LinearTimeRegressor().linear.in_features == 771


@pytest.mark.parametrize("clip_shape,date_shape", [((2, 3), (2, 3)), ((2, 2), (2, 2)), ((2, 2), (1, 3)), ((2,), (1, 3))])
def test_linear_model_rejects_misaligned_feature_shapes(clip_shape, date_shape):
    model = regression().LinearTimeRegressor(feature_dim=2)
    with pytest.raises(ValueError):
        model(torch.zeros(clip_shape), torch.zeros(date_shape))


def test_polygon_projection_preserves_inside_and_boundary_and_projects_outside():
    polygon = box(2600000., 1200000., 2600100., 1200100.)
    xy = np.array([[2600020., 1200030.], [2600000., 1200050.],
                   [2600120., 1200040.], [2599990., 1199990.]])
    original = xy.copy()
    actual = regression().clip_to_switzerland(xy, polygon)
    np.testing.assert_array_equal(actual, [[2600020., 1200030.], [2600000., 1200050.],
                                           [2600100., 1200040.], [2600000., 1200000.]])
    np.testing.assert_array_equal(xy, original)
    assert actual.dtype == np.float64 and not np.shares_memory(actual, xy)


def test_polygon_projection_handles_holes_and_disconnected_territory():
    land_with_hole = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)],
                             holes=[[(3, 3), (7, 3), (7, 7), (3, 7)]])
    country = MultiPolygon([land_with_hole, box(20, 0, 25, 5)])
    actual = regression().clip_to_switzerland(np.array([[4., 5.], [19., 2.]]), country)
    np.testing.assert_array_equal(actual, [[3., 5.], [20., 2.]])
    assert regression().clip_to_switzerland(np.empty((0, 2)), country).shape == (0, 2)


@pytest.mark.parametrize("xy", [np.array([1., 2.]), np.ones((2, 3)), np.array([[np.nan, 2.]]), np.array([[np.inf, 2.]])])
def test_polygon_projection_rejects_invalid_coordinates(xy):
    with pytest.raises(ValueError):
        regression().clip_to_switzerland(xy, box(0, 0, 1, 1))


@pytest.mark.parametrize("polygon", [Polygon(), Polygon([(0, 0), (1, 1), (0, 1), (1, 0)]), None])
def test_polygon_projection_rejects_empty_invalid_or_missing_boundary(polygon):
    with pytest.raises(ValueError):
        regression().clip_to_switzerland(np.zeros((1, 2)), polygon)
