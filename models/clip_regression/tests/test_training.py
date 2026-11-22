import numpy as np
import torch

from models.clip_regression.scripts.train import fit_linear


def test_least_squares_recovers_an_affine_map_from_train_rows_only():
    rng = np.random.default_rng(1)
    clip = rng.normal(size=(18, 3)).astype(np.float32)
    times = rng.normal(size=(18, 3)).astype(np.float32)
    coefficients = rng.normal(size=(6, 2))
    coordinates = np.column_stack((clip, times)) @ coefficients + [12., -8.]
    train_rows = np.arange(0, len(clip), 2)
    model, rank = fit_linear(clip, times, train_rows, coordinates[train_rows])
    assert rank == 7
    with torch.no_grad():
        predicted = model(torch.from_numpy(clip).double(), torch.from_numpy(times).double())
    np.testing.assert_allclose(predicted.numpy(), coordinates, rtol=1e-10, atol=1e-10)
