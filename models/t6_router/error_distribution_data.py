"""Load paired +8,192 T1/T3 predictions and learned-T6 inputs on the final test set."""
from pathlib import Path

import pandas as pd

from models.t5_reranker.evaluate import GPS, original_rows
from models.t6_router.evaluate import TEST_SPLITS, paired_predictions
from models.t6_router.validate_spatial import ARCHIVE
from scripts.confidence.evaluate import EXPECTED, sha256


def load_test_comparison(path, *, workspace=None):
    """Return one ID-sorted row per original test photo with coordinates and errors.

    T1 is the 1 km grid prediction; T3 is the year-reranked real reference.
    T4/T5 probabilities are the archived deployment inputs of the frozen gate.
    All four error columns are recomputed from the same SMAPshot coordinates.
    """
    path = Path(path)
    workspace = Path(workspace) if workspace is not None else path.resolve().parents[3]
    root = workspace / "data/t1_original"
    split = workspace / "data/splits/main.csv.gz"
    files = {
        "data/local_runs/confidence_models/predictions.parquet": path,
        "data/splits/main.csv.gz": split,
        "data/t1_original/meta/smapshot.parquet": root / "meta/smapshot.parquet",
        "data/t1_original/features/clip_vitl14_ids.npy": root / "features/clip_vitl14_ids.npy",
    }
    hashes = pd.read_csv(ARCHIVE, sep="\t").query("root == 'workspace'").set_index("path").sha256
    for name, location in files.items():
        if sha256(location) != hashes.at[name]:
            raise ValueError(f"Input differs from the frozen experiment: {name}")
    truth, _, _ = original_rows(root, split)
    out = paired_predictions(path, truth, models=("T1", "T3", "T4", "T5"))
    if out.split.value_counts().to_dict() != {name: EXPECTED[name] for name in TEST_SPLITS}:
        raise ValueError("Incomplete coverage of the three final test splits")
    return out.join(truth[list(GPS)], validate="one_to_one").reset_index()
