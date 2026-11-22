"""Resolve the private data volume and baseline runs used by the historical experiments."""
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
_windows_volume = Path("D:/CS433-T1")
_default_volume = _windows_volume if os.name == "nt" and _windows_volume.is_dir() else ROOT / "data/experiment_assets"
VOLUME = Path(os.environ.get("CS433_EXPERIMENT_ASSETS", _default_volume)).expanduser().resolve()
PUBLIC = VOLUME / "public-assets"
T1_ORIGINAL = Path(os.environ.get("CS433_ORIGINAL_T1_RUN", ROOT / "data/local_runs/t1_full_20261122T111311_449581Z"))
T1_AUGMENTED = Path(os.environ.get("CS433_AUGMENTED_T1_RUN", VOLUME / "aug8192/t1_runs/aug_20261122T140802_053796Z"))
