"""Compare T7's original training sample with every photo in the same seven development regions."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from models.t6_router.coverage_gate import USED, _signature
from models.t6_router.expanded_regions import SEVEN
from models.t6_router.run_coverage_study import read_fold
from models.t7_error_regression.run_learning_curve import T6_REPORT, load_development


EXPECTED_SIZES = (46_235, 59_196, 12_961)


def load_samples(study_root):
    """Recover all seven folds from the verified original sample and its omitted dev1 photos."""
    study_root = Path(study_root)
    matched, source = load_development(study_root)
    folds = json.loads(T6_REPORT.read_text(encoding="utf-8"))["folds"]
    sizes = {name: folds[name]["query_count"] for name in SEVEN}
    dev1, receipt = read_fold(study_root, "dev1", study_root / "splits/dev1.csv.gz")
    if receipt != folds["dev1"]:
        raise ValueError("The full dev1 expert receipt differs from the archived fold")
    old = matched.loc[matched.region.eq("dev1")]
    retained = dev1.id.isin(old.id)
    if _signature(dev1.loc[retained], USED) != _signature(old, USED):
        raise ValueError("The retained dev1 photos differ from the original T7 sample")
    added = dev1.loc[~retained]
    if not added.region.eq("dev1").all():
        raise ValueError("Only dev1 may contribute additional photos")
    full = pd.concat((matched, added), ignore_index=True).sort_values("id").reset_index(drop=True)
    if (not pd.api.types.is_integer_dtype(full.id) or full.id.isna().any()
            or full.id.le(0).any() or full.id.duplicated().any()):
        raise ValueError("Every full-development row must have a unique positive real-photo ID")
    counts = full.region.value_counts().sort_index().to_dict()
    if counts != sizes or (len(matched), len(full), len(added)) != EXPECTED_SIZES:
        raise ValueError("Training sample sizes differ from the expected complete seven development folds")
    provenance = {"matched": source,
                  "full": {"n": len(full), "photos_by_region": counts, "ids_sha256": _signature(full, ["id"]),
                           "sample_sha256": _signature(full, USED)},
                  "added": {"n": len(added), "photos_by_region": {"dev1": len(added)},
                            "ids_sha256": _signature(added, ["id"]), "sample_sha256": _signature(added, USED)}}
    return matched, full, provenance


def run(study_root, output):
    from models.t7_error_regression.sample_size import evaluate_sample_size

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new sample-size comparison JSON file: {output}")
    matched, full, provenance = load_samples(study_root)
    print(f"Verified {len(matched):,} original and {len(full):,} full development photos", flush=True)
    report = {"provenance": provenance, "comparison": evaluate_sample_size(matched, full)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        draft = Path(temp) / "comparison.json"
        draft.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        output.hardlink_to(draft)
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--study-root", type=Path, required=True, help="Existing T6 job/out with dev1 through dev7")
    cli.add_argument("--output", type=Path, required=True, help="New comparison JSON file")
    args = cli.parse_args()
    run(**vars(args))
    print("COMPLETE", args.output, flush=True)
