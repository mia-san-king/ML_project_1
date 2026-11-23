"""Run T7's learning curve on its frozen seven-region development sample."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from models.t6_router.coverage_gate import SEVEN, USED, _signature
from models.t6_router.expanded_regions import SEVEN as DEV_REGIONS, matched_samples
from models.t6_router.run_coverage_study import read_fold
from scripts.confidence.evaluate import sha256


REPORT_DIR = Path(__file__).resolve().parents[2] / "report/experiments/2026-11-22-local-models"
T6_REPORT = REPORT_DIR / "t6/matched_region_coverage.json"
T7_REPORT = REPORT_DIR / "t7/error_regression.json"


def load_development(study_root):
    """Check each development fold and reproduce the photo IDs and inputs used to fit T7."""
    t6 = json.loads(T6_REPORT.read_text(encoding="utf-8"))
    t7 = json.loads(T7_REPORT.read_text(encoding="utf-8"))
    frames, sources = [], {}
    for name in DEV_REGIONS:
        split = study_root / "splits" / f"{name}.csv.gz"
        frame, receipt = read_fold(study_root, name, split)
        if receipt != t6["folds"][name]:
            raise ValueError(f"{name}: expert receipt differs from the archived development fold")
        frames.append(frame)
        sources[name] = {"split_sha256": receipt["inputs"]["split_sha256"],
                         "predictions_sha256": receipt["predictions_sha256"]}
    budget = t6["comparison"]["development_n_per_arm"]
    train = matched_samples(pd.concat(frames, ignore_index=True), budget=budget, seed=t6["sampling_seed"])["seven"]
    archived = t6["comparison"]["training"][SEVEN]
    ids = _signature(train, ["id"])
    if len(train) != archived["gate_train_n"] or ids != archived["gate_train_ids_sha256"]:
        raise ValueError("Development IDs differ from the archived seven-region sample")
    signature = _signature(train, USED)
    if signature != archived["sample_sha256"] or signature != t7["inputs"]["training_sample_sha256"]:
        raise ValueError("Development data signature differs from the original T7 training sample")
    counts = train.region.value_counts().sort_index().to_dict()
    if counts != t7["training"]["photos_by_region"] or len(train) != t7["training"]["n"]:
        raise ValueError("Development photo counts differ from T7 training")
    provenance = {"training_n": len(train), "photos_by_region": counts, "training_ids_sha256": ids,
                  "training_sample_sha256": signature, "sampling_seed": t6["sampling_seed"], "folds": sources,
                  "t6_report_sha256": sha256(T6_REPORT), "t7_report_sha256": sha256(T7_REPORT)}
    return train, provenance


def run(study_root, output):
    from models.t7_error_regression.learning_curve import evaluate_curve

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new learning-curve JSON file: {output}")
    train, provenance = load_development(Path(study_root))
    print(f"Verified {len(train):,} T7 development photos across {train.region.nunique()} regions", flush=True)
    result = {"training_n": len(train), "provenance": provenance, "curve": evaluate_curve(train)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        draft = Path(temporary) / "curve.json"
        draft.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Linking the complete file fails atomically if another run published this output first.
        output.hardlink_to(draft)
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--study-root", type=Path, required=True, help="Existing T6 job/out with dev1 through dev7")
    cli.add_argument("--output", type=Path, required=True, help="New learning-curve JSON file")
    args = cli.parse_args()
    run(**vars(args))
    print("COMPLETE", args.output, flush=True)
