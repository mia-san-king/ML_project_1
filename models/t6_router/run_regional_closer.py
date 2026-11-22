"""Compare closer-expert routing using the predictions from a completed regional study."""
import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.confidence.evaluate import sha256


def load_expert_rows(study_dir):
    report = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))
    frames = {}
    for region in ("dev1", "dev2", "dev3", "final"):
        record = report["inputs"]["folds"][region]
        path = study_dir / region / "experts" / "predictions.parquet"
        if sha256(path) != record["predictions_sha256"]:
            raise ValueError(f"{region}: expert predictions differ from the regional study")
        frame = pd.read_parquet(path)
        if len(frame) != record["query_photos"] or not frame.region.eq(region).all():
            raise ValueError(f"{region}: unexpected query count or region assignment")
        frames[region] = frame
    final = frames.pop("final")
    return pd.concat(frames.values(), ignore_index=True), final.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    from models.t6_router.regional_closer import run_closer

    development, final = load_expert_rows(args.study_dir)
    run_closer(development, final, reference_dir=args.study_dir / "gate", output_dir=args.output_dir)
    print("COMPLETE", args.output_dir / "summary.json", flush=True)


if __name__ == "__main__":
    main()
