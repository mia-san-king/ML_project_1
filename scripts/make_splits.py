#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from lib.paths import META
from lib.splits import make_splits


def main():
    parser = argparse.ArgumentParser(description="Freeze the SMAPshot train/test split.")
    parser.add_argument("--region", default="Ticino")
    parser.add_argument("--city-lat", type=float, default=46.9480)
    parser.add_argument("--city-lon", type=float, default=7.4474)
    parser.add_argument("--out", default="data/splits/main.csv.gz")
    args = parser.parse_args()

    source = META / "smapshot.parquet"
    if not source.exists():
        sys.exit(f"missing {source}; metadata fetch has not finished")
    frame = pd.read_parquet(source)
    if len(frame) < 350_000:
        sys.exit(f"metadata looks incomplete: {len(frame)} rows")

    splits = make_splits(frame, region=args.region, city_center=(args.city_lat, args.city_lon))
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(destination, index=False)
    print(f"wrote {destination}")
    print(splits.split.value_counts().to_string())


if __name__ == "__main__":
    main()
