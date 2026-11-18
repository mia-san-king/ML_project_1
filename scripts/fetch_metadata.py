#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.smapshot import fetch_pages, pages_to_parquet


def main():
    parser = argparse.ArgumentParser(description="Download SMAPshot validated metadata.")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument(
        "--parquet-only",
        action="store_true",
        help="Build smapshot.parquet from pages already on disk.",
    )
    args = parser.parse_args()
    if not args.parquet_only:
        fetch_pages(page=args.page_size, delay=args.delay, max_pages=args.max_pages)
    frame = pages_to_parquet()
    located = frame["latitude"].notna() & frame["longitude"].notna()
    print(f"rows={len(frame)} unique_ids={frame['id'].nunique()} with_coords={int(located.sum())}")


if __name__ == "__main__":
    main()
