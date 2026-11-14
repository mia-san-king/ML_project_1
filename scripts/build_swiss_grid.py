#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lib.geo import swiss_grid
from lib.paths import GEO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-m", type=int, default=1000)
    parser.add_argument("--buffer-m", type=int, default=5000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out) if args.out else GEO / f"grid_{args.step_m // 1000}km.npy"
    grid = swiss_grid(step_m=args.step_m, buffer_m=args.buffer_m)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, grid)
    print(f"wrote {out} shape={grid.shape}")


if __name__ == "__main__":
    main()