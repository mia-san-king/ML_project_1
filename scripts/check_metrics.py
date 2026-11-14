#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lib.geo import haversine_km
from lib.metrics import summarize


def main():
    d = haversine_km(46.5197, 6.6323, 47.3769, 8.5417)
    assert 172 <= d <= 175, d
    print(f"lausanne-zurich km={d:.2f}")

    hand = np.array([0.0, 3.0, 10.0, 25.0])
    summary = summarize(hand)
    assert summary["n"] == 4
    assert summary["median_km"] == 6.5
    assert abs(summary["acc@5km"] - 50.0) < 1e-6
    assert abs(summary["acc@25km"] - 100.0) < 1e-6
    print("summarize hand check ok", summary["median_km"], summary["acc@5km"])


if __name__ == "__main__":
  main()