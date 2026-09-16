#!/usr/bin/env python3
"""Merge sharded demonstration files into one, renumbering the episodes.

    python scripts/merge_shards.py data/shards/*.npz --out data/blind.npz

Generating demonstrations is embarrassingly parallel and the episode index is
the only thing that collides, so this offsets it per shard and concatenates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    parts, offset, actuators = [], 0, None
    for path in args.shards:
        z = np.load(path, allow_pickle=False)
        if not len(z["episode"]):
            print(f"  {path}: empty, skipped")
            continue
        ep = z["episode"] + offset
        offset = int(ep.max()) + 1
        actuators = z["actuators"] if actuators is None else actuators
        parts.append(dict(episode=ep, t=z["t"], phase=z["phase"],
                          obs=z["obs"], action=z["action"]))
        print(f"  {path}: {len(ep)} samples, {len(np.unique(z['episode']))} episodes")

    if not parts:
        raise SystemExit("nothing to merge")
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, actuators=actuators, **merged)
    print(f"wrote {args.out}: {len(merged['episode'])} samples, "
          f"{len(np.unique(merged['episode']))} episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
