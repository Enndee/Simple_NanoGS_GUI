from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from simplification import edge_costs, greedy_pairs_from_edges, knn_indices, knn_undirected_edges, prune_by_opacity
from utils.params import CostParams, RunParams
from utils.ply_utils import read_ply


SOURCE = Path("20230717_110552_benchmark_full.ply")
SIZES = [20_000, 60_000, 120_000, 250_000]
REPEATS = 3


def benchmark_once(size: int, device: str) -> tuple[float, int, int, int]:
    _, mu, op, sc, q, sh, _ = read_ply(str(SOURCE))
    mu = mu[:size]
    op = op[:size]
    sc = sc[:size]
    q = q[:size]
    sh = sh[:size]

    rp = RunParams(ratio=0.9, k=8, opacity_threshold=0.05)
    cp = CostParams(lam_geo=1.0, lam_sh=1.0, device=device, block_edges=0)

    mu, sc, q, op, sh = prune_by_opacity(mu, sc, q, op, sh, rp.opacity_threshold)
    target = max(int(np.ceil(size * rp.ratio)), 1)
    merges_needed = max(mu.shape[0] - target, 0)

    start = time.perf_counter()
    nbr = knn_indices(mu, k=min(max(1, rp.k), max(1, mu.shape[0] - 1)))
    edges = knn_undirected_edges(nbr)
    w = edge_costs(edges, mu, sc, q, op, sh, cp)
    pairs = greedy_pairs_from_edges(edges, w, N=int(mu.shape[0]), P=merges_needed)
    elapsed = time.perf_counter() - start
    return elapsed, int(mu.shape[0]), int(edges.shape[0]), int(pairs.shape[0])


def main() -> None:
    print(f"source={SOURCE}")
    for size in SIZES:
        print(f"\nsize={size}")
        for device in ("cpu", "gpu"):
            timings: list[float] = []
            pruned = edges = pairs = 0
            for _ in range(REPEATS):
                elapsed, pruned, edges, pairs = benchmark_once(size, device)
                timings.append(elapsed)
            avg = sum(timings) / len(timings)
            print(
                f"  {device}: avg={avg:.4f}s min={min(timings):.4f}s max={max(timings):.4f}s "
                f"pruned={pruned} edges={edges} pairs={pairs}"
            )


if __name__ == "__main__":
    main()
