from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from simplification import edge_costs, knn_indices, knn_undirected_edges, prune_by_opacity
from utils.params import CostParams
from utils.ply_utils import read_ply


def main() -> None:
    path = Path('20230717_110552_benchmark_subset.ply')
    hdr, mu, op, sc, q, sh, _ = read_ply(str(path))
    mu, sc, q, op, sh = prune_by_opacity(mu, sc, q, op, sh, 0.05)
    k = 8
    nbr = knn_indices(mu, k=k)
    edges = knn_undirected_edges(nbr)
    print(f'splats={mu.shape[0]} edges={edges.shape[0]}')

    block_sizes = [50000, 100000, 200000, 400000]
    devices = ['cpu', 'gpu']

    for device in devices:
        for block_edges in block_sizes:
            cp = CostParams(lam_geo=1.0, lam_sh=1.0, device=device)
            start = time.perf_counter()
            weights = edge_costs(edges, mu, sc, q, op, sh, cp, block_edges=block_edges)
            elapsed = time.perf_counter() - start
            print(f'device={device} block_edges={block_edges} seconds={elapsed:.4f} weights_mean={float(np.mean(weights)):.6f}')


if __name__ == '__main__':
    main()
