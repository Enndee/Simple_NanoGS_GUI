from __future__ import annotations

import argparse
import math
import os
import numpy as np
from tqdm import tqdm
from utils.params import RunParams, CostParams
from utils.ply_utils import (
    read_ply,
    store_ply,
)
from utils.cost import edge_costs_gpu_precomputed, gpu_backend_available, make_mc_samples, precompute_cost_state, full_cost_pairs_precomputed, warmup_gpu_backend
from utils.merge import merge_pairs


def resolve_block_edges(device: str, requested_block_edges: int, edge_count: int) -> int:
    if requested_block_edges and requested_block_edges > 0:
        return requested_block_edges

    if device == "gpu":
        if edge_count >= 400_000:
            return 400_000
        if edge_count >= 150_000:
            return 200_000
        return 100_000

    if edge_count >= 400_000:
        return 50_000
    if edge_count >= 150_000:
        return 75_000
    return 50_000


def knn_indices(means: np.ndarray, k: int) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception as e:
        raise RuntimeError("This baseline requires scipy (scipy.spatial.cKDTree).") from e
    tree = cKDTree(means)
    _, idx = tree.query(means, k=k + 1, workers=-1)
    return idx[:, 1:]

def edge_costs(
    edges: np.ndarray,          # (M,2) int32, u<v
    mu: np.ndarray,
    sc: np.ndarray,
    q: np.ndarray,
    op: np.ndarray,
    sh: np.ndarray,
    cp,
    block_edges: int = 0,
) -> np.ndarray:
    """
    Compute symmetric costs w_e for each undirected edge once.
    Returns w: (M,) float32
    """
    M = edges.shape[0]
    state = precompute_cost_state(sc, q, op, cp)
    device = getattr(cp, "device", "auto")

    if device == "auto":
        device = "gpu" if gpu_backend_available() else "cpu"

    block_edges = resolve_block_edges(device, block_edges, M)

    if device == "gpu":
        return edge_costs_gpu_precomputed(edges, mu, sh, state, cp, block_edges)

    w = np.empty((M,), dtype=np.float32)
    mc_samples = make_mc_samples(cp)

    for e0 in tqdm(range(0, M, block_edges), desc="Edge costs"):
        e1 = min(M, e0 + block_edges)
        uv = edges[e0:e1]
        u = uv[:, 0]
        v = uv[:, 1]

        mu_u, sc_u, q_u, op_u = mu[u], sc[u], q[u], op[u]
        mu_v, sc_v, q_v, op_v = mu[v], sc[v], q[v], op[v]

        if sh.shape[1]:
            sh_u = sh[u]
            sh_v = sh[v]
        else:
            # keep shape consistent; full_cost_pairs expects (B,C) even if C=0
            sh_u = sh[u]
            sh_v = sh_u

        state_u = type(state)(
            R=state.R[u],
            Rt=state.Rt[u],
            v=state.v[u],
            invdiag=state.invdiag[u],
            logdet=state.logdet[u],
            weight=state.weight[u],
        )
        state_v = type(state)(
            R=state.R[v],
            Rt=state.Rt[v],
            v=state.v[v],
            invdiag=state.invdiag[v],
            logdet=state.logdet[v],
            weight=state.weight[v],
        )

        w[e0:e1] = full_cost_pairs_precomputed(
            mu_u, sh_u, state_u,
            mu_v, sh_v, state_v,
            cp,
            mc_samples=mc_samples,
        ).astype(np.float32)

    return w

def knn_undirected_edges(nbr: np.ndarray) -> np.ndarray:
    """
    nbr: (N,k) int32 indices (directed kNN).
    Return edges: (M,2) int32 undirected edges with i<j, unique.
    Includes {i,j} if j in kNN(i) OR i in kNN(j) (union).
    """
    N, k = nbr.shape
    ii = np.repeat(np.arange(N, dtype=np.uint32), k)
    jj = nbr.reshape(-1).astype(np.uint32, copy=False)

    u = np.minimum(ii, jj)
    v = np.maximum(ii, jj)

    # remove self edges if any
    mask = u != v
    u = u[mask]
    v = v[mask]

    packed = (u.astype(np.uint64) << np.uint64(32)) | v.astype(np.uint64)
    packed = np.unique(packed)

    edges = np.empty((packed.shape[0], 2), dtype=np.int32)
    edges[:, 0] = (packed >> np.uint64(32)).astype(np.int32)
    edges[:, 1] = (packed & np.uint64(0xFFFFFFFF)).astype(np.int32)
    return edges

def greedy_pairs_from_edges(
    edges: np.ndarray,   # (M,2) int32, u<v
    w: np.ndarray,       # (M,) float32 costs
    N: int,
    P: int | None,       # how many pairs you want this pass
) -> np.ndarray:
    """
    Sort all edges by weight and greedily pick disjoint pairs.
    """
    if edges.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)

    # filter invalid costs if any
    valid = np.isfinite(w)
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.int32)

    idx = np.nonzero(valid)[0]
    order = idx[np.argsort(w[idx], kind="mergesort")]  # stable

    used = np.zeros(N, dtype=bool)
    pairs = []
    for ei in order:
        u, v = int(edges[ei, 0]), int(edges[ei, 1])
        if used[u] or used[v]:
            continue
        used[u] = True
        used[v] = True
        pairs.append((u, v))
        if P is not None and len(pairs) >= P:
            break

    if not pairs:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(pairs, dtype=np.int32)


def prune_by_opacity(mu, sc, q, op, sh, threshold=0.1):
    print("Opacity Mean:", np.mean(op), "Median:", np.median(op))
    threshold = min(threshold, np.median(op))
    print(f"Pruning splats with opacity below {threshold:.4f}")
    keep_idx = np.nonzero(op >= threshold)[0]
    print(f"Original count: {mu.shape[0]}, after opacity pruning: {keep_idx.shape[0]}")

    mu = mu[keep_idx]
    sc = sc[keep_idx]
    q  = q[keep_idx]
    op = op[keep_idx]
    if sh.shape[1]:
        sh = sh[keep_idx]
    return mu, sc, q, op, sh

def simplify(in_path: str, out_path: str, rp: RunParams, cp: CostParams) -> None:

    print(f"Loading PLY: {in_path}")
    hdr, mu, op, sc, q, sh, app_names = read_ply(in_path)
    N0 = int(mu.shape[0])
    print(f"Initial splats: {mu.shape[0]}")
    target = max(int(math.ceil(N0 * rp.ratio)), 1)
    print(f"Pruned splats: {N0}, target: {target}")
    selected_device = getattr(cp, "device", "auto")
    if selected_device == "auto":
        selected_device = "gpu" if gpu_backend_available() else "cpu"
    print(f"Cost device: {selected_device}")
    if selected_device == "gpu":
        warmup_gpu_backend()
    configured_block_edges = getattr(cp, "block_edges", 0)
    mu, sc, q, op, sh = prune_by_opacity(mu, sc, q, op, sh, rp.opacity_threshold)
    print(f"After opacity pruning, {mu.shape[0]} splats remain.")


    iteration = 0
    
    while True:
        if mu.shape[0] <= target:
            break
        N = int(mu.shape[0])
        print(f"Pass {iteration + 1}: {N} splats")

        k_eff = min(max(1, rp.k), max(1, N - 1))
        nbr = knn_indices(mu, k=k_eff)

        edges = knn_undirected_edges(nbr)
        effective_block_edges = resolve_block_edges(selected_device, configured_block_edges, edges.shape[0])
        print(f"  block_edges: {effective_block_edges}")
        w = edge_costs(edges, mu, sc, q, op, sh, cp, block_edges=effective_block_edges)

        merges_needed = N - target
        P = merges_needed if merges_needed > 0 else None

        pairs = greedy_pairs_from_edges(edges, w, N=N, P=P)

        print(f"  edges: {edges.shape[0]}, pairs: {pairs.shape[0]} (need {merges_needed})")

        mu, sc, q, op, sh = merge_pairs(mu, sc, q, op, sh, pairs)

        iteration += 1

    print(f"Final splats: {mu.shape[0]}")
    op = np.clip(op, 0.0, 1.0).astype(np.float32)
    store_ply(out_path, hdr, mu, op, sc, q, sh, app_names)

    


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ply",
        dest="ply",
        required=True,
        help="Input PLY file (raw 3DGS attributes)."
    )
    ap.add_argument(
        "-o", "--output",
        dest="output",
        default=None,
        help="Output PLY path. If omitted, auto-generated from input name and ratio."
    )
    ap.add_argument(
        "-r", "--ratio",
        type=float,
        default=0.5,
        help="Fraction of splats to keep, in (0,1). Example: 0.25 keeps 25%%."
    )

    ap.add_argument("--k", type=int, default=16, help="k for KNN candidates.")
    ap.add_argument("--opacity_threshold", type=float, default=0.1, help="Prune splats with opacity below this threshold before merging.")

    ap.add_argument("--lam_geo", type=float, default=1.0)
    ap.add_argument("--lam_sh", type=float, default=1.0)
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="Edge-cost backend device.")
    ap.add_argument("--block_edges", type=int, default=0, help="Edge-cost block size. Use 0 for auto-tuned defaults.")

    args = ap.parse_args()
    if not (0.0 < args.ratio < 1.0):
        raise ValueError("--ratio must be in the open interval (0, 1).")

    if args.output is not None:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.ply)
        if ext.lower() != ".ply":
            raise ValueError("Input file must have .ply extension.")
        ratio_tag = f"{args.ratio}".rstrip("0").rstrip(".")
        out_path = f"{base}_{ratio_tag}.ply"

    rp = RunParams(
        ratio=args.ratio,
        k=args.k,
        opacity_threshold=args.opacity_threshold,
    )
    cp = CostParams(
        lam_geo=args.lam_geo,
        lam_sh=args.lam_sh,
        device=args.device,
        block_edges=args.block_edges,
    )

    simplify(args.ply, out_path, rp, cp)
    print(f"Done. Wrote: {out_path}")


if __name__ == "__main__":
    main()
