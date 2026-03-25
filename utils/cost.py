from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np


from utils.params import CostParams
from utils.splat_utils import quat_to_rotmat_batch


_GPU_WARMED = False


def _get_cupy() -> Any:
    try:
        os.environ.setdefault("CUPY_COMPILE_WITH_PTX", "1")
        import cupy as cp  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "GPU backend requested, but CuPy is not available. Install a CUDA-compatible CuPy build, for example cupy-cuda12x."
        ) from exc
    return cp


def gpu_backend_available() -> bool:
    try:
        cp = _get_cupy()
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def warmup_gpu_backend() -> None:
    global _GPU_WARMED

    if _GPU_WARMED:
        return

    cp = _get_cupy()
    if cp.cuda.runtime.getDeviceCount() <= 0:
        raise RuntimeError("GPU backend requested, but no CUDA device is available.")

    sample = cp.arange(64, dtype=cp.float32)
    sample = sample * sample + cp.float32(1.0)
    _ = float(sample.sum().get())
    cp.cuda.Stream.null.synchronize()
    _GPU_WARMED = True


@dataclass
class CostState:
    R: Any
    Rt: Any
    v: Any
    invdiag: Any
    logdet: Any
    weight: Any


def _quat_to_rotmat_batch_xp(xp: Any, q_wxyz: Any) -> Any:
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    R = xp.empty((q_wxyz.shape[0], 3, 3), dtype=xp.float32)
    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = 1 - 2 * (xx + yy)
    return R


def _gauss_logpdf_diagrot_batch_xp(
    xp: Any,
    x: Any,
    mu: Any,
    R: Any,
    invdiag: Any,
    logdet: Any,
) -> Any:
    log2pi = xp.log(xp.asarray(2.0 * np.pi, dtype=xp.float32))
    d = x - mu[:, None, :]
    y = _batched_vecmat_xp(xp, d, R)
    quad = xp.sum((y * y) * invdiag[:, None, :], axis=2)
    return -0.5 * (xp.float32(3.0) * log2pi + logdet[:, None] + quad)


def _batched_vecmat_xp(xp: Any, vecs: Any, mats: Any) -> Any:
    out = xp.empty_like(vecs, dtype=xp.float32)
    out[:, :, 0] = (
        vecs[:, :, 0] * mats[:, 0, 0][:, None]
        + vecs[:, :, 1] * mats[:, 1, 0][:, None]
        + vecs[:, :, 2] * mats[:, 2, 0][:, None]
    )
    out[:, :, 1] = (
        vecs[:, :, 0] * mats[:, 0, 1][:, None]
        + vecs[:, :, 1] * mats[:, 1, 1][:, None]
        + vecs[:, :, 2] * mats[:, 2, 1][:, None]
    )
    out[:, :, 2] = (
        vecs[:, :, 0] * mats[:, 0, 2][:, None]
        + vecs[:, :, 1] * mats[:, 1, 2][:, None]
        + vecs[:, :, 2] * mats[:, 2, 2][:, None]
    )
    return out


def _covariance_from_rot_var_xp(xp: Any, R: Any, v: Any) -> Any:
    out = xp.empty((R.shape[0], 3, 3), dtype=xp.float32)
    out[:, 0, 0] = xp.sum(R[:, 0, :] * R[:, 0, :] * v, axis=1)
    out[:, 0, 1] = xp.sum(R[:, 0, :] * R[:, 1, :] * v, axis=1)
    out[:, 0, 2] = xp.sum(R[:, 0, :] * R[:, 2, :] * v, axis=1)
    out[:, 1, 0] = out[:, 0, 1]
    out[:, 1, 1] = xp.sum(R[:, 1, :] * R[:, 1, :] * v, axis=1)
    out[:, 1, 2] = xp.sum(R[:, 1, :] * R[:, 2, :] * v, axis=1)
    out[:, 2, 0] = out[:, 0, 2]
    out[:, 2, 1] = out[:, 1, 2]
    out[:, 2, 2] = xp.sum(R[:, 2, :] * R[:, 2, :] * v, axis=1)
    return out


def _logdet_3x3_xp(xp: Any, A: Any) -> Any:
    a00 = A[:, 0, 0]
    a01 = A[:, 0, 1]
    a02 = A[:, 0, 2]
    a10 = A[:, 1, 0]
    a11 = A[:, 1, 1]
    a12 = A[:, 1, 2]
    a20 = A[:, 2, 0]
    a21 = A[:, 2, 1]
    a22 = A[:, 2, 2]
    det = (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )
    return xp.log(xp.maximum(det, xp.asarray(1e-30, dtype=xp.float32))).astype(xp.float32)


def precompute_cost_state(
    scales: np.ndarray,
    quats: np.ndarray,
    opacity: np.ndarray,
    cost: CostParams,
) -> CostState:
    return _precompute_cost_state_xp(np, scales, quats, opacity, cost)


def _precompute_cost_state_xp(
    xp: Any,
    scales: Any,
    quats: Any,
    opacity: Any,
    cost: CostParams,
) -> CostState:
    eps_cov = float(getattr(cost, "eps_cov", 1e-8))

    R = _quat_to_rotmat_batch_xp(xp, quats).astype(xp.float32, copy=False)
    Rt = xp.transpose(R, (0, 2, 1))

    scales32 = scales.astype(xp.float32, copy=False)
    v = scales32 * scales32 + xp.float32(eps_cov)
    invdiag = (1.0 / xp.maximum(v, 1e-30)).astype(xp.float32)
    logdet = xp.sum(xp.log(xp.maximum(v, 1e-30)), axis=1).astype(xp.float32)
    weight = (
        xp.float32((2 * np.pi) ** 1.5)
        * opacity.astype(xp.float32, copy=False)
        * xp.prod(scales32, axis=1).astype(xp.float32)
        + xp.float32(1e-12)
    )

    return CostState(
        R=R,
        Rt=Rt,
        v=v,
        invdiag=invdiag,
        logdet=logdet,
        weight=weight.astype(xp.float32, copy=False),
    )


def make_mc_samples(cost: CostParams) -> np.ndarray:
    n_mc = int(getattr(cost, "n_mc", 1))
    seed = int(getattr(cost, "seed", 0))
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n_mc, 3)).astype(np.float32)


def full_cost_pairs_precomputed(
    mu_i: np.ndarray,
    sh_i: np.ndarray,
    state_i: CostState,
    mu_j: np.ndarray,
    sh_j: np.ndarray,
    state_j: CostState,
    cost: CostParams,
    mc_samples: np.ndarray | None = None,
) -> np.ndarray:
    return _full_cost_pairs_precomputed_xp(np, mu_i, sh_i, state_i, mu_j, sh_j, state_j, cost, mc_samples)


def _full_cost_pairs_precomputed_xp(
    xp: Any,
    mu_i: Any,
    sh_i: Any,
    state_i: CostState,
    mu_j: Any,
    sh_j: Any,
    state_j: CostState,
    cost: CostParams,
    mc_samples: Any | None = None,
) -> Any:
    eps_cov = float(getattr(cost, "eps_cov", 1e-8))

    R_i = state_i.R
    R_j = state_j.R
    Rt_i = state_i.Rt
    Rt_j = state_j.Rt
    v_i = state_i.v
    v_j = state_j.v
    invdiag_i = state_i.invdiag
    invdiag_j = state_j.invdiag
    logdet_i = state_i.logdet
    logdet_j = state_j.logdet
    w_i = state_i.weight
    w_j = state_j.weight

    W = w_i + w_j
    W_safe = xp.where(W > 0.0, W, 1.0).astype(xp.float32)

    pi = (w_i / W_safe).astype(xp.float32)
    pi = xp.clip(pi, 1e-12, 1.0 - 1e-12).astype(xp.float32)
    log_pi = xp.log(pi).astype(xp.float32)
    log_pj = xp.log(1.0 - pi).astype(xp.float32)

    mu_i32 = mu_i.astype(xp.float32, copy=False)
    mu_j32 = mu_j.astype(xp.float32, copy=False)

    mu_m = pi[:, None] * mu_i32 + (1.0 - pi)[:, None] * mu_j32

    di = mu_i32 - mu_m
    dj = mu_j32 - mu_m
    odi = di[:, :, None] * di[:, None, :]
    odj = dj[:, :, None] * dj[:, None, :]

    Sig_i = _covariance_from_rot_var_xp(xp, R_i, v_i)
    Sig_j = _covariance_from_rot_var_xp(xp, R_j, v_j)

    Sig_m = pi[:, None, None] * (Sig_i + odi) + (1.0 - pi)[:, None, None] * (Sig_j + odj)
    I = xp.eye(3, dtype=xp.float32)[None, :, :]
    Sig_m = 0.5 * (Sig_m + xp.transpose(Sig_m, (0, 2, 1))) + xp.float32(eps_cov) * I

    logdet_m = _logdet_3x3_xp(xp, Sig_m)

    k = xp.float32(3.0)
    log2pi = xp.log(xp.asarray(2.0 * np.pi, dtype=xp.float32))
    E_p_neglogq = 0.5 * (k * log2pi + logdet_m + k).astype(xp.float32)

    Z = mc_samples if mc_samples is not None else make_mc_samples(cost)

    std_i = xp.sqrt(xp.maximum(v_i, 0.0)).astype(xp.float32)
    std_j = xp.sqrt(xp.maximum(v_j, 0.0)).astype(xp.float32)

    Zi = Z[None, :, :] * std_i[:, None, :]
    Zj = Z[None, :, :] * std_j[:, None, :]

    x_i = mu_i32[:, None, :] + _batched_vecmat_xp(xp, Zi, Rt_i)
    x_j = mu_j32[:, None, :] + _batched_vecmat_xp(xp, Zj, Rt_j)

    logNi_on_i = _gauss_logpdf_diagrot_batch_xp(xp, x_i, mu_i32, R_i, invdiag_i, logdet_i)
    logNj_on_i = _gauss_logpdf_diagrot_batch_xp(xp, x_i, mu_j32, R_j, invdiag_j, logdet_j)
    logp_on_i = xp.logaddexp(log_pi[:, None] + logNi_on_i, log_pj[:, None] + logNj_on_i)
    Ei = xp.mean(logp_on_i, axis=1).astype(xp.float32)

    logNi_on_j = _gauss_logpdf_diagrot_batch_xp(xp, x_j, mu_i32, R_i, invdiag_i, logdet_i)
    logNj_on_j = _gauss_logpdf_diagrot_batch_xp(xp, x_j, mu_j32, R_j, invdiag_j, logdet_j)
    logp_on_j = xp.logaddexp(log_pi[:, None] + logNi_on_j, log_pj[:, None] + logNj_on_j)
    Ej = xp.mean(logp_on_j, axis=1).astype(xp.float32)

    E_p_logp = (pi * Ei + (1.0 - pi) * Ej).astype(xp.float32)
    geo = (E_p_logp + E_p_neglogq).astype(xp.float32)

    if sh_i.shape[1] == 0:
        c_sh = xp.zeros_like(geo, dtype=xp.float32)
    else:
        diff = sh_i.astype(xp.float32, copy=False) - sh_j.astype(xp.float32, copy=False)
        c_sh = xp.sum(diff * diff, axis=1).astype(xp.float32)

    return (cost.lam_geo * geo + cost.lam_sh * c_sh).astype(xp.float32)


def edge_costs_gpu_precomputed(
    edges: np.ndarray,
    mu: np.ndarray,
    sh: np.ndarray,
    state: CostState,
    cost: CostParams,
    block_edges: int,
) -> np.ndarray:
    cp = _get_cupy()

    mu_gpu = cp.asarray(mu, dtype=cp.float32)
    sh_gpu = cp.asarray(sh, dtype=cp.float32)
    state_gpu = CostState(
        R=cp.asarray(state.R, dtype=cp.float32),
        Rt=cp.asarray(state.Rt, dtype=cp.float32),
        v=cp.asarray(state.v, dtype=cp.float32),
        invdiag=cp.asarray(state.invdiag, dtype=cp.float32),
        logdet=cp.asarray(state.logdet, dtype=cp.float32),
        weight=cp.asarray(state.weight, dtype=cp.float32),
    )
    mc_samples_gpu = cp.asarray(make_mc_samples(cost), dtype=cp.float32)

    w = np.empty((edges.shape[0],), dtype=np.float32)
    for e0 in range(0, edges.shape[0], block_edges):
        e1 = min(edges.shape[0], e0 + block_edges)
        uv = edges[e0:e1]
        u = cp.asarray(uv[:, 0], dtype=cp.int32)
        v = cp.asarray(uv[:, 1], dtype=cp.int32)

        sh_u = sh_gpu[u] if sh_gpu.shape[1] else sh_gpu[:0]
        sh_v = sh_gpu[v] if sh_gpu.shape[1] else sh_gpu[:0]
        if sh_gpu.shape[1] == 0:
            sh_u = sh_gpu[u]
            sh_v = sh_u

        state_u = CostState(
            R=state_gpu.R[u],
            Rt=state_gpu.Rt[u],
            v=state_gpu.v[u],
            invdiag=state_gpu.invdiag[u],
            logdet=state_gpu.logdet[u],
            weight=state_gpu.weight[u],
        )
        state_v = CostState(
            R=state_gpu.R[v],
            Rt=state_gpu.Rt[v],
            v=state_gpu.v[v],
            invdiag=state_gpu.invdiag[v],
            logdet=state_gpu.logdet[v],
            weight=state_gpu.weight[v],
        )

        block_w = _full_cost_pairs_precomputed_xp(
            cp,
            mu_gpu[u],
            sh_u,
            state_u,
            mu_gpu[v],
            sh_v,
            state_v,
            cost,
            mc_samples=mc_samples_gpu,
        )
        w[e0:e1] = cp.asnumpy(block_w)

    cp.cuda.Stream.null.synchronize()
    return w


def full_cost_pairs(
    mu_i: np.ndarray,
    s_i: np.ndarray,
    q_i: np.ndarray,
    a_i: np.ndarray,
    sh_i: np.ndarray,
    mu_j: np.ndarray,
    s_j: np.ndarray,
    q_j: np.ndarray,
    a_j: np.ndarray,
    sh_j: np.ndarray,
    cost: CostParams,
) -> np.ndarray:
    state_i = precompute_cost_state(s_i, q_i, a_i, cost)
    state_j = precompute_cost_state(s_j, q_j, a_j, cost)
    return full_cost_pairs_precomputed(mu_i, sh_i, state_i, mu_j, sh_j, state_j, cost)

def full_cost_pairs_ij(
    mu_i: np.ndarray,
    s_i: np.ndarray,
    q_i: np.ndarray,
    a_i: np.ndarray,
    sh_i: np.ndarray,
    mu_j: np.ndarray,
    s_j: np.ndarray,
    q_j: np.ndarray,
    a_j: np.ndarray,
    sh_j: np.ndarray,
    cost: CostParams,
) -> np.ndarray:
    # knobs
    eps_cov = float(getattr(cost, "eps_cov", 1e-8))  # scalar variance jitter

    # ---- rotations ----
    R_i = quat_to_rotmat_batch(q_i).astype(np.float32)  # [B,3,3] world->local
    R_j = quat_to_rotmat_batch(q_j).astype(np.float32)

    Rt_i = np.transpose(R_i, (0, 2, 1))
    Rt_j = np.transpose(R_j, (0, 2, 1))

    # ---- variances (scalar jitter) ----
    s_i32 = s_i.astype(np.float32, copy=False)
    s_j32 = s_j.astype(np.float32, copy=False)

    v_i = s_i32 * s_i32 + np.float32(eps_cov)  # [B,3]
    v_j = s_j32 * s_j32 + np.float32(eps_cov)

    invdiag_i = (1.0 / np.maximum(v_i, 1e-30)).astype(np.float32)
    invdiag_j = (1.0 / np.maximum(v_j, 1e-30)).astype(np.float32)

    logdet_i = np.sum(np.log(np.maximum(v_i, 1e-30)), axis=1).astype(np.float32)  # [B]
    logdet_j = np.sum(np.log(np.maximum(v_j, 1e-30)), axis=1).astype(np.float32)

    mu_i32 = mu_i.astype(np.float32, copy=False)
    mu_j32 = mu_j.astype(np.float32, copy=False)

    # delta in world
    d_ij = (mu_i32 - mu_j32).astype(np.float32)  # [B,3]
    d_ji = -d_ij

    k = np.float32(3.0)

    # ---------- KL(i || j) ----------
    # mean quadratic: (mu_j - mu_i)^T Sigma_j^{-1} (mu_j - mu_i)
    # with R_j world->local: local coords = d @ R_j
    d_ij_in_j = np.einsum("bi,bij->bj", d_ij, R_j)  # [B,3]
    quad_ij = np.sum(d_ij_in_j * d_ij_in_j * invdiag_j, axis=1).astype(np.float32)

    # trace term: tr(Sigma_j^{-1} Sigma_i)
    # Let M = R_j^T R_i (maps i-local axes into j-local axes in terms of world rotations)
    # In j-local frame: R_j^T Sigma_i R_j = M diag(v_i) M^T
    # diag of that is sum_k M[a,k]^2 * v_i[k]
    M_ji = np.matmul(Rt_j, R_i).astype(np.float32)  # [B,3,3]
    diagC_ij = np.sum((M_ji * M_ji) * v_i[:, None, :], axis=2).astype(np.float32)  # [B,3]
    tr_ij = np.sum(invdiag_j * diagC_ij, axis=1).astype(np.float32)

    KL_ij = 0.5 * ((logdet_j - logdet_i) - k + tr_ij + quad_ij).astype(np.float32)

    # ---------- KL(j || i) ----------
    d_ji_in_i = np.einsum("bi,bij->bj", d_ji, R_i)  # [B,3]
    quad_ji = np.sum(d_ji_in_i * d_ji_in_i * invdiag_i, axis=1).astype(np.float32)

    M_ij = np.matmul(Rt_i, R_j).astype(np.float32)  # [B,3,3]
    diagC_ji = np.sum((M_ij * M_ij) * v_j[:, None, :], axis=2).astype(np.float32)
    tr_ji = np.sum(invdiag_i * diagC_ji, axis=1).astype(np.float32)

    KL_ji = 0.5 * ((logdet_i - logdet_j) - k + tr_ji + quad_ji).astype(np.float32)

    # symmetrized KL (Jeffreys)
    geo = 0.5 * (KL_ij + KL_ji).astype(np.float32)

    # ---- SH L2 ----
    if sh_i.shape[1] == 0:
        c_sh = np.zeros_like(geo, dtype=np.float32)
    else:
        diff = sh_i.astype(np.float32) - sh_j.astype(np.float32)
        c_sh = np.sum(diff * diff, axis=1).astype(np.float32)

    out = (cost.lam_geo * geo + cost.lam_sh * c_sh).astype(np.float32)
    return out