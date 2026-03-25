from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from utils.ply_utils import write_ply_binary_little_endian


SQRT2 = math.sqrt(2.0)


@dataclass
class SogHeader:
    version: int
    count: int
    sh_degree: int
    palette_size: int


def _decode_webp_rgba(blob: bytes) -> tuple[np.ndarray, int]:
    with Image.open(BytesIO(blob)) as image:
        rgba = image.convert("RGBA")
        width, _ = rgba.size
        data = np.frombuffer(rgba.tobytes(), dtype=np.uint8)
    return data, width


def _decode_log(encoded: np.ndarray) -> np.ndarray:
    encoded = encoded.astype(np.float32, copy=False)
    decoded = np.exp(np.abs(encoded)) - 1.0
    return np.where(encoded < 0.0, -decoded, decoded).astype(np.float32)


def _decode_sog_quaternions(quat_rgba: np.ndarray) -> np.ndarray:
    r0 = ((quat_rgba[:, 0].astype(np.float32) / 255.0) - 0.5) * SQRT2
    r1 = ((quat_rgba[:, 1].astype(np.float32) / 255.0) - 0.5) * SQRT2
    r2 = ((quat_rgba[:, 2].astype(np.float32) / 255.0) - 0.5) * SQRT2
    ri = np.sqrt(np.maximum(0.0, 1.0 - r0 * r0 - r1 * r1 - r2 * r2)).astype(np.float32)

    out = np.empty((quat_rgba.shape[0], 4), dtype=np.float32)
    indices = quat_rgba[:, 3].astype(np.int16) - 252

    mask = indices == 0
    out[mask] = np.stack([ri[mask], r0[mask], r1[mask], r2[mask]], axis=1)

    mask = indices == 1
    out[mask] = np.stack([r0[mask], ri[mask], r1[mask], r2[mask]], axis=1)

    mask = indices == 2
    out[mask] = np.stack([r0[mask], r1[mask], ri[mask], r2[mask]], axis=1)

    mask = indices == 3
    out[mask] = np.stack([r0[mask], r1[mask], r2[mask], ri[mask]], axis=1)

    invalid = (indices < 0) | (indices > 3)
    if np.any(invalid):
        raise ValueError("Unsupported SOG quaternion encoding index.")

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (out / norms).astype(np.float32)


def _decode_opacity(alpha_bytes: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha_bytes.astype(np.float32), 1.0, 254.0) / 255.0
    return np.log(alpha / (1.0 - alpha)).astype(np.float32)


def _decode_sh_rest(meta: dict, archive: zipfile.ZipFile, count: int) -> np.ndarray:
    shn_meta = meta.get("shN")
    if not shn_meta:
        return np.zeros((count, 0), dtype=np.float32)

    files = shn_meta.get("files") or []
    if len(files) < 2:
        return np.zeros((count, 0), dtype=np.float32)

    centroids_bytes, width = _decode_webp_rgba(archive.read(files[0]))
    labels_bytes, _ = _decode_webp_rgba(archive.read(files[1]))
    palette_size = int(shn_meta.get("count") or 65536)
    sh_degree = int(shn_meta.get("bands") or 3)
    sh_dims = {1: 3, 2: 8, 3: 15}
    sh_dim = sh_dims.get(sh_degree)
    if sh_dim is None:
        return np.zeros((count, 0), dtype=np.float32)

    labels = labels_bytes[: count * 4].reshape(count, 4)
    centroids = centroids_bytes.reshape(-1, 4)

    label_values = labels[:, 0].astype(np.int32) | (labels[:, 1].astype(np.int32) << 8)
    rows = label_values >> 6
    cols = label_values & 63
    offsets = rows * width + cols * sh_dim

    valid = label_values < palette_size
    rest = np.zeros((count, sh_dim, 3), dtype=np.float32)
    if not np.any(valid):
        return rest.reshape(count, sh_dim * 3)

    palette_codebook = np.asarray(shn_meta.get("codebook", []), dtype=np.float32)
    if palette_codebook.shape[0] < 256:
        raise ValueError("SOG SH codebook is incomplete.")

    valid_offsets = offsets[valid][:, None] + np.arange(sh_dim, dtype=np.int32)[None, :]
    valid_centroids = centroids[valid_offsets]
    rest_valid = np.empty((valid_offsets.shape[0], sh_dim, 3), dtype=np.float32)
    rest_valid[:, :, 0] = palette_codebook[valid_centroids[:, :, 0]]
    rest_valid[:, :, 1] = palette_codebook[valid_centroids[:, :, 1]]
    rest_valid[:, :, 2] = palette_codebook[valid_centroids[:, :, 2]]
    rest[valid] = rest_valid
    return rest.reshape(count, sh_dim * 3)


def read_sog(path: str | Path) -> tuple[SogHeader, dict[str, np.ndarray]]:
    sog_path = Path(path)
    with zipfile.ZipFile(sog_path) as archive:
        meta = json.loads(archive.read("meta.json"))
        version = int(meta.get("version", 0))
        if version != 2:
            raise ValueError(f"Unsupported SOG version: {version}")

        count = int(meta["count"])
        means_l_bytes, _ = _decode_webp_rgba(archive.read(meta["means"]["files"][0]))
        means_u_bytes, _ = _decode_webp_rgba(archive.read(meta["means"]["files"][1]))
        scales_bytes, _ = _decode_webp_rgba(archive.read(meta["scales"]["files"][0]))
        quats_bytes, _ = _decode_webp_rgba(archive.read(meta["quats"]["files"][0]))
        sh0_bytes, _ = _decode_webp_rgba(archive.read(meta["sh0"]["files"][0]))

        means_l = means_l_bytes[: count * 4].reshape(count, 4)
        means_u = means_u_bytes[: count * 4].reshape(count, 4)
        scales_rgba = scales_bytes[: count * 4].reshape(count, 4)
        quats_rgba = quats_bytes[: count * 4].reshape(count, 4)
        sh0_rgba = sh0_bytes[: count * 4].reshape(count, 4)

        if means_l.shape[0] != count:
            raise ValueError("SOG means payload does not match the declared count.")

        mins = np.asarray(meta["means"]["mins"], dtype=np.float32)
        maxs = np.asarray(meta["means"]["maxs"], dtype=np.float32)

        fx = ((means_u[:, 0].astype(np.uint16) << 8) | means_l[:, 0].astype(np.uint16)).astype(np.float32) / 65535.0
        fy = ((means_u[:, 1].astype(np.uint16) << 8) | means_l[:, 1].astype(np.uint16)).astype(np.float32) / 65535.0
        fz = ((means_u[:, 2].astype(np.uint16) << 8) | means_l[:, 2].astype(np.uint16)).astype(np.float32) / 65535.0

        means_encoded = np.stack([
            mins[0] + (maxs[0] - mins[0]) * fx,
            mins[1] + (maxs[1] - mins[1]) * fy,
            mins[2] + (maxs[2] - mins[2]) * fz,
        ], axis=1).astype(np.float32)
        means = _decode_log(means_encoded)

        scale_codebook = np.asarray(meta["scales"]["codebook"], dtype=np.float32)
        scales = np.stack([
            scale_codebook[scales_rgba[:, 0]],
            scale_codebook[scales_rgba[:, 1]],
            scale_codebook[scales_rgba[:, 2]],
        ], axis=1).astype(np.float32)

        quats = _decode_sog_quaternions(quats_rgba)

        sh0_codebook = np.asarray(meta["sh0"]["codebook"], dtype=np.float32)
        sh0 = np.stack([
            sh0_codebook[sh0_rgba[:, 0]],
            sh0_codebook[sh0_rgba[:, 1]],
            sh0_codebook[sh0_rgba[:, 2]],
        ], axis=1).astype(np.float32)

        opacity = _decode_opacity(sh0_rgba[:, 3])
        sh_rest = _decode_sh_rest(meta, archive, count)
        palette_size = int((meta.get("shN") or {}).get("count") or 0)
        sh_degree = int((meta.get("shN") or {}).get("bands") or 0)

    columns = {
        "x": means[:, 0],
        "y": means[:, 1],
        "z": means[:, 2],
        "nx": np.zeros(count, dtype=np.float32),
        "ny": np.zeros(count, dtype=np.float32),
        "nz": np.zeros(count, dtype=np.float32),
        "f_dc_0": sh0[:, 0],
        "f_dc_1": sh0[:, 1],
        "f_dc_2": sh0[:, 2],
        "opacity": opacity,
        "scale_0": scales[:, 0],
        "scale_1": scales[:, 1],
        "scale_2": scales[:, 2],
        "rot_0": quats[:, 0],
        "rot_1": quats[:, 1],
        "rot_2": quats[:, 2],
        "rot_3": quats[:, 3],
    }

    for index in range(sh_rest.shape[1]):
        columns[f"f_rest_{index}"] = sh_rest[:, index]

    header = SogHeader(
        version=version,
        count=count,
        sh_degree=sh_degree,
        palette_size=palette_size,
    )
    return header, columns


def convert_sog_to_ply(input_path: str | Path, output_path: str | Path) -> SogHeader:
    header, columns = read_sog(input_path)
    properties: list[tuple[str, str]] = [
        ("float", "x"),
        ("float", "y"),
        ("float", "z"),
        ("float", "nx"),
        ("float", "ny"),
        ("float", "nz"),
        ("float", "f_dc_0"),
        ("float", "f_dc_1"),
        ("float", "f_dc_2"),
    ]
    rest_names = sorted(name for name in columns if name.startswith("f_rest_"))
    properties.extend(("float", name) for name in rest_names)
    properties.extend(
        [
            ("float", "opacity"),
            ("float", "scale_0"),
            ("float", "scale_1"),
            ("float", "scale_2"),
            ("float", "rot_0"),
            ("float", "rot_1"),
            ("float", "rot_2"),
            ("float", "rot_3"),
        ]
    )
    write_ply_binary_little_endian(str(output_path), properties, columns)
    return header