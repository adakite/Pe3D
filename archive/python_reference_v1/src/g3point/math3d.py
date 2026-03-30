from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


EPS = 1e-12


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return vectors / norms


def fit_plane(points: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    if abs(normal[2]) < EPS:
        normal = normal + np.array([0.0, 0.0, np.sign(normal[2]) or 1.0]) * EPS
    plane = (-1.0 / normal[2]) * normal
    a = float(plane[0])
    b = float(plane[1])
    c = float(-centroid @ plane)
    denom = math.sqrt(a * a + b * b + 1.0)
    signed = (c + a * points[:, 0] + b * points[:, 1] - points[:, 2]) / denom
    return a, b, c, signed, np.abs(signed)


def rotation_matrix_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    a = normalize(np.asarray(v1, dtype=np.float64).reshape(1, 3))[0]
    b = normalize(np.asarray(v2, dtype=np.float64).reshape(1, 3))[0]
    cross = np.cross(a, b)
    dot = float(np.clip(a @ b, -1.0, 1.0))
    norm_cross = np.linalg.norm(cross)

    if norm_cross < EPS:
        if dot > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = normalize(np.cross(a, axis).reshape(1, 3))[0]
        return rotation_matrix_from_axis_angle(axis, math.pi)

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (norm_cross * norm_cross))


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(np.asarray(axis, dtype=np.float64).reshape(1, 3))[0]
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )


def rotate_points(points: np.ndarray, rotation: np.ndarray, origin: np.ndarray | None = None) -> np.ndarray:
    if origin is None:
        origin = np.zeros(3)
    shifted = points - origin
    return shifted @ rotation.T + origin


def detrend_quadratic(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    design = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
    detrended = points.copy()
    detrended[:, 2] = z - design @ coeffs
    return detrended, coeffs


def orient_vectors(points: np.ndarray, vectors: np.ndarray, sensor_center: np.ndarray) -> np.ndarray:
    oriented = vectors.copy()
    directions = sensor_center.reshape(1, 3) - points
    flip = np.einsum("ij,ij->i", directions, oriented) < 0
    oriented[flip] *= -1.0
    return oriented


def estimate_normals(points: np.ndarray, k: int, sensor_center: np.ndarray | None = None) -> np.ndarray:
    if points.shape[0] < 3:
        return np.zeros_like(points)

    query_k = min(max(k + 1, 3), points.shape[0])
    tree = cKDTree(points)
    _, indices = tree.query(points, k=query_k)
    if indices.ndim == 1:
        indices = indices[:, None]
    if query_k > 1:
        indices = indices[:, 1:]

    normals = np.empty_like(points)
    chunk_size = 4096
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        neighbours = points[indices[start:stop]]
        centred = neighbours - neighbours.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centred, centred) / max(neighbours.shape[1] - 1, 1)
        _, eigenvectors = np.linalg.eigh(cov)
        normals[start:stop] = eigenvectors[:, :, 0]

    if sensor_center is not None:
        normals = orient_vectors(points, normals, sensor_center)
    return normalize(normals)


def angles_to_reference(reference: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    ref = normalize(np.asarray(reference, dtype=np.float64).reshape(1, 3))[0]
    vecs = normalize(vectors)
    cross_norm = np.linalg.norm(np.cross(np.repeat(ref[None, :], vecs.shape[0], axis=0), vecs), axis=1)
    dot = np.clip(vecs @ ref, -1.0, 1.0)
    return np.degrees(np.arctan2(cross_norm, dot))


def fibonacci_sphere(samples: int) -> np.ndarray:
    if samples <= 0:
        return np.empty((0, 3))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    points = np.empty((samples, 3), dtype=np.float64)
    for idx in range(samples):
        y = 1.0 - (2.0 * idx) / max(samples - 1, 1)
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * idx
        points[idx] = [math.cos(theta) * radius, y, math.sin(theta) * radius]
    return points

