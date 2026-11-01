"""Symmetric vectorization utilities.

The ``svec`` map is an isometry from symmetric matrices with the Frobenius
inner product to Euclidean vectors: diagonal entries are unchanged and
strictly upper-triangular entries are multiplied by sqrt(2).
"""
from __future__ import annotations

import numpy as np


def svec(A: np.ndarray) -> np.ndarray:
    """Symmetric-vectorize one matrix or a batch of matrices.

    Parameters
    ----------
    A : ndarray, shape (..., d, d)
        Symmetric matrix/matrices.

    Returns
    -------
    ndarray, shape (..., d(d+1)/2)
    """
    A = np.asarray(A)
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("A must have shape (..., d, d)")
    d = A.shape[-1]
    pieces = []
    rt2 = np.sqrt(2.0)
    for i in range(d):
        pieces.append(A[..., i, i])
        for j in range(i + 1, d):
            pieces.append(rt2 * A[..., i, j])
    return np.stack(pieces, axis=-1)


def smat(v: np.ndarray, d: int) -> np.ndarray:
    """Inverse of :func:`svec` for one vector or a batch of vectors."""
    v = np.asarray(v)
    p = d * (d + 1) // 2
    if v.shape[-1] != p:
        raise ValueError(f"last dimension must be {p} for d={d}")
    out = np.zeros(v.shape[:-1] + (d, d), dtype=v.dtype)
    k = 0
    rt2 = np.sqrt(2.0)
    for i in range(d):
        out[..., i, i] = v[..., k]
        k += 1
        for j in range(i + 1, d):
            out[..., i, j] = v[..., k] / rt2
            out[..., j, i] = out[..., i, j]
            k += 1
    return out


def svec_features(W: np.ndarray) -> np.ndarray:
    """Return columns ``svec(w_j w_j^T)`` for a direction matrix W.

    Parameters
    ----------
    W : ndarray, shape (d, r)

    Returns
    -------
    ndarray, shape (d(d+1)/2, r)
    """
    W = np.asarray(W)
    if W.ndim != 2:
        raise ValueError("W must have shape (d, r)")
    d, r = W.shape
    feats = np.empty((d * (d + 1) // 2, r), dtype=W.dtype)
    k = 0
    rt2 = np.sqrt(2.0)
    for i in range(d):
        feats[k] = W[i] * W[i]
        k += 1
        for j in range(i + 1, d):
            feats[k] = rt2 * W[i] * W[j]
            k += 1
    return feats
