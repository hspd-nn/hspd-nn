"""Evaluation metrics for hidden-direction and network recovery."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _normalize_columns(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be a matrix")
    n = np.linalg.norm(W, axis=0, keepdims=True)
    return W / np.maximum(n, 1e-15)


def matched_cosines(W_true: np.ndarray, W_est: np.ndarray):
    """Absolute cosine of each true direction with its Hungarian match.

    Returns ``(cos, est_index)``: ``cos[j]`` is the matched absolute cosine of
    true direction j (zero if unmatched because fewer directions were returned)
    and ``est_index[j]`` the matched estimate (-1 if unmatched).
    """
    T = _normalize_columns(W_true)
    r_true = T.shape[1]
    cos = np.zeros(r_true)
    est = -np.ones(r_true, dtype=int)
    if W_est is None or np.asarray(W_est).ndim != 2 or np.asarray(W_est).shape[1] == 0:
        return cos, est
    E = _normalize_columns(W_est)
    S = np.abs(T.T @ E)
    ti, ei = linear_sum_assignment(-S)
    cos[ti] = S[ti, ei]
    est[ti] = ei
    return cos, est


def match_directions(W_true: np.ndarray, W_est: np.ndarray):
    """Backward-compatible summary: ``mean_cosine, min_cosine, true_idx, est_idx``."""
    cos, est = matched_cosines(W_true, W_est)
    ti = np.flatnonzero(est >= 0)
    return float(np.mean(cos)), float(np.min(cos)), ti, est[ti]


def direction_summary(W_true, W_est, certified=None, threshold=0.99):
    """Recovery metrics used in the experiments.

    mean/min matched cosine, the fraction of true directions recovered with
    cosine >= threshold, success (every direction recovered), and, when
    certification flags are given, how many returned candidates were certified
    and how many of those do not match a true direction at the threshold.
    """
    cos, est = matched_cosines(W_true, W_est)
    out = {
        'mean_cosine': float(np.mean(cos)),
        'min_cosine': float(np.min(cos)),
        'frac_recovered': float(np.mean(cos >= threshold)),
        'success': bool(np.all(cos >= threshold)),
        'n_returned': int(0 if W_est is None else np.asarray(W_est).shape[1]),
    }
    if certified is not None and out['n_returned']:
        certified = np.asarray(certified, dtype=bool)
        good = np.zeros(out['n_returned'], dtype=bool)
        for j, e in enumerate(est):
            if e >= 0 and cos[j] >= threshold:
                good[e] = True
        out['n_certified'] = int(certified.sum())
        out['n_false_certified'] = int(np.sum(certified & ~good))
        out['n_missed_certification'] = int(np.sum(~certified & good))
    elif certified is not None:
        out.update({'n_certified': 0, 'n_false_certified': 0, 'n_missed_certification': 0})
    return out, cos, est


def relative_rmse(Y_hat: np.ndarray, Y: np.ndarray) -> float:
    """Root mean square error normalized by the RMS magnitude of ``Y``."""
    Y_hat = np.asarray(Y_hat)
    Y = np.asarray(Y)
    num = np.sqrt(np.mean((Y_hat - Y) ** 2))
    den = np.sqrt(np.mean(Y ** 2))
    return float(num / max(den, 1e-15))
