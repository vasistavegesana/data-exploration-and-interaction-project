"""Local orthogonal affine fitting, following Joia et al. (TVCG 2011).

Independent NumPy implementation; inverse-Euclidean weighting follows the
authors' reference implementation at https://github.com/lgnonato/LAMP.
No optional automatic rescaling: controls stay in the initial metric-MDS units.
"""
import numpy as np
from scipy.spatial.distance import cdist


def choose_controls(X, count=24):
    """Deterministic farthest-point coverage, starting near the data centroid."""
    X = np.asarray(X, dtype=float)
    chosen = [int(np.argmin(np.sum((X-X.mean(axis=0))**2, axis=1)))]
    nearest = np.full(len(X), np.inf)
    while len(chosen) < min(count, len(X)):
        nearest = np.minimum(nearest, np.sum((X-X[chosen[-1]])**2, axis=1))
        nearest[chosen] = -1
        nxt = int(np.argmax(nearest))
        if nearest[nxt] <= 1e-16:
            break
        chosen.append(nxt)
    return np.array(chosen)


def classical_mds(X):
    """Classical metric MDS of Euclidean control distances into two dimensions."""
    squared = cdist(X, X, 'sqeuclidean')
    centered = -.5*(squared-squared.mean(axis=0)-squared.mean(axis=1)[:,None]+squared.mean())
    vals, vecs = np.linalg.eigh(centered)
    order = np.argsort(vals)[::-1][:2]
    return vecs[:,order]*np.sqrt(np.maximum(vals[order],0))


def project(X, control_ids, targets):
    """Fit M per input record with M.T @ M = I and map about weighted centers.

    Given C= X[control_ids], each local cross covariance is
    H = (C-xbar).T @ diag(w) @ (targets-ybar). For H=U S V.T,
    the orthogonal least-squares solution is M=U @ V.T.
    Exact control matches interpolate the supplied target without division by 0.
    """
    X = np.asarray(X, dtype=float)
    control_ids = np.asarray(control_ids, dtype=int)
    targets = np.asarray(targets, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2 or targets.shape != (len(control_ids),2):
        raise ValueError('Expected N x d input (d>=2) and m x 2 control targets')
    if len(control_ids) < 3 or len(set(control_ids)) != len(control_ids):
        raise ValueError('Use at least three distinct control IDs')
    if not np.isfinite(X).all() or not np.isfinite(targets).all():
        raise ValueError('Inputs and controls must be finite')
    C = X[control_ids]
    distances = cdist(X,C)
    result = np.empty((len(X),2))
    for i,x in enumerate(X):
        close = np.flatnonzero(distances[i] <= 1e-10)
        if len(close):
            result[i] = targets[close[0]]
            continue
        weights = 1 / distances[i]
        weights /= weights.sum()
        xbar = weights @ C
        ybar = weights @ targets
        H = (C-xbar).T @ (weights[:,None]*(targets-ybar))
        U, _, Vt = np.linalg.svd(H, full_matrices=False)
        result[i] = (x-xbar) @ (U@Vt)+ybar
    return result
