import numpy as np

from .contracts import DepthEstimation


def ordering_agreement(
    a: DepthEstimation,
    b: DepthEstimation,
    pairs: int = 20000,
    min_ratio: float = 1.10,
    seed: int = 0,
) -> float:
    both = a.valid & b.valid
    idx = np.flatnonzero(both)
    if idx.size < 2:
        return float("nan")

    rng = np.random.default_rng(seed)
    i, j = rng.choice(idx, pairs), rng.choice(idx, pairs)
    da, db = a.depth.ravel(), b.depth.ravel()

    near, far = np.minimum(db[i], db[j]), np.maximum(db[i], db[j])
    offset = 0.0 if b.is_metric else 0.05            # relative depth reaches 0 at the nearest pixel
    clear = (far + offset) >= min_ratio * (near + offset)
    if not clear.any():
        return float("nan")

    same = np.sign(da[i] - da[j]) == np.sign(db[i] - db[j])
    return float(same[clear].mean())


def align_to_metric(
    relative: DepthEstimation,
    metric: DepthEstimation,
    samples: int = 20000,
    iterations: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Express a relative estimation in the metric one's metres.

    Fits  1/metric ≈ s * disparity + t  (the relationship relative models are trained for),
    dropping outliers over a few rounds, then returns depth = 1 / (s * disparity + t).
    Shape is untouched: only one scale and one shift are chosen for the whole image.

    Returns (aligned depth in metres with NaN where unusable, {"scale", "shift", "inliers"}).
    """
    if relative.disparity is None:
        raise ValueError("relative estimation has no raw disparity (is it a metric model?)")
    if not metric.is_metric:
        raise ValueError("reference must be metric_depth")

    both = relative.valid & metric.valid & np.isfinite(relative.disparity)
    idx = np.flatnonzero(both)
    if idx.size < 10:
        return np.full(relative.depth.shape, np.nan, np.float32), {"scale": np.nan, "shift": np.nan, "inliers": 0.0}

    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, min(samples, idx.size), replace=False)
    x = relative.disparity.ravel()[pick].astype(np.float64)
    y = 1.0 / metric.depth.ravel()[pick].astype(np.float64)

    # Robust start (RANSAC): a plain least-squares fit can be dragged far off by a few extreme
    # values (e.g. a patch of garbage), after which trimming would throw away the GOOD points.
    keep = np.zeros(x.size, bool)
    for _ in range(200):
        a, b = rng.choice(x.size, 2, replace=False)
        if abs(x[a] - x[b]) < 1e-9:
            continue
        s = (y[a] - y[b]) / (x[a] - x[b])
        t = y[a] - s * x[a]
        if s <= 0:
            continue                                # more disparity must mean nearer
        inliers = np.abs(s * x + t - y) < 0.10 * np.abs(y)
        if inliers.sum() > keep.sum():
            keep = inliers
    if keep.sum() < 2:
        keep[:] = True

    # Refine on the inliers, re-trimming outliers a few times.
    for _ in range(iterations):
        s, t = np.polyfit(x[keep], y[keep], 1)
        residual = np.abs(s * x + t - y)
        mad = np.median(residual[keep]) + 1e-12
        keep = residual < 3.0 * 1.4826 * mad

    disparity = s * relative.disparity.astype(np.float64) + t
    aligned = np.where(relative.valid & (disparity > 1e-6), 1.0 / np.maximum(disparity, 1e-6), np.nan)
    return aligned.astype(np.float32), {"scale": float(s), "shift": float(t), "inliers": float(keep.mean())}


def relative_difference(aligned: np.ndarray, metric: DepthEstimation) -> np.ndarray:
    """|aligned - metric| / metric per pixel (NaN where either is missing)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = np.abs(aligned - metric.depth) / metric.depth
    diff[~(np.isfinite(aligned) & metric.valid)] = np.nan
    return diff.astype(np.float32)
