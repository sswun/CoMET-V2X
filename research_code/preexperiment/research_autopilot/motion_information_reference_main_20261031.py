"""Small reference for one receiver update with motion-direction uncertainty.

This kernel has no data loading, labels, associations, tuning or detector code.
It marginalizes independent along-heading nuisance motion for an already legal
set of residuals. Per-frame weights sum to one: duplicating the pair list cannot
silently count the same frame as additional independent evidence. Coordinate
transport, process noise and temporal evidence deduplication belong to the caller.
"""
from __future__ import annotations

import numpy as np


def update(mean, covariance, residuals, headings, ages_seconds,
           sigma_perp, sigma_velocity, weights=None, isotropic_control=False):
    """Return posterior mean/covariance and the measurement information matrix.

    R_i = sigma_perp**2 I + (sigma_velocity * age_i)**2 u_i u_i.T.
    The isotropic control preserves each pair's information trace and weight.
    No observation returns the complete prior, including its unobserved axes.
    """
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64).reshape(-1, 2)
    headings = np.asarray(headings, dtype=np.float64).reshape(-1)
    ages_seconds = np.asarray(ages_seconds, dtype=np.float64).reshape(-1)
    count = len(residuals)
    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("A two-dimensional prior is required")
    if len(headings) != count or len(ages_seconds) != count:
        raise ValueError("Residual, heading and physical age identities differ")
    if not all(np.isfinite(a).all() for a in (mean, covariance, residuals, headings, ages_seconds)):
        raise ValueError("Nonfinite inputs")
    if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12):
        raise ValueError("Prior covariance must be symmetric")
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("Prior covariance must be positive definite")
    if not np.isfinite(sigma_perp) or not np.isfinite(sigma_velocity) \
            or sigma_perp <= 0 or sigma_velocity < 0 or np.any(ages_seconds < 0):
        raise ValueError("Invalid noise scales or noncausal ages")
    if not count:
        return mean.copy(), covariance.copy(), np.zeros((2, 2), np.float64)
    weights = np.ones(count) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (count,) or not np.isfinite(weights).all() \
            or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid per-frame pair weights")
    weights = weights / weights.sum()
    identity = np.eye(2)
    unit = np.column_stack((np.cos(headings), np.sin(headings)))
    variances = (sigma_velocity * ages_seconds) ** 2
    noise = sigma_perp**2 * identity + variances[:, None, None] * unit[:, :, None] * unit[:, None, :]
    precision = np.linalg.inv(noise)
    if isotropic_control:
        precision = np.trace(precision, axis1=1, axis2=2)[:, None, None] * identity / 2
    information = np.einsum("n,nij->ij", weights, precision)
    rhs = np.einsum("n,nij,nj->i", weights, precision, residuals)
    prior_information = np.linalg.solve(covariance, identity)
    posterior_information = prior_information + information
    posterior_covariance = np.linalg.solve(posterior_information, identity)
    posterior_mean = np.linalg.solve(posterior_information, prior_information @ mean + rhs)
    return posterior_mean, (posterior_covariance + posterior_covariance.T) / 2, information
