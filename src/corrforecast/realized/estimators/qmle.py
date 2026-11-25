"""Quasi-maximum-likelihood covariance estimator of Aït-Sahalia, Fan & Xiu (2010).

Univariate building block (Xiu 2010)
------------------------------------
Observed log prices ``Y_i = X_{t_i} + U_i`` with i.i.d. noise of variance ``a^2``
are treated *as if* X were a Brownian motion with constant volatility.  The
returns ``y = (y_1..y_n)`` are then a Gaussian MA(1) with covariance

    Omega = v I + w T,   v = sigma^2 Delta,  w = a^2,  T = tridiag(-1, 2, -1).

T is diagonalised by the orthonormal type-I discrete sine transform (DST-I):
eigenvalues ``c_k = 2 - 2 cos(k pi / (n+1))``.  With ``z = DST(y)`` the
negative log-likelihood is ``0.5 * sum_k [log(v + w c_k) + z_k^2 / (v + w c_k)]``,
a two-parameter problem we solve by vectorised Fisher scoring over many series
at once.  The integrated variance estimate is ``n * v_hat``.  The QMLE remains
consistent (rate n^{1/4}) with stochastic volatility and is free of tuning
parameters.

Bivariate extension (AFX 2010)
------------------------------
Synchronise each pair with refresh time (their Generalized Sampling Time) and
use the polarisation identity on variance-normalised prices,

    rho_ij = [V(X_i/s_i + X_j/s_j) - V(X_i/s_i - X_j/s_j)] / 4,

where every V is a univariate QMLE.  The covariance is ``rho_ij * sqrt(IV_i IV_j)``.
The resulting matrix is not necessarily PSD; ``psd=True`` projects it.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dst

from corrforecast.realized.estimators.base import CovEstimator, DailyEstimate, register
from corrforecast.realized.estimators.preaveraging import synchronized_returns
from corrforecast.realized.psd import nearest_correlation_psd
from corrforecast.realized.sampling import DayObservations, previous_tick, refresh_times


def dst_coefficients(y: np.ndarray) -> np.ndarray:
    """Orthonormal DST-I along axis 0 (works for 1-D or n x m arrays)."""
    return dst(y, type=1, norm="ortho", axis=0)


def _eigs(n: int, n_max: int | None = None) -> np.ndarray:
    k = np.arange(1, (n_max or n) + 1)
    return 2.0 - 2.0 * np.cos(np.pi * k / (n + 1))


def qmle_batch(
    z2: np.ndarray, n: np.ndarray | int, max_iter: int = 200, tol: float = 1e-10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised QMLE for many MA(1)-noise series.

    Parameters
    ----------
    z2:
        ``(m, n_max)`` squared DST-I coefficients; series shorter than ``n_max`` are zero-padded.
    n:
        Length of each series (scalar if all equal).

    Returns
    -------
    v, w, converged:
        Per-return efficient variance ``sigma^2 Delta``, noise variance ``a^2``, convergence flags.
        Integrated variance is ``n * v``.
    """
    z2 = np.atleast_2d(np.asarray(z2, dtype=float))
    m, n_max = z2.shape
    n_arr = np.full(m, int(n)) if np.isscalar(n) else np.asarray(n, dtype=int)
    if np.any(n_arr < 2):
        raise ValueError("each series needs at least 2 returns")
    k = np.arange(1, n_max + 1)[None, :]
    mask = k <= n_arr[:, None]
    c = np.where(mask, 2.0 - 2.0 * np.cos(np.pi * k / (n_arr[:, None] + 1)), 0.0)
    s = np.where(mask, z2, 0.0)

    # --- method-of-moments start: E[s_k] = v + w c_k (OLS of s on (1, c)) -------
    cnt = mask.sum(1)
    cm, sm = c.sum(1) / cnt, s.sum(1) / cnt
    cc = np.where(mask, c - cm[:, None], 0.0)
    w0 = (cc * (s - sm[:, None])).sum(1) / np.maximum((cc**2).sum(1), 1e-300)
    scale = np.maximum(sm, 1e-300)
    w0 = np.clip(w0, 1e-6 * scale, None)
    v0 = np.clip(sm - w0 * cm, 1e-6 * scale, None)
    theta = np.column_stack([np.log(v0), np.log(w0)])

    f = nll_rows(theta, c, s, mask)
    converged = np.zeros(m, dtype=bool)
    active = np.ones(m, dtype=bool)
    lo = np.log(scale)[:, None] + np.array([[-40.0, -40.0]])
    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        th, cc_, ss, mk = theta[idx], c[idx], s[idx], mask[idx]
        v, w = np.exp(th[:, 0:1]), np.exp(th[:, 1:2])
        lam = v + w * cc_
        a = np.where(mk, 1.0 / lam - ss / lam**2, 0.0)
        b = np.where(mk, 1.0 / lam**2, 0.0)
        gv, gw = 0.5 * a.sum(1) * v[:, 0], 0.5 * (cc_ * a).sum(1) * w[:, 0]
        ivv = 0.5 * b.sum(1) * v[:, 0] ** 2
        ivw = 0.5 * (cc_ * b).sum(1) * v[:, 0] * w[:, 0]
        iww = 0.5 * (cc_**2 * b).sum(1) * w[:, 0] ** 2
        det = ivv * iww - ivw**2
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        dv = -(iww * gv - ivw * gw) / det
        dw = -(-ivw * gv + ivv * gw) / det
        step = np.clip(np.column_stack([dv, dw]), -3.0, 3.0)
        f_old = f[idx]
        t_new = np.maximum(th + step, lo[idx])
        f_new = nll_rows(t_new, cc_, ss, mk)
        for _h in range(30):  # backtracking
            bad = f_new > f_old + 1e-12 * np.abs(f_old)
            if not bad.any():
                break
            step[bad] *= 0.5
            t_new[bad] = np.maximum(th[bad] + step[bad], lo[idx][bad])
            f_new[bad] = nll_rows(t_new[bad], cc_[bad], ss[bad], mk[bad])
        bad = f_new > f_old
        t_new[bad], f_new[bad] = th[bad], f_old[bad]
        theta[idx], f[idx] = t_new, f_new
        done = (
            (np.abs(step).max(1) < tol) | bad | (np.abs(f_old - f_new) <= tol * (1 + np.abs(f_old)))
        )
        converged[idx[done]] = True
        active[idx[done]] = False
    return np.exp(theta[:, 0]), np.exp(theta[:, 1]), converged


def nll_rows(th: np.ndarray, c: np.ndarray, s: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lam = np.exp(th[:, :1]) + np.exp(th[:, 1:]) * c
    return 0.5 * np.where(mask, np.log(lam) + s / lam, 0.0).sum(1)


def qmle_integrated_variance(y: np.ndarray) -> tuple[float, float]:
    """Univariate QMLE of one return series: ``(integrated variance, noise variance)``."""
    y = np.asarray(y, dtype=float)
    z = dst_coefficients(y)
    v, w, _ = qmle_batch(z[None, :] ** 2, len(y))
    return float(len(y) * v[0]), float(w[0])


def _pad(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n = np.array([len(r) for r in rows])
    out = np.zeros((len(rows), n.max()))
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out, n


def polarized_correlation(
    z_i: np.ndarray, z_j: np.ndarray, n: np.ndarray | int, iv_i: np.ndarray, iv_j: np.ndarray
) -> np.ndarray:
    """Polarisation on variance-normalised DST coefficients (rows are pairs)."""
    n_arr = np.broadcast_to(np.asarray(n), (z_i.shape[0],))
    si = np.sqrt(iv_i / n_arr)[:, None]  # per-return std so the normalised series has unit-ish IV/n
    sj = np.sqrt(iv_j / n_arr)[:, None]
    a, b = z_i / si, z_j / sj
    v, _, _ = qmle_batch(np.vstack([(a + b) ** 2, (a - b) ** 2]), np.concatenate([n_arr, n_arr]))
    m = z_i.shape[0]
    vp, vm = v[:m] * n_arr, v[m:] * n_arr  # integrated variances of the normalised sum/difference
    return np.clip((vp - vm) / (4.0 * n_arr), -1.0, 1.0)


@register
class QMLEAitSahaliaFanXiu(CovEstimator):
    """Parameters
    ----------
    sync:
        ``"pairwise_refresh"`` (AFX: refresh time per pair, most data-efficient, O(N^2) work),
        ``"refresh"`` (one refresh-time grid for the whole universe; fast, loses data when N is large),
        or a calendar grid such as ``"1min"`` / ``"30s"``.
    variance_source:
        ``"own"`` — diagonal from the univariate QMLE on each asset's own observations (all ticks);
        ``"sync"`` — from the synchronised data.
    min_obs:
        Minimum raw observations for an asset to be used that day.
    psd:
        Project the resulting correlation matrix onto the PSD cone.
    chunk:
        Pairs solved per vectorised batch (memory control).
    """

    name = "qmle_afx"

    def __init__(
        self,
        sync: str = "pairwise_refresh",
        variance_source: str = "own",
        min_obs: int = 10,
        psd: bool = False,
        chunk: int = 4096,
    ):
        if variance_source not in ("own", "sync"):
            raise ValueError("variance_source must be 'own' or 'sync'")
        self.sync, self.variance_source, self.min_obs, self.psd, self.chunk = (
            sync,
            variance_source,
            int(min_obs),
            bool(psd),
            int(chunk),
        )

    def _own_variances(self, obs: DayObservations, valid: np.ndarray) -> np.ndarray:
        iv = np.full(obs.n_assets, np.nan)
        idx = [i for i in np.flatnonzero(valid) if len(obs.logp[i]) >= 3]
        if not idx:
            return iv
        zs, n = _pad([dst_coefficients(np.diff(obs.logp[i])) for i in idx])
        v, _, _ = qmle_batch(zs**2, n)
        iv[idx] = n * v
        return iv

    def estimate(self, obs: DayObservations) -> DailyEstimate:
        N = obs.n_assets
        valid = obs.n_obs() >= max(self.min_obs, 3)
        corr = np.full((N, N), np.nan)
        np.fill_diagonal(corr, 1.0)
        diag: dict[str, object] = {}
        if self.sync == "pairwise_refresh":
            iv = self._own_variances(obs, valid)
            self._pairwise(obs, valid, corr)
        else:
            r, valid = synchronized_returns(
                obs, "refresh" if self.sync == "refresh" else self.sync, self.min_obs
            )
            n = r.shape[0]
            diag["n_returns"] = n
            if n < 3:
                return DailyEstimate(cov=np.full((N, N), np.nan), diagnostics=diag)
            z = dst_coefficients(r)  # (n, N)
            v, _, _ = qmle_batch((z**2).T, n)
            iv_sync = n * v
            iv = self._own_variances(obs, valid) if self.variance_source == "own" else iv_sync
            vi = np.flatnonzero(valid)
            ii, jj = np.triu_indices(len(vi), 1)
            for lo in range(0, len(ii), self.chunk):
                a, b = vi[ii[lo : lo + self.chunk]], vi[jj[lo : lo + self.chunk]]
                rho = polarized_correlation(z[:, a].T, z[:, b].T, n, iv_sync[a], iv_sync[b])
                corr[a, b] = corr[b, a] = rho
        corr[~valid, :] = np.nan
        corr[:, ~valid] = np.nan
        if self.psd and valid.sum() > 1:
            sub = np.ix_(valid, valid)
            corr[sub] = nearest_correlation_psd(corr[sub])
        sd = np.sqrt(np.where(iv > 0, iv, np.nan))
        cov = corr * np.outer(sd, sd)
        return DailyEstimate(cov=cov, diagnostics=diag)

    def _pairwise(self, obs: DayObservations, valid: np.ndarray, corr: np.ndarray) -> None:
        vi = np.flatnonzero(valid)
        ii, jj = np.triu_indices(len(vi), 1)
        pairs = list(zip(vi[ii], vi[jj], strict=True))
        for lo in range(0, len(pairs), self.chunk):
            batch = pairs[lo : lo + self.chunk]
            zi_rows, zj_rows, keep = [], [], []
            for a, b in batch:
                tau = refresh_times([obs.times[a], obs.times[b]])
                if len(tau) < 4:
                    continue
                ya = np.diff(previous_tick(obs.times[a], obs.logp[a], tau))
                yb = np.diff(previous_tick(obs.times[b], obs.logp[b], tau))
                zi_rows.append(dst_coefficients(ya))
                zj_rows.append(dst_coefficients(yb))
                keep.append((a, b))
            if not keep:
                continue
            zi, n = _pad(zi_rows)
            zj, _ = _pad(zj_rows)
            v, _, _ = qmle_batch(np.vstack([zi**2, zj**2]), np.concatenate([n, n]))
            m = len(keep)
            rho = polarized_correlation(zi, zj, n, n * v[:m], n * v[m:])
            for (a, b), r_ in zip(keep, rho, strict=True):
                corr[a, b] = corr[b, a] = r_
