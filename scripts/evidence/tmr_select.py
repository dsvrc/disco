"""
Replay-buffer selection rules for the evidence scripts: TMR (paper Algorithm 1, OMP variant) with its Tab. 4
ablations, and the classical baselines it is compared against. All rules work on a past pool of N windows with a
session label each and per-session quotas B_j = ceil(budget * n_j) (Eq. select), and return (indices, weights) with
weights on the simplex.

  tmr              whitened temporal modes (top-M_hat eigenvectors of C_k, BIC rank), simplex-constrained OMP
  tmr_uniform      TMR's support with uniform weights (separates the support from the refit weights)
  no_whiten        TMR without Lambda^{-1/2}                                    (Tab. 4 "No whitening")
  random_subspace  random orthonormal temporal basis of the same rank             (Tab. 4 "Random subspace")
  raw_omp          the same OMP on vec(phi_i)                                     (Tab. 4 "Raw-feature OMP")
  channel_pca      covariance over the feature axis (D x D) instead of time      (Tab. 4 "Channel PCA")
  herding          iCaRL herding to each session's mean trajectory (uniform weights)
  kcenter          k-center greedy coverage in trajectory space (uniform weights)
  random           uniform per-session sampling = ER (uniform weights)
  grad_omp         OMP on per-sample gradient sketches: the gradient-matching oracle (GCR / GradMatch)
  loss_max         highest frozen-noise loss per session (loss-based selection, MIR / CSReL proxy)
"""

from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np


# ----------------------------------------------------------------------------------------------
# Operators and rank
# ----------------------------------------------------------------------------------------------
def temporal_operator(phi: np.ndarray, chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """C_k = 1/(N D) sum_i phi~_i phi~_i^T (T x T, Eq. operator) and the mean trajectory phi_bar."""
    N, T, D = phi.shape
    mean = phi.mean(0, dtype=np.float64)
    C = np.zeros((T, T))
    for s in range(0, N, chunk):
        x = (phi[s:s + chunk] - mean).transpose(1, 0, 2).reshape(T, -1)
        C += x @ x.T
    return C / (N * D), mean


def channel_operator(phi: np.ndarray, mean: np.ndarray, chunk: int = 64) -> np.ndarray:
    """C^chan = 1/(N T) sum_i phi~_i^T phi~_i (D x D), the feature-axis operator of the Tab. 4 ablation."""
    N, T, D = phi.shape
    C = np.zeros((D, D))
    for s in range(0, N, chunk):
        x = (phi[s:s + chunk] - mean).reshape(-1, D)
        C += x.T @ x
    return C / (N * T)


def eig_desc(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lam, U = np.linalg.eigh(C)
    return np.clip(lam[::-1], 0, None), U[:, ::-1]


def bic_rank(lam: np.ndarray, n_eff: float) -> tuple[int, dict]:
    """Schwarz criterion on the PPCA model (Eq. bic). lam: descending eigenvalues of a d x d operator."""
    d = len(lam)
    rank = int((lam > lam[0] * 1e-12).sum()) if lam[0] > 0 else 0
    if rank <= 1:
        return 1, {"rank": rank, "M_max": 1, "criterion": []}
    M_max = min(d, rank) - 1
    crit = []
    for M in range(1, M_max + 1):
        s2 = lam[M:].sum() / (d - M)
        fit = np.log(lam[:M]).sum() + (d - M) * np.log(s2)
        crit.append(n_eff * fit + (d * M - M * (M - 1) / 2 + 1) * np.log(n_eff))
    return int(np.argmin(crit)) + 1, {"rank": rank, "M_max": M_max, "criterion": crit}


# ----------------------------------------------------------------------------------------------
# Representations (dictionaries) used for selection
# ----------------------------------------------------------------------------------------------
def modal_dictionary(phi: np.ndarray, kind: str = "temporal", whiten: bool = True, basis: str = "eig",
                     M: int | None = None, rng=None, ops: dict | None = None) -> tuple[np.ndarray, dict]:
    """Atoms z_i = vec(gamma_i) (Eq. whiten) with gamma_i = Lambda^{-1/2} V^T phi_i on the *uncentred* phi_i.
    kind='temporal': V = top-M eigenvectors of C_k (T x T); kind='channel': of C^chan, acting on phi_i^T.
    basis='random': a random orthonormal basis of the same rank, whitened by its own coordinate variances."""
    N, T, D = phi.shape
    ops = ops if ops is not None else operators(phi, channel=(kind == "channel"))
    if kind == "temporal":
        lam, U, n_eff = ops["lam"], ops["U"], N * D
    else:
        lam, U, n_eff = ops["lam_c"], ops["U_c"], N * T
    M = M or bic_rank(lam, n_eff)[0]
    if basis == "eig":
        V, var = U[:, :M], lam[:M]
    else:
        V, _ = np.linalg.qr((rng or np.random.default_rng(0)).standard_normal((U.shape[0], M)))
        Cop = ops["C"] if kind == "temporal" else ops["C_c"]
        var = np.einsum("tm,ts,sm->m", V, Cop, V)
    scale = 1.0 / np.sqrt(np.maximum(var, 1e-30)) if whiten else np.ones(M)
    Vs = (V * scale).astype(np.float32)  # columns already whitened
    if kind == "temporal":
        Z = np.einsum("tm,ntd->nmd", Vs, phi, optimize=True).reshape(N, -1)
    else:
        Z = np.einsum("dm,ntd->nmt", Vs, phi, optimize=True).reshape(N, -1)
    return np.ascontiguousarray(Z, dtype=np.float32), {"M": int(M), "kind": kind, "whiten": whiten, "basis": basis}


def operators(phi: np.ndarray, channel: bool = True) -> dict:
    C, mean = temporal_operator(phi)
    lam, U = eig_desc(C)
    out = {"C": C, "mean": mean, "lam": lam, "U": U}
    if channel:
        Cc = channel_operator(phi, mean)
        lam_c, U_c = eig_desc(Cc)
        out.update(C_c=Cc, lam_c=lam_c, U_c=U_c)
    return out


# ----------------------------------------------------------------------------------------------
# Simplex-constrained refit and OMP (Algorithm 1)
# ----------------------------------------------------------------------------------------------
def project_simplex(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    r = ind[u - css / ind > 0][-1]
    return np.maximum(v - css[r - 1] / r, 0.0)


def simplex_lsq(G: np.ndarray, b: np.ndarray, rho: float = 0.0, w0=None, iters: int = 5000,
                tol: float = 1e-13) -> np.ndarray:
    """argmin_{w >= 0, 1^T w = 1} w^T (G + rho I) w - 2 b^T w, i.e. ||y - Z_S w||^2 + rho ||w||^2 with G = Z_S^T Z_S,
    b = Z_S^T y (Algorithm 1, line refit). FISTA with exact simplex projection and adaptive restart, then an exact
    KKT solve on the identified support."""
    n = len(b)
    if n == 1:
        return np.ones(1)
    H = G + rho * np.eye(n)
    f = lambda w: w @ H @ w - 2 * b @ w
    step = 1.0 / (2 * max(np.linalg.eigvalsh(H)[-1], 1e-300))
    w = project_simplex(np.asarray(w0, float)) if w0 is not None else np.full(n, 1.0 / n)
    z, t = w.copy(), 1.0
    for _ in range(iters):
        w_new = project_simplex(z - step * 2 * (H @ z - b))
        if np.dot(z - w_new, w_new - w) > 0:  # restart
            t, z = 1.0, w.copy()
            continue
        t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
        z = w_new + (t - 1) / t_new * (w_new - w)
        done = np.linalg.norm(w_new - w) < tol
        w, t = w_new, t_new
        if done:
            break
    A = np.where(w > 1e-12)[0]
    K = np.zeros((len(A) + 1, len(A) + 1))
    K[:-1, :-1], K[:-1, -1], K[-1, :-1] = 2 * H[np.ix_(A, A)], 1.0, 1.0
    sol = np.linalg.lstsq(K, np.concatenate([2 * b[A], [1.0]]), rcond=None)[0][:-1]
    if (sol >= 0).all():
        w_pol = np.zeros(n)
        w_pol[A] = sol
        if f(w_pol) <= f(w) + 1e-12 * abs(f(w)):
            w = w_pol
    return w


def omp_simplex(Z: np.ndarray, budget: int, groups: np.ndarray | None = None, quotas: dict | None = None,
                y: np.ndarray | None = None, score: str = "raw", rho: float = 0.0, trace: bool = False,
                stop: str = "fw") -> tuple[np.ndarray, np.ndarray, list[float]]:
    """OMP variant of Algorithm 1 on atoms Z (N, d) with target y (default: mean atom): add the atom of largest
    correlation with the residual u (within the per-session quotas), refit simplex weights on the whole support, stop
    at the budget or when no atom can lower the error. trace: modal error ||y - Z_S w|| per iteration (Prop. fw).

    stop='fw': an atom lowers ||y - Z w||^2 on the simplex iff <z_i - Z_S w_S, u> > 0 (the Frank-Wolfe gap), i.e.
    <z_i, u> > <Z_S w_S, u>; this is the fully-corrective Frank-Wolfe of Proposition fw. stop='positive' is the
    literal augment filter <z_i, u> > 0 of Algorithm 1, which is the test for unconstrained / non-negative pursuit:
    with uncentred atoms sharing a large common component every correlation turns negative after the first pick, so
    it can return a one-atom buffer. Both choose the same atom whenever they continue. stop='none' (fixed budget)
    also chooses that atom, and keeps going to the budget when no atom lowers the error any more."""
    N = len(Z)
    y = Z.mean(0, dtype=np.float64) if y is None else np.asarray(y, np.float64)
    norms = np.linalg.norm(Z, axis=1) + 1e-30 if score == "normalized" else None
    S, w, b = [], np.zeros(0), np.zeros(0)
    G = np.zeros((0, 0))
    used = np.zeros(N, bool)
    count = defaultdict(int)
    u = y.copy()
    hist = []
    while len(S) < budget:
        corr = (Z @ u.astype(np.float32)).astype(np.float64)
        if stop == "fw":
            thr = float((y - u) @ u) if S else -np.inf
        elif stop == "none":
            thr = -np.inf
        else:
            thr = 0.0
        ok = corr > thr + 1e-9 * (abs(thr) if np.isfinite(thr) else 0.0) + 1e-12 * float(u @ u)
        ok &= ~used
        if quotas is not None:
            for g, q in quotas.items():
                if count[g] >= q:
                    ok &= groups != g
        if not ok.any():
            break
        sc = corr / norms if norms is not None else corr
        i = int(np.argmax(np.where(ok, sc, -np.inf)))
        zi = Z[i].astype(np.float64)
        gi = Z[S].astype(np.float64) @ zi if S else np.zeros(0)
        G = np.block([[G, gi[:, None]], [gi[None, :], np.array([[zi @ zi]])]])
        b = np.append(b, zi @ y)
        S.append(i)
        used[i] = True
        if groups is not None:
            count[groups[i]] += 1
        w = simplex_lsq(G, b, rho, np.append(w, 0.0))
        u = y - Z[S].astype(np.float64).T @ w
        if trace:
            hist.append(float(np.linalg.norm(u)))
    return np.array(S, dtype=int), w, hist


# ----------------------------------------------------------------------------------------------
# Baseline rules (per-session quotas, uniform weights)
# ----------------------------------------------------------------------------------------------
def quotas_for(groups: np.ndarray, budget: float) -> dict:
    return {int(g): int(math.ceil(budget * (groups == g).sum())) for g in np.unique(groups)}


def uniform(idx) -> np.ndarray:
    return np.full(len(idx), 1.0 / max(len(idx), 1))


def random_quota(groups, quotas, rng) -> np.ndarray:
    return np.concatenate([rng.choice(np.where(groups == g)[0], q, replace=False) for g, q in quotas.items()])


def tilted_quota(groups, quotas, cells, rng, alpha: float = 0.5) -> np.ndarray:
    """Per-session sampling with Dirichlet(alpha) weights over metadata cells (e.g. music pieces): realistic
    unbalanced buffers that spread the buffer errors in every representation alike."""
    out = []
    for g, q in quotas.items():
        ids = np.where(groups == g)[0]
        c = np.asarray([cells[i] for i in ids])
        uc, inv = np.unique(c, return_inverse=True)
        wc = rng.dirichlet(np.full(len(uc), alpha))
        p = wc[inv] / np.bincount(inv)[inv]
        p = np.maximum(p, 1e-12)
        out.append(rng.choice(ids, q, replace=False, p=p / p.sum()))
    return np.concatenate(out)


def herding(X: np.ndarray, groups, quotas) -> np.ndarray:
    """iCaRL herding inside each session: greedily keep the running mean closest to the session mean."""
    out = []
    for g, q in quotas.items():
        ids = np.where(groups == g)[0]
        Xg = X[ids].astype(np.float64)
        mu, acc, chosen = Xg.mean(0), np.zeros(X.shape[1]), np.zeros(len(ids), bool)
        for t in range(1, q + 1):
            d = np.linalg.norm(mu * t - acc - Xg, axis=1)
            d[chosen] = np.inf
            j = int(np.argmin(d))
            chosen[j] = True
            acc += Xg[j]
            out.append(ids[j])
    return np.array(out)


def kcenter(X: np.ndarray, groups, quotas) -> np.ndarray:
    """k-center greedy (farthest-first) inside each session, started at the window nearest the session mean."""
    out = []
    for g, q in quotas.items():
        ids = np.where(groups == g)[0]
        Xg = X[ids].astype(np.float64)
        j = int(np.argmin(np.linalg.norm(Xg - Xg.mean(0), axis=1)))
        sel = [j]
        dmin = np.linalg.norm(Xg - Xg[j], axis=1)
        while len(sel) < q:
            j = int(np.argmax(dmin))
            sel.append(j)
            dmin = np.minimum(dmin, np.linalg.norm(Xg - Xg[j], axis=1))
        out += [ids[k] for k in sel]
    return np.array(out)


def top_quota(score: np.ndarray, groups, quotas) -> np.ndarray:
    return np.concatenate([np.where(groups == g)[0][np.argsort(-score[groups == g])[:q]] for g, q in quotas.items()])


# ----------------------------------------------------------------------------------------------
# Standard buffer set
# ----------------------------------------------------------------------------------------------
PHI_RULES = ("tmr", "tmr_uniform", "no_whiten", "random_subspace", "raw_omp", "channel_pca", "herding", "kcenter",
             "random")


def select_buffers(phi: np.ndarray, groups: np.ndarray, budget: float, rules=PHI_RULES, seed: int = 0,
                   ops: dict | None = None, grad_atoms: np.ndarray | None = None,
                   loss: np.ndarray | None = None, score: str = "raw", stop: str = "fw") -> tuple[dict, dict]:
    """Buffers of every rule on one past pool. Returns ({rule: {idx, w, seconds}}, info)."""
    rng = np.random.default_rng(seed)
    quotas = quotas_for(groups, budget)
    B = sum(quotas.values())
    N, T, D = phi.shape
    ops = ops if ops is not None else operators(phi, channel="channel_pca" in rules)
    out, info = {}, {"quotas": quotas, "B": B, "N": N}
    raw = None

    def run(name, fn):
        t0 = time.time()
        idx, w = fn()
        out[name] = {"idx": np.asarray(idx, int), "w": np.asarray(w, float), "seconds": time.time() - t0}

    def omp_on(Z):
        idx, w, _ = omp_simplex(Z, B, groups, quotas, score=score, stop=stop)
        return idx, w

    if {"tmr", "tmr_uniform"} & set(rules):
        Z, zi = modal_dictionary(phi, "temporal", ops=ops)
        info["M_hat"] = zi["M"]
        run("tmr", lambda: omp_on(Z))
        if "tmr_uniform" in rules:
            out["tmr_uniform"] = dict(out["tmr"], w=uniform(out["tmr"]["idx"]))
        del Z
    if "no_whiten" in rules:
        run("no_whiten", lambda: omp_on(modal_dictionary(phi, "temporal", whiten=False, ops=ops)[0]))
    if "random_subspace" in rules:
        run("random_subspace", lambda: omp_on(modal_dictionary(phi, "temporal", basis="random", rng=rng, ops=ops)[0]))
    if "channel_pca" in rules:
        Zc, ci = modal_dictionary(phi, "channel", ops=ops)
        info["M_channel"] = ci["M"]
        run("channel_pca", lambda: omp_on(Zc))
        del Zc
    if {"raw_omp", "herding", "kcenter"} & set(rules):
        raw = phi.reshape(N, -1)
    if "raw_omp" in rules:
        run("raw_omp", lambda: omp_on(raw))
    if "herding" in rules:
        run("herding", lambda: (lambda i: (i, uniform(i)))(herding(raw, groups, quotas)))
    if "kcenter" in rules:
        run("kcenter", lambda: (lambda i: (i, uniform(i)))(kcenter(raw, groups, quotas)))
    if "random" in rules:
        run("random", lambda: (lambda i: (i, uniform(i)))(random_quota(groups, quotas, rng)))
    if grad_atoms is not None:
        run("grad_omp", lambda: omp_on(grad_atoms))
    if loss is not None:
        run("loss_max", lambda: (lambda i: (i, uniform(i)))(top_quota(loss, groups, quotas)))
    return out, info


def mean_error(X: np.ndarray, idx, w, target: np.ndarray | None = None) -> float:
    """|| mean_i x_i - sum_{i in B} w_i x_i || (Frobenius for trajectories): eps_B on raw trajectories,
    eps_mod on modal atoms, delta on gradient sketches."""
    target = X.mean(0, dtype=np.float64) if target is None else target
    return float(np.linalg.norm(target - np.asarray(w, np.float64) @ X[np.asarray(idx, int)].astype(np.float64)))


def composition(meta: list[dict], groups: np.ndarray, idx) -> dict:
    """How a buffer covers the past pool's metadata: share of distinct music pieces / dancers / genres it contains
    and the normalised entropy of its music-piece histogram (1 = as even as possible)."""
    idx = np.asarray(idx, int)
    out = {}
    for key in ("music", "dancer", "genre"):
        vals = [m.get(key) for m in meta]
        if vals[0] is None:
            continue
        pool_set, buf = set(vals), [vals[i] for i in idx]
        out[f"{key}_coverage"] = len(set(buf)) / len(pool_set)
        _, c = np.unique(buf, return_counts=True)
        p = c / c.sum()
        out[f"{key}_entropy"] = float(-(p * np.log(p)).sum() / math.log(max(len(pool_set), 2)))
    return out


def grid_coverage(X2: np.ndarray, idx, groups: np.ndarray | None = None, bins: int = 20) -> float:
    """Paper Fig. 3c metric: share of occupied cells of a bins x bins grid over the 2-D PCA projection of the pool
    that the buffer also occupies (averaged over sessions when groups is given)."""
    def cov(rows, sel):
        lo, hi = X2[rows].min(0), X2[rows].max(0) + 1e-12
        cell = lambda r: set(map(tuple, np.floor((X2[r] - lo) / (hi - lo) * bins).clip(0, bins - 1).astype(int)))
        full = cell(rows)
        return len(cell(sel) & full) / len(full)

    idx = np.asarray(idx, int)
    if groups is None:
        return cov(np.arange(len(X2)), idx)
    return float(np.mean([cov(np.where(groups == g)[0], idx[groups[idx] == g]) for g in np.unique(groups)]))
