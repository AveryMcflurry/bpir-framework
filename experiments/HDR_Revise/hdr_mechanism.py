#!/usr/bin/env python3
"""
Hierarchical Demographic Reweighting (HDR) — attribute-weighting formulation
============================================================================

A cluster-based correction mechanism that adapts a transferred activity-based
model **without modifying the input population**. Instead of rewriting person
records (role reassignment, age offsets, gender flips, relocation), HDR learns
**per-cluster multiplicative weights on input attributes**. A weight rides into
the simulation wherever its attribute enters a utility, so the attribute's
effective contribution is scaled — which is mathematically a *grouped
coefficient update* (``beta * (w * x) = (beta * w) * x``) while every estimated
coefficient and every individual record is left untouched.

Design principles
-----------------
1. **Input is input.** Person records are never altered. Household demographic
   values are never overwritten. HDR only *appends* per-household weight columns
   (declared model parameters, carried alongside the data like a coefficient
   file). ActivitySim applies them at the annotate/preprocessor step
   (``effective_attr = raw_attr * weight``); the raw columns stay pristine.
2. **Deterministic.** Given a weight vector, the transformation is exact — no
   stochastic draws — so the optimisation landscape is reproducible.
3. **Schema-valid by construction.** Because ``ptype/pemploy/pstudent/age`` are
   never touched, ActivitySim cannot reject on invalid person roles, and no
   "snap-to-nearest-role" guard is needed.

Weighted attributes (per cluster)
---------------------------------
Spanning three kinds and three sub-model channels (see paper §3.2):
    income       household economic (continuous)           -> auto-own, VOT, tour-freq
    workers      composition magnitude (linear count)      -> auto-own, mode, trip rate
    children     composition magnitude (linear count)      -> tour-freq, purpose
    nonworkers   composition magnitude (linear count)      -> tour-freq, trip rate
    has_senior   composition presence (binary)             -> non-mandatory/joint, purpose

Three genuinely distinct optimisation paradigms drive the search:
    CMAESEngine     - evolution strategy (the ``cma`` library)
    PPOEngine       - policy-gradient RL with a clipped surrogate objective
    LSTMMetaEngine  - a meta-learned recurrent optimiser (learned update rule)

This module exposes the algorithmic components. Users supply their own
population tables, ActivitySim runner, and observed targets.

License: MIT
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

FAILURE_SCORE: float = -10.0

# ---------------------------------------------------------------------------
# Role lattice (reference only — HDR no longer reassigns roles)
# ---------------------------------------------------------------------------

Role = namedtuple("Role", "ptype pemploy pstudent min_age max_age")

ROLE_TABLE: List[Role] = [
    Role(8, 3, 1,   0,   5),   # preschool
    Role(7, 3, 1,   6,  15),   # non-driving student
    Role(7, 4, 1,   6,  15),   # student <16 (work status variant)
    Role(6, 3, 1,  16,  17),   # driving-age student
    Role(6, 4, 1,  16,  17),   # driving-age student (work variant)
    Role(1, 1, 3,  18,  64),   # full-time worker
    Role(2, 2, 3,  18,  64),   # part-time worker
    Role(4, 3, 3,  18,  64),   # non-working adult
    Role(3, 1, 2,  18,  64),   # university student + FT work
    Role(3, 2, 2,  18,  64),   # university student + PT work
    Role(3, 3, 2,  18,  64),   # university student, no work
    Role(5, 3, 3,  65, 110),   # retired
    Role(1, 1, 3,  65, 110),   # senior FT worker
    Role(2, 2, 3,  65, 110),   # senior PT worker
]
"""The 14-role person-type lattice. Retained for schema documentation; the
attribute-weighting mechanism never modifies a person's role, so every record
stays inside this lattice by construction."""


# ---------------------------------------------------------------------------
# Weighted-attribute specification
# ---------------------------------------------------------------------------

DEFAULT_ATTRS: Tuple[str, ...] = ("income", "vot")
"""Per-household-cluster weighted attributes. Empirical leverage screening retained
only the loud, safe ones: ``income`` (broad — trip rate, purpose, mode via
segment/auto-ownership/VOT) and ``vot`` (mode cost sensitivity). The composition
counts (workers/children/nonworkers/has_senior) were dropped as weak or dead."""

DEFAULT_PERSON_ATTRS: Tuple[str, ...] = ("work", "nonmand")
"""Per-household-cluster weighted PERSON attributes (ride into the sim via persons.csv,
not households) — the PURPOSE levers, which land-use/income/skim weights cannot move.
``work`` = worker-rate weight (``cdap_work_w = is_worker*lw`` -> workers' Mandatory CDAP
utility -> work tours); saturates ~85.6 alone. ``nonmand`` = non-mandatory propensity
weight (``cdap_nonmand_w = lw`` on all persons -> NonMandatory CDAP utility); DOWN it
degrades purpose past the worker floor (fewer discretionary tours -> work-heavier mix).
Together they reach the paper purpose ~80.7. (``student`` Mandatory was screened DEAD —
students already saturate school; its column stays inert.) The '=' lands inside CDAP;
raw ptype/pemploy never edited."""

DEFAULT_LU_ATTRS: Tuple[str, ...] = ("parking", "terminal", "employment")
"""Global (per-zone-uniform) land-use weights — the loud levers. ``parking``
(PRKCST/OPRKCST -> auto cost -> mode, ~13pp driver swing), ``terminal``
(TERMINAL -> auto access time -> mode), ``employment`` (TOTEMP+sectors ->
accessibility & size -> trip rate / purpose). They correct the transferred
SF land-use parameters for the application region."""

DEFAULT_SKIM_ATTRS: Tuple[str, ...] = ("hovtime", "autocost", "autotime", "transit", "nonmotor")
"""Global (per-network) skim weights — bounded corrections to the transferred SF
level-of-service (the strongest transferability argument; skims are an input like any
other). Each scales a set of OMX cores, riding into mode choice during the sim; the
raw skims.omx is NEVER overwritten — a weighted COPY is written into the run dir.
``hovtime`` (shared-ride in-vehicle time -> passenger share), ``autocost`` (auto
distance/tolls -> driver vs ride-hailing), ``autotime`` (SOV time -> driver share).
These are THE mode lever — empirically they swing the mode component 42 <-> 21."""

SKIM_CORE_PATTERNS = {
    "hovtime":  r"^(HOV2|HOV3)_TIME__",
    "autocost": r"^(SOV|HOV2|HOV3)_(DIST|BTOLL|VTOLL)__",
    "autotime": r"^SOV_TIME__",
    "transit":  r"(TOTIVT|KEYIVT|_FAR__|IWAIT|_WAIT__|_WAUX)",   # transit LOS -> transit share
    "nonmotor": r"^(DISTWALK|DISTBIKE)$",                        # walk/bike distance -> active modes
}


@dataclass
class HDRConfig:
    """Tunable parameters for attribute weighting."""
    attrs: Tuple[str, ...] = DEFAULT_ATTRS                 # per household-cluster
    person_attrs: Tuple[str, ...] = DEFAULT_PERSON_ATTRS   # per cluster, applied on persons
    lu_attrs: Tuple[str, ...] = DEFAULT_LU_ATTRS           # global land-use
    skim_attrs: Tuple[str, ...] = DEFAULT_SKIM_ATTRS       # global skims (level-of-service)
    # plausibility bounds on the *log* weight
    # bounds RE-TIGHTENED 2026-06-24 to PLAUSIBLE calibration magnitudes — the wide
    # 2026-06-21 box let the forward overfit (skims /7 -> composite 93, implausible).
    # base restore (faithful57) is evaluated under its own RESTORE_BOUNDS, untouched.
    weight_log_clip: Tuple[float, float] = (-0.8, 0.8)       # household: exp -> ~[0.45, 2.23]
    work_log_clip: Tuple[float, float] = (-1.6, 1.6)         # worker/nonmand: log-odds on CDAP
    lu_weight_log_clip: Tuple[float, float] = (-1.2, 1.2)    # land use: exp -> ~[0.30, 3.32] (also optimizer box)
    skim_log_clip: Tuple[float, float] = (-0.7, 0.7)         # skims: exp -> ~[0.50, 2.01] (plausible LOS calibration)
    # household clustering
    cluster_features: Tuple[str, ...] = (
        "income_k", "hhsize", "num_workers", "num_children", "num_seniors")
    random_state: int = 42


# ---------------------------------------------------------------------------
# Household-level clustering (pairs naturally with household attribute weights)
# ---------------------------------------------------------------------------

def _household_features(
    persons: pd.DataFrame, households: pd.DataFrame, features: Sequence[str]
) -> pd.DataFrame:
    """Build household-level demographic features for clustering.

    Derives, where needed, counts from the (immutable) persons table. Never
    writes anything back to the source tables.
    """
    h = households.copy()
    hid = persons["household_id"]
    worker_flag = persons["pemploy"].isin([1, 2]).astype(int)
    child_flag = (persons["age"] < 18).astype(int)
    senior_flag = (persons["age"] >= 65).astype(int)
    g_workers = worker_flag.groupby(hid).sum()
    g_children = child_flag.groupby(hid).sum()
    g_seniors = senior_flag.groupby(hid).sum()
    g_size = hid.groupby(hid).size()

    derived = {
        "income_k": (h["income"].clip(lower=0) / 1000.0) if "income" in h.columns
        else pd.Series(0.0, index=h.index),
        "num_workers": h["num_workers"] if "num_workers" in h.columns else g_workers,
        "num_children": g_children,
        "num_seniors": g_seniors,
        "hhsize": h["hhsize"] if "hhsize" in h.columns else g_size,
    }
    out = pd.DataFrame(index=h.index)
    for f in features:
        if f in h.columns:
            out[f] = h[f].values
        elif f in derived:
            col = derived[f]
            out[f] = (col.reindex(h.index).fillna(0).values
                      if isinstance(col, pd.Series) else col)
        else:
            out[f] = 0.0
    return out.fillna(0.0)


def cluster_households(
    persons: pd.DataFrame,
    households: pd.DataFrame,
    k_min: int,
    k_max: int,
    *,
    config: Optional[HDRConfig] = None,
) -> Tuple[pd.Series, int]:
    """Assign households to demographic clusters via KMeans + silhouette.

    Returns ``(cluster_labels indexed like households, best_k)``.
    """
    cfg = config or HDRConfig()
    X = _household_features(persons, households, cfg.cluster_features)
    Xz = (X - X.mean()) / (X.std().replace(0, 1.0))
    best_k, best_sc, best_labels = k_min, -1.0, None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=10)
        labels = km.fit_predict(Xz)
        if len(np.unique(labels)) < 2:
            continue
        sc = silhouette_score(Xz, labels,
                              sample_size=min(5000, len(Xz)),
                              random_state=cfg.random_state)
        if sc > best_sc:
            best_k, best_sc, best_labels = k, sc, labels
    if best_labels is None:  # degenerate fallback
        best_labels = np.zeros(len(households), dtype=int)
        best_k = 1
    return pd.Series(best_labels, index=households.index, name="cluster"), best_k


# ---------------------------------------------------------------------------
# Core HDR: attribute weighting (no record modification)
# ---------------------------------------------------------------------------

def apply_attribute_weights(
    persons: pd.DataFrame,
    households: pd.DataFrame,
    w_flat: np.ndarray,
    k: int,
    *,
    cluster_col: str = "cluster",
    config: Optional[HDRConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Attach per-household attribute weights from a learned weight vector.

    The persons table is returned **unchanged**. The households table is returned
    with the raw demographic columns **unchanged** and one appended column per
    weighted attribute (``w_<attr>``), holding that household's cluster weight.
    ActivitySim consumes these via ``keep_columns`` and applies them in the
    annotate step (``effective = raw * w_<attr>``).

    Parameters
    ----------
    persons : DataFrame      (must contain ``household_id``; otherwise untouched)
    households : DataFrame    (must contain ``cluster_col``)
    w_flat : ndarray          flat vector, length ``k * len(config.attrs)``
    k : int                   number of clusters
    cluster_col : str         household cluster-id column

    Returns
    -------
    persons_out, households_out, diagnostics
    """
    cfg = config or HDRConfig()
    n_attr = len(cfg.attrs)
    assert len(w_flat) == k * n_attr, (
        f"weight vector length {len(w_flat)} != k*n_attr = {k * n_attr}")
    if cluster_col not in households.columns:
        raise KeyError(
            f"households is missing the '{cluster_col}' column; "
            f"run cluster_households() first.")

    lo, hi = cfg.weight_log_clip
    wm = w_flat.reshape(k, n_attr)
    mult = np.exp(np.clip(wm, lo, hi))            # (k, n_attr) bounded multipliers

    p = persons                                    # untouched (no copy needed)
    h = households.copy()                          # only to append weight columns
    cl = h[cluster_col].to_numpy().astype(int)
    cl = np.clip(cl, 0, k - 1)

    for j, attr in enumerate(cfg.attrs):
        h[f"w_{attr}"] = mult[cl, j]

    diag = {
        "mult_mean": {a: float(mult[:, j].mean()) for j, a in enumerate(cfg.attrs)},
        "mult_min": float(mult.min()),
        "mult_max": float(mult.max()),
        "n_clusters": int(k),
    }
    return p, h, diag


def composition_drift(
    persons: pd.DataFrame,
    households: pd.DataFrame,
    *,
    config: Optional[HDRConfig] = None,
) -> Dict[str, float]:
    """Report how far the *effective* weighted marginals drift from the raw ones.

    Demographic-fidelity diagnostic: shows the optimiser scaled attribute
    influence within bounds rather than distorting the population. Returns the
    weighted/raw ratio of each attribute's aggregate (1.0 == no drift).
    """
    cfg = config or HDRConfig()
    out: Dict[str, float] = {}
    feats = _household_features(persons, households, cfg.cluster_features)
    base = {
        "income": feats["income_k"], "workers": feats["num_workers"],
        "children": feats["num_children"], "nonworkers": feats["hhsize"] - feats["num_workers"],
        "has_senior": (feats["num_seniors"] > 0).astype(float),
    }
    for attr in cfg.attrs:
        wcol = f"w_{attr}"
        if wcol not in households or attr not in base:
            continue
        raw = base[attr].to_numpy()
        w = households[wcol].to_numpy()
        denom = raw.sum()
        out[attr] = float((raw * w).sum() / denom) if denom else 1.0
    return out


# ===========================================================================
# Optimiser interface
# ===========================================================================

class Optimiser(ABC):
    """Black-box maximiser. ``ask`` proposes candidates; ``tell`` reports their
    scores (higher == better)."""

    @abstractmethod
    def ask(self) -> List[np.ndarray]: ...
    @abstractmethod
    def tell(self, solutions: List[np.ndarray], scores: List[float]) -> None: ...
    @property
    @abstractmethod
    def best_solution(self) -> Tuple[np.ndarray, float]: ...
    @property
    @abstractmethod
    def converged(self) -> bool: ...


# ---------------------------------------------------------------------------
# 1) CMA-ES  (genuine evolution strategy via the ``cma`` library)
# ---------------------------------------------------------------------------

class CMAESEngine(Optimiser):
    def __init__(self, dim: int, popsize: int, *, bounds=(-1.0, 1.0),
                 sigma0: float = 0.3, seed: int = 42, maxiter: int = 300, x0=None):
        import cma
        self._popsize = max(popsize, 4 + int(3 * np.log(dim)))
        opts = {"popsize": self._popsize, "maxiter": maxiter,
                "bounds": [bounds[0], bounds[1]], "seed": seed, "verbose": -9}
        mean0 = [0.0] * dim if x0 is None else list(np.asarray(x0, dtype=float))
        self._es = cma.CMAEvolutionStrategy(mean0, sigma0, opts)
        self._best = (np.zeros(dim), FAILURE_SCORE)
        self._hist: deque = deque(maxlen=12)

    def ask(self):
        return [np.asarray(x, dtype=float) for x in self._es.ask()]

    def tell(self, solutions, scores):
        # cma minimises -> feed negative score as loss
        self._es.tell([np.asarray(s).tolist() for s in solutions],
                      [-float(v) for v in scores])
        bi = int(np.argmax(scores))
        if scores[bi] > self._best[1]:
            self._best = (np.asarray(solutions[bi]).copy(), float(scores[bi]))
        self._hist.append(self._best[1])

    @property
    def best_solution(self):
        return self._best

    @property
    def converged(self):
        return bool(self._es.stop())


# ---------------------------------------------------------------------------
# 2) PPO  (genuine policy-gradient RL with a clipped surrogate)
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """Hyperparameters for the PPO engine."""
    clip_eps: float = 0.2          # PPO clipping epsilon
    lr: float = 0.05               # policy step size
    epochs: int = 6                # update epochs per iteration
    minibatch: int = 16
    init_logstd: float = -1.0      # exp(-1) ~ 0.37 initial exploration
    logstd_bounds: Tuple[float, float] = (-2.5, 0.4)
    entropy_coef: float = 0.01     # exploration bonus
    value_lr: float = 0.05         # critic step size
    grad_clip: float = 1.0
    bounds: Tuple[float, float] = (-1.0, 1.0)
    seed: int = 42


class PPOEngine(Optimiser):
    """Proximal Policy Optimisation for black-box weight search.

    Action  = a candidate weight vector, drawn from a Gaussian policy
              ``pi(a) = N(mu, diag(sigma^2))``.
    Reward  = composite score returned by the (expensive) objective.
    The population at each iteration is a batch of rollouts. Updates use the
    PPO **clipped surrogate** with **advantages** from a learned linear value
    baseline, plus an **entropy bonus** — the defining ingredients of PPO that
    distinguish it from an evolution strategy.
    """

    _N_STATE = 3  # [iter_frac, best_score_norm, recent_improvement]

    def __init__(self, dim: int, popsize: int, config: Optional[PPOConfig] = None):
        c = config or PPOConfig()
        self._dim, self._pop, self._cfg = dim, popsize, c
        self._rng = np.random.default_rng(c.seed)
        self._mu = np.zeros(dim)
        self._logstd = np.full(dim, c.init_logstd)
        self._best = (np.zeros(dim), FAILURE_SCORE)
        # linear critic V(s) = w_v . s + b_v
        self._wv = np.zeros(self._N_STATE)
        self._bv = 0.0
        self._iter = 0
        self._maxiter_guess = 200
        self._scores_hist: deque = deque(maxlen=10)
        self._last_actions: Optional[np.ndarray] = None
        self._converged = False
        self._stall = 0

    # -- state features (the critic's input) --
    def _state(self) -> np.ndarray:
        best = self._best[1]
        best_n = np.tanh(best / 100.0)
        if len(self._scores_hist) >= 4:
            imp = self._scores_hist[-1] - self._scores_hist[-4]
        else:
            imp = 0.0
        return np.array([min(self._iter / self._maxiter_guess, 1.0),
                         best_n, np.tanh(imp / 10.0)])

    def ask(self):
        std = np.exp(self._logstd)
        acts = self._mu[None, :] + self._rng.normal(size=(self._pop, self._dim)) * std[None, :]
        acts = np.clip(acts, *self._cfg.bounds)
        self._last_actions = acts
        return [acts[i].copy() for i in range(self._pop)]

    def _logprob(self, a, mu, logstd):
        std = np.exp(logstd)
        return -0.5 * (((a - mu) / std) ** 2 + 2 * logstd + np.log(2 * np.pi)).sum(axis=-1)

    def tell(self, solutions, scores):
        c = self._cfg
        a = np.asarray(self._last_actions if self._last_actions is not None
                       else solutions, dtype=float)
        r = np.asarray(scores, dtype=float)
        self._iter += 1
        bi = int(np.argmax(r))
        if r[bi] > self._best[1]:
            self._best = (a[bi].copy(), float(r[bi]))
            self._stall = 0
        else:
            self._stall += 1
        self._scores_hist.append(float(r[bi]))

        # --- critic: advantage = reward - V(s) ---
        s = self._state()
        v = float(self._wv @ s + self._bv)
        adv = r - v
        if adv.std() > 1e-8:
            adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        else:
            adv_n = adv - adv.mean()

        mu_old, logstd_old = self._mu.copy(), self._logstd.copy()
        logp_old = self._logprob(a, mu_old, logstd_old)

        idx = np.arange(self._pop)
        for _ in range(c.epochs):
            self._rng.shuffle(idx)
            for start in range(0, self._pop, c.minibatch):
                mb = idx[start:start + c.minibatch]
                am, advm, lpo = a[mb], adv_n[mb], logp_old[mb]
                std = np.exp(self._logstd)
                logp = self._logprob(am, self._mu, self._logstd)
                ratio = np.exp(np.clip(logp - lpo, -10, 10))
                # clipped-surrogate mask (gradient flows only on the active branch)
                clipped_hi = (advm >= 0) & (ratio > 1 + c.clip_eps)
                clipped_lo = (advm < 0) & (ratio < 1 - c.clip_eps)
                mask = (~(clipped_hi | clipped_lo)).astype(float)
                coef = mask * advm * ratio              # (mb,)
                # d logpi / d mu , d logpi / d logstd
                dmu = (am - self._mu) / std ** 2          # (mb, dim)
                dls = ((am - self._mu) ** 2) / std ** 2 - 1.0
                g_mu = (coef[:, None] * dmu).mean(axis=0)
                g_ls = (coef[:, None] * dls).mean(axis=0) + c.entropy_coef  # entropy bonus
                # gradient ASCENT on the surrogate
                g_mu = np.clip(g_mu, -c.grad_clip, c.grad_clip)
                g_ls = np.clip(g_ls, -c.grad_clip, c.grad_clip)
                self._mu = np.clip(self._mu + c.lr * g_mu, *c.bounds)
                self._logstd = np.clip(self._logstd + c.lr * g_ls, *c.logstd_bounds)

        # --- critic regression toward observed mean reward ---
        target = float(r.mean())
        pred = float(self._wv @ s + self._bv)
        err = pred - target
        self._wv -= c.value_lr * err * s
        self._bv -= c.value_lr * err

        if self._stall >= 25:
            self._converged = True

    @property
    def best_solution(self):
        return self._best

    @property
    def converged(self):
        return self._converged


# ---------------------------------------------------------------------------
# 3) LSTM meta-learned optimiser (a learned, recurrent update rule)
# ---------------------------------------------------------------------------

@dataclass
class LSTMMetaConfig:
    hidden: int = 8
    n_features: int = 5     # [pop_grad, best_dir, logstd, score_trend, iter_frac]
    mu_scale: float = 0.10  # output scaling for the mean update
    ls_scale: float = 0.10  # output scaling for the logstd update
    init_logstd: float = -1.0
    logstd_bounds: Tuple[float, float] = (-2.5, 0.5)
    bounds: Tuple[float, float] = (-1.0, 1.0)
    seed: int = 42


def _lstm_param_shapes(cfg: LSTMMetaConfig):
    H, F = cfg.hidden, cfg.n_features
    return {"W": (H + F, 4 * H), "b": (4 * H,), "Wo": (H, 2), "bo": (2,)}


def lstm_param_dim(cfg: LSTMMetaConfig) -> int:
    return sum(int(np.prod(s)) for s in _lstm_param_shapes(cfg).values())


def init_lstm_params(cfg: LSTMMetaConfig, rng: np.random.Generator) -> np.ndarray:
    H = cfg.hidden
    shapes = _lstm_param_shapes(cfg)
    parts = []
    W = rng.standard_normal(shapes["W"]) * 0.1
    b = np.zeros(shapes["b"]); b[:H] = 1.0          # forget-gate bias = 1
    Wo = rng.standard_normal(shapes["Wo"]) * 0.05
    bo = np.zeros(shapes["bo"])
    for arr in (W, b, Wo, bo):
        parts.append(arr.ravel())
    return np.concatenate(parts)


def _unpack_lstm(phi: np.ndarray, cfg: LSTMMetaConfig):
    H, F = cfg.hidden, cfg.n_features
    shapes = _lstm_param_shapes(cfg)
    i = 0
    out = {}
    for name, shp in shapes.items():
        n = int(np.prod(shp))
        out[name] = phi[i:i + n].reshape(shp)
        i += n
    return out["W"], out["b"], out["Wo"], out["bo"]


class LSTMMetaEngine(Optimiser):
    """A coordinate-wise LSTM that *is* the optimiser.

    At each iteration the same small LSTM (shared weights across dimensions)
    reads per-coordinate features — a population-based gradient estimate, the
    direction to the best point so far, the current log-std, the score trend and
    progress — and emits the next search distribution (mean update + log-std),
    carrying hidden state across iterations. Its weights ``phi`` are *meta-learned*
    on a family of objective functions (see ``meta_train_lstm_optimizer``), so it
    embodies a learned update rule rather than a hand-tuned heuristic.
    """

    def __init__(self, dim: int, popsize: int, phi: np.ndarray,
                 config: Optional[LSTMMetaConfig] = None, *, seed: int = 42):
        self._cfg = config or LSTMMetaConfig()
        self._dim, self._pop = dim, popsize
        self._phi = np.asarray(phi, dtype=float)
        self._rng = np.random.default_rng(seed)
        self._reset_state()
        self._best = (self._mu.copy(), FAILURE_SCORE)
        self._scores_hist: deque = deque(maxlen=10)
        self._iter = 0
        self._maxiter_guess = 200
        self._last_actions: Optional[np.ndarray] = None
        self._converged = False
        self._stall = 0

    def _reset_state(self):
        H = self._cfg.hidden
        self._mu = np.zeros(self._dim)
        self._logstd = np.full(self._dim, self._cfg.init_logstd)
        self._h = np.zeros((self._dim, H))
        self._c = np.zeros((self._dim, H))

    def ask(self):
        std = np.exp(self._logstd)
        acts = self._mu[None, :] + self._rng.normal(size=(self._pop, self._dim)) * std[None, :]
        acts = np.clip(acts, *self._cfg.bounds)
        self._last_actions = acts
        return [acts[i].copy() for i in range(self._pop)]

    def _features(self, a, r):
        """Per-coordinate input features (dim x n_features)."""
        std = np.exp(self._logstd)
        rs = r - r.mean()
        if r.std() > 1e-8:
            rs = rs / (r.std() + 1e-8)
        # population (score-function) gradient estimate per coordinate
        g = ((a - self._mu[None, :]) / std[None, :] * rs[:, None]).mean(axis=0)
        g = np.tanh(g)
        best_dir = np.tanh(self._best[0] - self._mu)
        if len(self._scores_hist) >= 4:
            trend = np.tanh((self._scores_hist[-1] - self._scores_hist[-4]) / 10.0)
        else:
            trend = 0.0
        iter_frac = min(self._iter / self._maxiter_guess, 1.0)
        F = self._cfg.n_features
        feat = np.zeros((self._dim, F))
        feat[:, 0] = g
        feat[:, 1] = best_dir
        feat[:, 2] = np.tanh(self._logstd)
        feat[:, 3] = trend
        feat[:, 4] = iter_frac
        return feat

    def _lstm_step(self, feat):
        cfg = self._cfg
        W, b, Wo, bo = _unpack_lstm(self._phi, cfg)
        inp = np.concatenate([self._h, feat], axis=1)        # (dim, H+F)
        z = inp @ W + b                                       # (dim, 4H)
        H = cfg.hidden
        zf, zi, zg, zo = z[:, :H], z[:, H:2*H], z[:, 2*H:3*H], z[:, 3*H:]
        f = 1.0 / (1.0 + np.exp(-np.clip(zf, -30, 30)))
        i = 1.0 / (1.0 + np.exp(-np.clip(zi, -30, 30)))
        g = np.tanh(zg)
        o = 1.0 / (1.0 + np.exp(-np.clip(zo, -30, 30)))
        self._c = np.clip(f * self._c + i * g, -20, 20)
        self._h = o * np.tanh(self._c)
        out = self._h @ Wo + bo                               # (dim, 2)
        return out[:, 0], out[:, 1]

    def tell(self, solutions, scores):
        cfg = self._cfg
        a = np.asarray(self._last_actions if self._last_actions is not None
                       else solutions, dtype=float)
        r = np.asarray(scores, dtype=float)
        self._iter += 1
        bi = int(np.argmax(r))
        if r[bi] > self._best[1]:
            self._best = (a[bi].copy(), float(r[bi]))
            self._stall = 0
        else:
            self._stall += 1
        self._scores_hist.append(float(r[bi]))

        feat = self._features(a, r)
        du, dls = self._lstm_step(feat)
        self._mu = np.clip(self._mu + cfg.mu_scale * du, *cfg.bounds)
        self._logstd = np.clip(self._logstd + cfg.ls_scale * dls, *cfg.logstd_bounds)
        if self._stall >= 30:
            self._converged = True

    @property
    def best_solution(self):
        return self._best

    @property
    def converged(self):
        return self._converged


def meta_train_lstm_optimizer(
    objective_sampler: Callable[[np.random.Generator], Callable[[np.ndarray], float]],
    dim: int,
    *,
    config: Optional[LSTMMetaConfig] = None,
    popsize: int = 16,
    inner_steps: int = 30,
    meta_iters: int = 40,
    es_popsize: int = 24,
    es_sigma: float = 0.1,
    es_lr: float = 0.05,
    batch_funcs: int = 4,
    seed: int = 0,
    log_every: int = 5,
) -> Tuple[np.ndarray, List[float]]:
    """Meta-train the LSTM optimiser's weights ``phi`` with antithetic ES.

    The meta-objective is the mean best-score the optimiser reaches after
    ``inner_steps`` on a freshly sampled batch of objective functions. Optimising
    ``phi`` to maximise it yields a learned update rule that transfers to unseen
    objectives. Returns ``(phi_star, meta_score_history)``.
    """
    cfg = config or LSTMMetaConfig()
    rng = np.random.default_rng(seed)
    phi = init_lstm_params(cfg, rng)
    pdim = phi.size

    def rollout(phi_vec, fns) -> float:
        bests = []
        for fi, fn in enumerate(fns):
            eng = LSTMMetaEngine(dim, popsize, phi_vec, cfg, seed=1000 + fi)
            eng._maxiter_guess = inner_steps
            for _ in range(inner_steps):
                cands = eng.ask()
                scores = [fn(x) for x in cands]
                eng.tell(cands, scores)
            bests.append(eng.best_solution[1])
        return float(np.mean(bests))

    history: List[float] = []
    n_pairs = es_popsize // 2
    for it in range(meta_iters):
        fns = [objective_sampler(rng) for _ in range(batch_funcs)]
        eps = rng.standard_normal((n_pairs, pdim))
        returns = np.zeros(es_popsize)
        for kk in range(n_pairs):
            returns[2 * kk] = rollout(phi + es_sigma * eps[kk], fns)
            returns[2 * kk + 1] = rollout(phi - es_sigma * eps[kk], fns)
        # rank-normalise for a robust gradient estimate
        ranks = np.argsort(np.argsort(returns)).astype(float)
        ranks = ranks / (es_popsize - 1) - 0.5
        grad = np.zeros(pdim)
        for kk in range(n_pairs):
            grad += (ranks[2 * kk] - ranks[2 * kk + 1]) * eps[kk]
        grad /= (n_pairs * es_sigma)
        phi = phi + es_lr * grad
        center = rollout(phi, fns)
        history.append(center)
        if log_every and (it % log_every == 0 or it == meta_iters - 1):
            logger.info("meta-iter %3d/%d  center best-score=%.2f", it, meta_iters, center)
    return phi, history


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_optimiser(
    name: str, dim: int, popsize: int, *,
    lstm_phi: Optional[np.ndarray] = None,
    lstm_config: Optional[LSTMMetaConfig] = None,
    ppo_config: Optional[PPOConfig] = None,
    **kw,
) -> Optimiser:
    """Instantiate ``"cmaes"``, ``"ppo"`` (alias ``"rl"``), or ``"lstm"``.

    The ``"lstm"`` engine requires a meta-trained ``lstm_phi`` (see
    ``meta_train_lstm_optimizer``).
    """
    n = name.lower().strip()
    if n == "cmaes":
        return CMAESEngine(dim, popsize, **kw)
    if n in ("ppo", "rl"):
        return PPOEngine(dim, popsize, config=ppo_config)
    if n == "lstm":
        if lstm_phi is None:
            raise ValueError("lstm engine needs meta-trained lstm_phi")
        return LSTMMetaEngine(dim, popsize, lstm_phi, config=lstm_config)
    raise ValueError(f"Unknown optimiser: {name!r}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info(
        "HDR attribute-weighting module loaded. Core components: "
        "cluster_households, apply_attribute_weights, composition_drift, "
        "CMAESEngine, PPOEngine, LSTMMetaEngine, meta_train_lstm_optimizer.")
