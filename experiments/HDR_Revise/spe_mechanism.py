#!/usr/bin/env python3
"""
Segmentation-Based Population Exploration (SPE)
=================================================

A group-based population scaling mechanism for optimising the composition of
an activity-based model's input population. Households are partitioned into
*k* groups via deterministic hashing, and per-group scaling weights control
replication (up-sampling) or removal (down-sampling) of entire household
units — without modifying any individual-level attributes.

Methodology described in:

    Ma, Y., Rashidi, T.H., Najmi, A. (2026). Activity-Based Model Correction
    Mechanism Development: A Framework for Optimizing Input Composition Weights
    to Improve Spatial Transferability. Transportation Research Part C.

Three optimisation engines are provided:
    CMA-ES
    LSTM policy network
    PPO-style RL

This module exposes the core algorithmic components. Users must supply
their own population tables, simulation runner, and observed targets.
No proprietary data, model weights, or survey records are distributed.

Requirements:
    numpy >= 1.24
    pandas >= 2.0
    cma (optional, for CMA-ES engine)

License: MIT
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FAILURE_SCORE: float = -10.0


# ---------------------------------------------------------------------------
# Parameter encoding / decoding
# ---------------------------------------------------------------------------

def decode_parameters(
    raw: Sequence[float],
    k_min: int,
    k_max: int,
) -> Tuple[int, int, np.ndarray]:
    """Decode a raw parameter vector into (k, hash_seed, weights).

    Parameters
    ----------
    raw : array-like
        At least two elements: ``[k_norm, seed_norm, w_1, …, w_kmax]``.
    k_min, k_max : int
        Bounds for the number of groups.

    Returns
    -------
    k : int
        Number of groups.
    seed : int
        Hash seed for group assignment.
    weights : ndarray
        Softmax-normalised group scaling weights of length *k*.
    """
    if len(raw) < 2:
        raise ValueError("Parameter vector must have at least 2 elements")

    k_norm, seed_norm = raw[0], raw[1]
    k_frac = (k_norm + 1.0) / 2.0   # map [-1,1] → [0,1]
    k = int(np.clip(k_min + k_frac * (k_max - k_min), k_min, k_max))
    seed = int(abs(seed_norm) * 999_999) % 1_000_000

    weight_raw = list(raw[2:])
    while len(weight_raw) < k:
        weight_raw.append(0.0)
    w = np.array(weight_raw[:k])
    w = np.exp(np.clip(w, -0.8, 0.8))
    w /= w.sum()

    return k, seed, w


# ---------------------------------------------------------------------------
# Hash-based grouping
# ---------------------------------------------------------------------------

def hash_group(household_ids: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Assign each household to one of *k* groups deterministically."""
    salt = hashlib.md5(str(seed).encode()).hexdigest()[:8]
    return np.array([
        int(hashlib.sha256(f"{int(hid)}_{salt}".encode()).hexdigest()[:8], 16) % k
        for hid in household_ids
    ])


# ---------------------------------------------------------------------------
# Population scaling with UUID-based ID management
# ---------------------------------------------------------------------------

def scale_population(
    persons: pd.DataFrame,
    households: pd.DataFrame,
    weights: np.ndarray,
    groups: np.ndarray,
    *,
    zone_col: Optional[str] = "TAZ",
    rng_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scale population by group weights via sampling / replication.
    Down-weighted groups are subsampled (preserving spatial proportions
    when *zone_col* is available). Up-weighted groups are replicated with
    fresh UUID-based household and person IDs to avoid duplicates.
    """
    rng = np.random.default_rng(rng_seed)
    used_hh = set(households["household_id"])
    used_pid = set(persons["person_id"])
    new_hh_parts: List[pd.DataFrame] = []
    new_p_parts: List[pd.DataFrame] = []

    for gid, w in enumerate(weights):
        mask = groups == gid
        grp_hh = households[mask]
        if grp_hh.empty:
            continue

        orig_n = len(grp_hh)
        target_n = max(1, int(np.round(w * orig_n)))

        if target_n <= orig_n:
            # --- subsample ---
            if zone_col and zone_col in grp_hh.columns and grp_hh[zone_col].nunique() > 1:
                parts = []
                for _, zdf in grp_hh.groupby(zone_col):
                    zn = max(1, int(len(zdf) * target_n / orig_n))
                    parts.append(zdf.sample(n=min(zn, len(zdf)), random_state=42))
                sel = pd.concat(parts, ignore_index=True)
            else:
                sel = grp_hh.sample(n=target_n, random_state=42)
            new_hh_parts.append(sel)
            new_p_parts.append(persons[persons["household_id"].isin(sel["household_id"])])
        else:
            # --- keep originals + replicate extras ---
            new_hh_parts.append(grp_hh.copy())
            new_p_parts.append(persons[persons["household_id"].isin(grp_hh["household_id"])])

            for _ in range(target_n - orig_n):
                row = grp_hh.sample(n=1, random_state=rng.integers(0, 2**31)).iloc[0].copy()
                old_hid = row["household_id"]
                new_hid = int(str(uuid.uuid4().int)[:8])
                while new_hid in used_hh:
                    new_hid = int(str(uuid.uuid4().int)[:8])
                used_hh.add(new_hid)
                row["household_id"] = new_hid
                new_hh_parts.append(pd.DataFrame([row]))

                p_copy = persons[persons["household_id"] == old_hid].copy()
                p_copy["household_id"] = new_hid
                for idx in p_copy.index:
                    new_pid = int(str(uuid.uuid4().int)[:8])
                    while new_pid in used_pid:
                        new_pid = int(str(uuid.uuid4().int)[:8])
                    used_pid.add(new_pid)
                    p_copy.at[idx, "person_id"] = new_pid
                new_p_parts.append(p_copy)

    if not new_hh_parts:
        return persons.copy(), households.copy()

    hh_out = pd.concat(new_hh_parts, ignore_index=True)
    p_out = pd.concat(new_p_parts, ignore_index=True)

    if hh_out["household_id"].duplicated().any():
        raise ValueError("Duplicate household IDs after scaling")
    if p_out["person_id"].duplicated().any():
        raise ValueError("Duplicate person IDs after scaling")

    return p_out, hh_out


# ---------------------------------------------------------------------------
# Optimiser interface
# ---------------------------------------------------------------------------

class Optimiser(ABC):
    """Common interface for all SPE optimisation engines."""

    @abstractmethod
    def ask(self) -> List[np.ndarray]:
        """Return a list of candidate parameter vectors."""

    @abstractmethod
    def tell(self, solutions: List[np.ndarray], scores: List[float]) -> None:
        """Ingest evaluation results and update internal state."""

    @property
    @abstractmethod
    def best_solution(self) -> Tuple[np.ndarray, float]:
        """Return ``(params, score)`` for the best candidate seen so far."""

    @property
    @abstractmethod
    def converged(self) -> bool:
        """Whether the optimiser has detected convergence."""


# ---------------------------------------------------------------------------
# CMA-ES engine
# ---------------------------------------------------------------------------

class CMAESEngine(Optimiser):
    """CMA-ES optimiser wrapping the ``cma`` package."""

    def __init__(self, dim: int, popsize: int, *, seed: int = 42):
        import cma
        self._dim = dim
        self._popsize = max(popsize, 4 + int(3 * np.log(dim)))
        opts = {
            "popsize": self._popsize, "maxiter": 200,
            "tolfun": 1e-6, "tolx": 1e-8, "tolstagnation": 15,
            "bounds": [-6, 6], "seed": seed,
        }
        self._es = cma.CMAEvolutionStrategy([0.0] * dim, 0.6, opts)
        self._best = (np.zeros(dim), FAILURE_SCORE)
        self._history: deque = deque(maxlen=20)
        self._converged = False

    def ask(self):
        return [np.array(x) for x in self._es.ask()]

    def tell(self, solutions, scores):
        losses = [200.0 if s <= FAILURE_SCORE else 100.0 - s for s in scores]
        self._es.tell([s.tolist() for s in solutions], losses)
        best_idx = int(np.argmax(scores))
        if scores[best_idx] > self._best[1]:
            self._best = (solutions[best_idx].copy(), scores[best_idx])
        self._history.append(self._best[1])
        self._check()

    @property
    def best_solution(self):
        return self._best

    @property
    def converged(self):
        return self._converged

    def _check(self):
        if len(self._history) >= 8:
            recent = list(self._history)[-8:]
            if all(recent[i+1] - recent[i] < 1.0 for i in range(len(recent)-1)):
                self._converged = True


# ---------------------------------------------------------------------------
# LSTM policy-network engine
# ---------------------------------------------------------------------------

class _ParameterMemory:
    """Stores successful parameter vectors for memory-guided generation."""

    def __init__(self, capacity: int = 25, store_threshold: float = 45.0):
        self._cap = capacity
        self._threshold = store_threshold
        self.entries: List[Dict] = []

    def store(self, params: np.ndarray, score: float, iteration: int):
        if score > self._threshold:
            self.entries.append({"params": params.copy(), "score": score, "iter": iteration})
            if len(self.entries) > self._cap:
                self.entries.sort(key=lambda e: e["score"], reverse=True)
                self.entries = self.entries[:self._cap]

    def guidance(self) -> Optional[np.ndarray]:
        if not self.entries:
            return None
        scores = np.array([e["score"] for e in self.entries])
        w = np.exp((scores - scores.min()) / max(scores.std(), 1.0))
        w /= w.sum()
        return np.dot(w, np.array([e["params"] for e in self.entries]))


@dataclass
class LSTMConfig:
    """Hyperparameters for the LSTM policy-network engine.

    Users should tune these for their specific optimisation landscape.
    The defaults provided are generic starting points and are **not**
    the values used in the published experiments.
    """
    hidden_size: int = 16
    feature_dim: int = 12
    memory_capacity: int = 20
    memory_store_threshold: float = 40.0
    curriculum_stages: Tuple[int, ...] = (0, 8, 16)
    stage_spread: Tuple[float, ...] = (0.7, 0.85, 1.0)
    stage_k_clip: Tuple[float, ...] = (0.5, 0.8, 4.0)
    stage_lr_scale: Tuple[float, ...] = (0.9, 1.0, 1.1)
    stage_div_target: Tuple[float, ...] = (0.9, 1.0, 1.1)
    stage_conv_tol: Tuple[float, ...] = (0.8, 1.0, 1.2)
    init_std: float = 0.5
    base_lr: float = 0.15
    gradient_weight: float = 0.65
    lstm_influence: float = 0.2
    memory_mutation_std: float = 0.1
    memory_fraction: float = 0.25
    memory_guidance_weight: float = 0.1
    weight_init_scale: float = 0.1
    gate_init_scale: float = 0.08
    forget_bias_init: float = 1.5
    param_clip: float = 4.0
    mean_clip: float = 3.0
    std_bounds: Tuple[float, float] = (0.1, 1.5)
    grad_clip: float = 1.5
    convergence_window: int = 8
    convergence_patience: int = 7


class LSTMEngine(Optimiser):
    """Memory-augmented LSTM policy network with curriculum learning.
    """

    def __init__(self, dim: int, popsize: int, config: Optional[LSTMConfig] = None):
        cfg = config or LSTMConfig()
        self._dim = dim
        self._popsize = popsize
        self._cfg = cfg
        self._memory = _ParameterMemory(cfg.memory_capacity, cfg.memory_store_threshold)

        # curriculum
        self._stage = 0
        self._stage_thresholds = list(cfg.curriculum_stages)
        self._iteration = 0

        # initialisation
        self._mean = np.zeros(dim)
        self._std = np.full(dim, cfg.init_std)
        self._best = (self._mean.copy(), FAILURE_SCORE)

        # LSTM cell parameters
        hid = max(cfg.hidden_size, dim // 2)
        fd = cfg.feature_dim
        self._feat_dim = fd
        self._hid = hid
        gs = cfg.gate_init_scale
        ws = cfg.weight_init_scale
        self._Wf = np.random.randn(hid, hid) * gs
        self._Wi = np.random.randn(hid, hid) * gs
        self._Wo = np.random.randn(hid, hid) * gs
        self._Wc = np.random.randn(hid, hid) * gs
        self._Win = np.random.randn(fd, hid) * ws
        self._bf = np.ones(hid) * cfg.forget_bias_init
        self._bi = np.zeros(hid)
        self._bo = np.zeros(hid)
        self._bc = np.zeros(hid)
        self._bin = np.zeros(hid)
        self._Wout = np.random.randn(hid, dim) * ws
        self._bout = np.zeros(dim)
        self._cell = np.zeros(hid)
        self._hidden = np.zeros(hid)

        # tracking
        self._scores: deque = deque(maxlen=cfg.convergence_window * 2)
        self._params_hist: deque = deque(maxlen=cfg.convergence_window * 2)
        self._grad_hist: deque = deque(maxlen=cfg.convergence_window)
        self._converged = False
        self._conv_count = 0

    # ----- ask -----

    def ask(self):
        cfg = self._cfg
        if self._iteration in self._stage_thresholds:
            new = min(len(cfg.stage_spread) - 1,
                      self._stage_thresholds.index(self._iteration))
            if new != self._stage:
                self._stage = new
                logger.info("LSTM curriculum → stage %d", self._stage)

        spread = cfg.stage_spread[self._stage]
        k_clip = cfg.stage_k_clip[self._stage]
        candidates = []
        for i in range(self._popsize):
            c = np.random.normal(self._mean, self._std * spread)
            c[0] = np.clip(c[0], -k_clip, k_clip)

            # memory-guided mutation
            n_mem = max(1, int(self._popsize * cfg.memory_fraction))
            if i < n_mem and self._memory.entries:
                best_mem = max(self._memory.entries, key=lambda e: e["score"])
                c = best_mem["params"] + np.random.normal(0, cfg.memory_mutation_std, self._dim)

            # LSTM bias
            if len(self._hidden) >= self._dim:
                c += cfg.lstm_influence * self._hidden[:self._dim]
            candidates.append(np.clip(c, -cfg.param_clip, cfg.param_clip))
        return candidates

    # ----- tell -----

    def tell(self, solutions, scores):
        cfg = self._cfg
        solutions = np.array(solutions)
        scores_arr = np.array(scores)
        self._iteration += 1

        best_idx = int(np.argmax(scores_arr))
        if scores_arr[best_idx] > self._best[1]:
            self._best = (solutions[best_idx].copy(), float(scores_arr[best_idx]))
            self._memory.store(self._best[0], self._best[1], self._iteration)

        features = self._extract_features(solutions, scores_arr)
        adjustments = self._forward(features)

        valid = scores_arr > FAILURE_SCORE
        if valid.any() and scores_arr[valid].std() > 1e-6:
            vs, vsc = solutions[valid], scores_arr[valid]
            ranks = np.argsort(np.argsort(-vsc))
            w = np.exp(-ranks / max(len(ranks) / 3.5, 1.0))
            w /= w.sum() + 1e-10
            grad = np.clip(np.dot(w, vs - self._mean), -cfg.grad_clip, cfg.grad_clip)
            lr = cfg.base_lr * cfg.stage_lr_scale[self._stage]
            gw = cfg.gradient_weight
            update = lr * (gw * grad + (1.0 - gw) * adjustments)
            self._mean = np.clip(self._mean + update, -cfg.mean_clip, cfg.mean_clip)

            div = self._diversity(vs)
            tgt = cfg.stage_div_target[self._stage]
            if div > tgt * 1.3:
                self._std *= 0.95
            elif div < tgt * 0.7:
                self._std *= 1.05
            lo, hi = cfg.std_bounds
            self._std = np.clip(self._std, lo, hi)
            self._grad_hist.append(grad)

        self._scores.append(float(scores_arr[best_idx]))
        self._params_hist.append(self._best[0].copy())
        self._check_convergence()

    # ----- internals -----

    def _forward(self, feat: np.ndarray) -> np.ndarray:
        cfg = self._cfg
        x = np.tanh(feat @ self._Win + self._bin)
        _sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -6, 6)))
        fg = _sig(self._hidden @ self._Wf + x + self._bf)
        ig = _sig(self._hidden @ self._Wi + x + self._bi)
        cand = np.tanh(np.clip(self._hidden @ self._Wc + x + self._bc, -6, 6))
        self._cell = np.clip(fg * self._cell + ig * cand, -10, 10)
        og = _sig(self._hidden @ self._Wo + x + self._bo)
        self._hidden = np.clip(og * np.tanh(self._cell), -5, 5)
        adj = self._hidden @ self._Wout + self._bout
        adj *= min(1.0, (self._iteration + 1) / max(len(self._stage_thresholds), 1))
        guide = self._memory.guidance()
        if guide is not None and len(self._scores) > 2:
            direction = np.clip(guide - self._mean, -cfg.grad_clip, cfg.grad_clip)
            adj += cfg.memory_guidance_weight * direction
        return np.tanh(adj)

    def _extract_features(self, sols: np.ndarray, sc: np.ndarray) -> np.ndarray:
        f = np.zeros(self._feat_dim)
        if len(sc) == 0:
            return f
        if len(self._scores) >= 4:
            t = np.polyfit(range(len(list(self._scores)[-4:])), list(self._scores)[-4:], 1)[0]
            f[0] = np.tanh(t / 2.0)
        if len(sols) > 1:
            d = [np.linalg.norm(sols[i] - sols[j])
                 for i in range(min(8, len(sols))) for j in range(i+1, min(8, len(sols)))]
            if d:
                f[1] = min(np.mean(d) / np.sqrt(self._dim) / 2.0, 1.0)
        f[2] = min(float(np.std(sc)) / 6.0, 1.0)
        f[5] = max(min((float(np.max(sc)) - FAILURE_SCORE) / (100 - FAILURE_SCORE), 1.0), 0.0)
        f[8] = self._stage / max(len(self._cfg.stage_spread) - 1, 1)
        f[9] = 1.0 if self._memory.entries else 0.0
        return f

    @staticmethod
    def _diversity(sols: np.ndarray) -> float:
        if len(sols) < 2:
            return 1.0
        n = min(len(sols), 10)
        d = [np.linalg.norm(sols[i] - sols[j]) for i in range(n) for j in range(i+1, n)]
        return float(np.mean(d) / np.sqrt(sols.shape[1])) if d else 1.0

    def _check_convergence(self):
        cfg = self._cfg
        w = cfg.convergence_window
        if len(self._scores) < w + 1:
            return
        recent = list(self._scores)[-w:]
        tol = cfg.stage_conv_tol[self._stage]
        if all(recent[i+1] - recent[i] < tol for i in range(len(recent)-1)):
            self._conv_count += 1
            if self._conv_count >= cfg.convergence_patience:
                self._converged = True
        else:
            self._conv_count = 0

    @property
    def best_solution(self):
        return self._best

    @property
    def converged(self):
        return self._converged


# ---------------------------------------------------------------------------
# PPO-style RL engine
# ---------------------------------------------------------------------------

@dataclass
class RLConfig:
    """Hyperparameters for the PPO-style RL engine.

    Users should tune these for their specific problem. The defaults
    are generic starting points and are **not** the values used in
    the published experiments.
    """
    init_cov_scale: float = 0.5
    base_lr: float = 0.1
    momentum: float = 0.85
    elite_fraction: float = 0.25
    elite_perturb_std: float = 0.05
    adapt_range: Tuple[float, float] = (0.3, 1.5)
    cov_lr: float = 0.1
    min_eigenvalue: float = 1e-5
    max_eigenvalue: float = 10.0
    velocity_clip: float = 0.8
    mean_clip: float = 4.0
    param_clip: float = 6.0
    diff_clip: float = 4.0
    convergence_window: int = 8
    convergence_tolerance: float = 1.0
    convergence_patience: int = 7


class RLEngine(Optimiser):
    """Policy-gradient optimiser with adaptive covariance and elite replay.
    """

    def __init__(self, dim: int, popsize: int, config: Optional[RLConfig] = None):
        cfg = config or RLConfig()
        self._dim = dim
        self._popsize = popsize
        self._cfg = cfg
        self._mean = np.zeros(dim)
        self._cov = np.eye(dim) * cfg.init_cov_scale
        self._vel = np.zeros(dim)
        self._best = (np.zeros(dim), FAILURE_SCORE)
        self._elite: List[np.ndarray] = []
        self._history: deque = deque(maxlen=cfg.convergence_window * 3)
        self._success: deque = deque(maxlen=cfg.convergence_window * 2)
        self._converged = False
        self._conv_count = 0

    def ask(self):
        cfg = self._cfg
        self._stabilise_cov()
        candidates = []
        n_elite = min(len(self._elite), max(1, int(self._popsize * cfg.elite_fraction)))
        n_new = self._popsize - n_elite
        try:
            for _ in range(n_new):
                candidates.append(np.clip(
                    np.random.multivariate_normal(self._mean, self._cov),
                    -cfg.param_clip, cfg.param_clip))
        except np.linalg.LinAlgError:
            diag = np.diag(np.maximum(np.diag(self._cov), cfg.min_eigenvalue))
            for _ in range(n_new):
                candidates.append(np.clip(
                    np.random.multivariate_normal(self._mean, diag),
                    -cfg.param_clip, cfg.param_clip))
        for e in self._elite[:n_elite]:
            candidates.append(np.clip(
                e + np.random.normal(0, cfg.elite_perturb_std, self._dim),
                -cfg.param_clip, cfg.param_clip))
        return candidates

    def tell(self, solutions, scores):
        cfg = self._cfg
        solutions = np.array(solutions)
        scores_arr = np.array(scores)
        best_idx = int(np.argmax(scores_arr))
        if scores_arr[best_idx] > self._best[1]:
            imp = (scores_arr[best_idx] - self._best[1]) / max(abs(self._best[1]), 1e-8)
            self._success.append(imp)
            self._best = (solutions[best_idx].copy(), float(scores_arr[best_idx]))
        else:
            self._success.append(0.0)
        self._history.append(self._best[1])

        valid = scores_arr > FAILURE_SCORE
        if valid.any() and valid.sum() > 1:
            vs, vsc = solutions[valid], scores_arr[valid]
            order = np.argsort(-vsc)
            n_keep = max(1, int(self._popsize * cfg.elite_fraction))
            self._elite = [vs[i].copy() for i in order[:n_keep]]

            if vsc.std() > 1e-6:
                ranks = np.argsort(np.argsort(-vsc))
                w = np.exp(-ranks / max(len(ranks) / 4, 1.0))
                w /= w.sum() + 1e-10
                grad = np.clip(np.dot(w, vs - self._mean),
                               -cfg.diff_clip, cfg.diff_clip)
                lo, hi = cfg.adapt_range
                adapt = lo + (hi - lo) * min(max(np.mean(list(self._success)), 0), 1.0)
                lr = cfg.base_lr * adapt
                self._vel = cfg.momentum * self._vel + lr * grad
                self._vel = np.clip(self._vel, -cfg.velocity_clip, cfg.velocity_clip)
                self._mean = np.clip(self._mean + self._vel,
                                     -cfg.mean_clip, cfg.mean_clip)

                diff = np.clip(vs - self._mean, -cfg.diff_clip, cfg.diff_clip)
                wc = np.dot((diff * w[:, None]).T, diff)
                self._cov = (1 - cfg.cov_lr) * self._cov + cfg.cov_lr * wc
                self._stabilise_cov()

        self._check()

    def _stabilise_cov(self):
        cfg = self._cfg
        try:
            ev, evec = np.linalg.eigh(self._cov)
            ev = np.clip(ev, cfg.min_eigenvalue, cfg.max_eigenvalue)
            self._cov = evec @ np.diag(ev) @ evec.T + 2e-7 * np.eye(self._dim)
        except np.linalg.LinAlgError:
            self._cov = np.diag(np.clip(np.diag(self._cov),
                                        cfg.min_eigenvalue, cfg.max_eigenvalue))

    def _check(self):
        cfg = self._cfg
        w = cfg.convergence_window
        if len(self._history) >= w:
            r = list(self._history)[-w:]
            if all(r[i+1] - r[i] < cfg.convergence_tolerance
                   for i in range(len(r)-1)):
                self._conv_count += 1
                if self._conv_count >= cfg.convergence_patience:
                    self._converged = True
            else:
                self._conv_count = 0

    @property
    def best_solution(self):
        return self._best

    @property
    def converged(self):
        return self._converged


# ---------------------------------------------------------------------------
# Public convenience: build optimiser by name
# ---------------------------------------------------------------------------

def make_optimiser(
    name: str,
    dim: int,
    popsize: int,
    *,
    lstm_config: Optional[LSTMConfig] = None,
    rl_config: Optional[RLConfig] = None,
) -> Optimiser:
    """Instantiate an optimiser by name: ``"cmaes"``, ``"lstm"``, or ``"rl"``.

    Pass a :class:`LSTMConfig` or :class:`RLConfig` to override default
    hyperparameters.
    """
    name = name.lower().strip()
    if name == "cmaes":
        return CMAESEngine(dim, popsize)
    elif name == "lstm":
        return LSTMEngine(dim, popsize, config=lstm_config)
    elif name in ("rl", "ppo"):
        return RLEngine(dim, popsize, config=rl_config)
    else:
        raise ValueError(f"Unknown optimiser: {name!r}. Choose from cmaes, lstm, rl.")


# ---------------------------------------------------------------------------
# CLI example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info(
        "SPE mechanism module loaded. Import and use `decode_parameters`, "
        "`hash_group`, `scale_population`, and `make_optimiser` to build your optimisation loop. " \
        "See the paper for the full pipeline."
    )