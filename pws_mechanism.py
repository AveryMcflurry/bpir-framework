#!/usr/bin/env python3
"""PWS mechanism: objective and household weighting."""
from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LogCoshObjective:
    """Log-cosh composite objective for evaluating household subsets."""

    def __init__(
        self,
        target_trip_rate: float,
        target_purpose_shares: Dict[str, float],
        target_mode_shares: Dict[str, float],
        alpha_trip: float = 2.0,
        alpha_dist: float = 4.0,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.target_trip_rate = target_trip_rate
        self.purpose_keys = list(target_purpose_shares.keys())
        self.mode_keys = list(target_mode_shares.keys())
        self._target_purpose = np.array([target_purpose_shares[k] for k in self.purpose_keys])
        self._target_mode = np.array([target_mode_shares[k] for k in self.mode_keys])
        self.alpha_trip = alpha_trip
        self.alpha_dist = alpha_dist
        self.weights = weights or {"trip": 0.25, "purpose": 0.40, "mode": 0.35}


    @staticmethod
    def _logcosh(x: np.ndarray | float) -> np.ndarray | float:
        return np.log(np.cosh(x))

    def trip_rate_loss(self, actual: float) -> float:
        if actual <= 0:
            return 10.0
        relative_error = (actual - self.target_trip_rate) / self.target_trip_rate
        return float(self._logcosh(self.alpha_trip * relative_error))

    def distribution_loss(self, actual: np.ndarray, target: np.ndarray) -> float:
        return float(np.sum(self._logcosh(self.alpha_dist * (actual - target))))


    def score(
        self,
        n_trips: int,
        n_households: int,
        purpose_counts: Dict[str, int],
        mode_counts: Dict[str, int],
    ) -> Tuple[float, Dict]:
        """Evaluate a candidate household subset."""
        if n_trips == 0 or n_households == 0:
            return -10.0, {"error": "empty_subset"}

        actual_trip_rate = n_trips / n_households
        actual_purpose = np.array([purpose_counts.get(k, 0) / n_trips for k in self.purpose_keys])
        actual_mode = np.array([mode_counts.get(k, 0) / n_trips for k in self.mode_keys])

        l_trip = self.trip_rate_loss(actual_trip_rate)
        l_purp = self.distribution_loss(actual_purpose, self._target_purpose)
        l_mode = self.distribution_loss(actual_mode, self._target_mode)

        total_loss = (
            self.weights["trip"] * l_trip
            + self.weights["purpose"] * l_purp
            + self.weights["mode"] * l_mode
        )
        composite = 100.0 * np.exp(-total_loss)

        components = {
            "composite_score": composite,
            "total_loss": total_loss,
            "trip_loss": l_trip,
            "purpose_loss": l_purp,
            "mode_loss": l_mode,
            "trip_rate_actual": actual_trip_rate,
            "trip_rate_target": self.target_trip_rate,
            "trip_score": 100.0 * np.exp(-l_trip),
            "purpose_score": 100.0 * np.exp(-l_purp),
            "mode_score": 100.0 * np.exp(-l_mode),
        }
        return composite, components


class WeightManager:
    """Tracks per-household performance and maintains adaptive sampling weights."""

    def __init__(self, household_ids: Sequence[int]):
        self._ids = list(household_ids)
        self._id_set = set(self._ids)
        self._weights: Dict[int, float] = {hid: 1.0 for hid in self._ids}
        self._good_freq: Dict[int, int] = defaultdict(int)
        self._tiers: Dict[str, List[int]] = {
            "excellent": [], "good": [], "average": [], "poor": [],
        }

    @property
    def ids(self) -> List[int]:
        return self._ids


    def update(self, teams: List[Dict], threshold: float = 90.0) -> None:
        """Take the evaluations of one iteration and update the weights."""
        per_hh_scores: Dict[int, List[float]] = defaultdict(list)

        for team in teams:
            score = team["score"]
            for hid in team["household_ids"]:
                per_hh_scores[hid].append(score)
                if score >= threshold:
                    self._good_freq[hid] += 1

        self._refresh_tiers(per_hh_scores)
        self._recompute_weights()

    def _refresh_tiers(self, per_hh: Dict[int, List[float]]) -> None:
        tiers = {"excellent": [], "good": [], "average": [], "poor": []}
        for hid, scores in per_hh.items():
            avg = float(np.mean(scores))
            if avg >= 95.0:
                tiers["excellent"].append(hid)
            elif avg >= 85.0:
                tiers["good"].append(hid)
            elif avg >= 70.0:
                tiers["average"].append(hid)
            else:
                tiers["poor"].append(hid)
        self._tiers = tiers

    def _recompute_weights(self) -> None:
        max_freq = max(self._good_freq.values()) if self._good_freq else 1
        for hid in self._ids:
            freq = self._good_freq.get(hid, 0)
            if freq == 0:
                self._weights[hid] = 0.1
            else:
                normed = freq / max_freq
                self._weights[hid] = max(1.0, freq * normed * 10.0)


    def probabilities(self) -> np.ndarray:
        """Return a normalised probability vector aligned with ``self.ids``."""
        w = np.array([self._weights[hid] for hid in self._ids])
        total = w.sum()
        return w / total if total > 0 else np.ones_like(w) / len(w)

    def top_performers(self, k: int = 50) -> List[int]:
        """Return the k household IDs that appear most often in good subsets."""
        ranked = sorted(self._ids, key=lambda h: self._good_freq.get(h, 0), reverse=True)
        excellent = self._tiers.get("excellent", [])
        combined = list(dict.fromkeys(ranked[:k] + excellent))
        return combined[:k]

    def tier(self, name: str) -> List[int]:
        return self._tiers.get(name, [])


class AdaptiveTarget:
    """Target relaxation schedule."""

    def __init__(self, initial: float = 95.0, good_fit: float = 80.0, patience: int = 15):
        self.initial = initial
        self.good_fit = good_fit
        self.patience = patience
        self._current = initial
        self._best = 0.0

    @property
    def current(self) -> float:
        return self._current

    def step(self, iteration: int, best_score: float, stall_count: int) -> float:
        """Advance the target schedule by one iteration."""
        self._best = max(self._best, best_score)

        if iteration <= 20:
            self._current = self.initial
        elif iteration <= 50:
            if stall_count >= 10 and self._best < 90.0:
                self._current = max(85.0, self._best + 2.0)
        else:
            if stall_count >= self.patience and self._best >= self.good_fit:
                self._current = max(self.good_fit, self._best - 2.0)

        return self._current

    def achieved(self, score: float) -> bool:
        return score >= self._current


def _collect_trips(
    household_ids: List[int],
    trip_index: Dict[int, pd.DataFrame],
) -> Tuple[pd.DataFrame, int]:
    """Gather trip records for a list of household IDs (with repetition)."""
    frames = []
    unique = set()
    for hid in household_ids:
        unique.add(hid)
        if hid in trip_index:
            frames.append(trip_index[hid])
    trips = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return trips, len(unique)


def sample_deterministic(
    n_teams: int,
    team_size: int,
    weight_mgr: WeightManager,
    trip_index: Dict[int, pd.DataFrame],
    good_cache: List[List[int]],
) -> List[Dict]:
    """Deterministic sampling from the top performers."""
    top = weight_mgr.top_performers(k=min(team_size * 3, len(weight_mgr.ids)))
    all_ids = weight_mgr.ids
    teams = []

    for i in range(n_teams):
        if len(top) >= team_size:
            if i < n_teams // 3:
                selection = top[:team_size]
            elif i < 2 * n_teams // 3:
                core_n = max(team_size // 2, 1)
                core = top[:core_n]
                rest_pool = top[core_n: team_size * 2]
                need = team_size - core_n
                if len(rest_pool) >= need:
                    extra = np.random.choice(rest_pool, size=need, replace=False).tolist()
                else:
                    extra = rest_pool + np.random.choice(
                        all_ids, size=need - len(rest_pool), replace=True,
                    ).tolist()
                selection = core + extra
            else:
                if good_cache:
                    base = list(good_cache[np.random.randint(len(good_cache))])
                    if len(base) < team_size:
                        base += top[: team_size - len(base)]
                    selection = base[:team_size]
                else:
                    selection = top[:team_size]
        else:
            selection = (top * (team_size // len(top) + 1))[:team_size]

        trips, n_unique = _collect_trips(selection, trip_index)
        teams.append({
            "household_ids": selection,
            "trips": trips,
            "n_unique": n_unique,
            "strategy": "deterministic",
        })
    return teams


def sample_progressive(
    n_teams: int,
    team_size: int,
    weight_mgr: WeightManager,
    trip_index: Dict[int, pd.DataFrame],
    relaxation: float,
) -> List[Dict]:
    """Deterministic core with weighted random fill."""
    probs = weight_mgr.probabilities()
    top = weight_mgr.top_performers(k=team_size * 2)
    excellent = weight_mgr.tier("excellent")
    good = weight_mgr.tier("good")
    all_ids = weight_mgr.ids
    teams = []

    for _ in range(n_teams):
        n_det = int(team_size * (1.0 - relaxation))
        n_rand = team_size - n_det

        det_ids: List[int] = []
        if n_det > 0 and top:
            if excellent:
                take = min(n_det // 2, len(excellent))
                det_ids.extend(excellent[:take])
            remaining = n_det - len(det_ids)
            if remaining > 0 and good:
                det_ids.extend(good[: min(remaining, len(good))])
            remaining = n_det - len(det_ids)
            if remaining > 0:
                det_ids.extend(top[:remaining])
            while len(det_ids) < n_det:
                det_ids.append(top[len(det_ids) % len(top)])
            det_ids = det_ids[:n_det]

        rand_ids = np.random.choice(all_ids, size=n_rand, replace=True, p=probs).tolist() if n_rand > 0 else []
        selection = det_ids + rand_ids

        while len(selection) < team_size:
            selection.append(np.random.choice(all_ids))
        selection = selection[:team_size]

        trips, n_unique = _collect_trips(selection, trip_index)
        teams.append({
            "household_ids": selection,
            "trips": trips,
            "n_unique": n_unique,
            "strategy": f"progressive_{relaxation:.2f}",
        })
    return teams


def sample_weighted(
    n_teams: int,
    team_size: int,
    weight_mgr: WeightManager,
    trip_index: Dict[int, pd.DataFrame],
) -> List[Dict]:
    """Weighted random sampling."""
    probs = weight_mgr.probabilities()
    all_ids = weight_mgr.ids
    teams = []

    for _ in range(n_teams):
        selection = np.random.choice(all_ids, size=team_size, replace=True, p=probs).tolist()
        trips, n_unique = _collect_trips(selection, trip_index)
        teams.append({
            "household_ids": selection,
            "trips": trips,
            "n_unique": n_unique,
            "strategy": "weighted",
        })
    return teams


def select_phase(iteration: int) -> Tuple[str, float]:
    """Return ``(strategy_name, relaxation_factor)`` for a given iteration."""
    if iteration <= 15:
        return "deterministic", 0.0
    elif iteration <= 50:
        progress = (iteration - 15) / 35.0
        return "progressive", min(0.8, progress)
    else:
        return "weighted", 1.0


@dataclass
class PWSResult:
    """Container for a completed PWS run."""
    success: bool
    best_score: float
    total_iterations: int
    elapsed_seconds: float
    valid_teams: List[Dict] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)
    phase_counts: Dict[str, int] = field(default_factory=dict)


def run_pws(
    trips: pd.DataFrame,
    household_ids: List[int],
    objective: LogCoshObjective,
    *,
    purpose_col: str = "purpose",
    mode_col: str = "trip_mode",
    household_col: str = "household_id",
    max_iterations: int = 100,
    n_teams: int = 1000,
    team_size: int = 100,
    good_threshold: float = 90.0,
    valid_threshold: float = 95.0,
    stall_limit: int = 20,
    seed: int = 42,
) -> PWSResult:
    """Run the sampling-based search over household subsets."""
    np.random.seed(seed)

    trips[household_col] = trips[household_col].astype(int)
    trip_index: Dict[int, pd.DataFrame] = {
        hid: group for hid, group in trips.groupby(household_col)
    }

    weight_mgr = WeightManager(household_ids)
    target = AdaptiveTarget()
    good_cache: List[List[int]] = []

    best_score = 0.0
    stall_count = 0
    valid_teams: List[Dict] = []
    history: List[Dict] = []
    phase_counts: Dict[str, int] = defaultdict(int)

    t_start = time.time()

    for iteration in range(1, max_iterations + 1):
        t_iter = time.time()
        phase, relaxation = select_phase(iteration)
        phase_counts[phase] += 1
        current_target = target.step(iteration, best_score, stall_count)

        if phase == "deterministic":
            teams = sample_deterministic(n_teams, team_size, weight_mgr, trip_index, good_cache)
        elif phase == "progressive":
            teams = sample_progressive(n_teams, team_size, weight_mgr, trip_index, relaxation)
        else:
            teams = sample_weighted(n_teams, team_size, weight_mgr, trip_index)

        evaluated: List[Dict] = []
        iter_valid = 0
        iter_good = 0

        for team in teams:
            trip_df = team["trips"]
            n_hh = team["n_unique"]

            if trip_df.empty:
                score, comp = -10.0, {"error": "empty"}
            else:
                purp_counts = trip_df[purpose_col].value_counts().to_dict() if purpose_col in trip_df.columns else {}
                mode_counts = trip_df[mode_col].value_counts().to_dict() if mode_col in trip_df.columns else {}
                score, comp = objective.score(len(trip_df), n_hh, purp_counts, mode_counts)

            record = {
                "household_ids": team["household_ids"],
                "score": score,
                "components": comp,
                "n_unique": n_hh,
                "strategy": team["strategy"],
            }
            evaluated.append(record)

            if score >= valid_threshold:
                iter_valid += 1
                valid_teams.append({**record, "iteration": iteration})

            if score >= good_threshold:
                iter_good += 1

            if score >= 85.0:
                good_cache.append(team["household_ids"][:])
                if len(good_cache) > 50:
                    good_cache.pop(0)

        weight_mgr.update(evaluated, threshold=good_threshold)

        iter_best = max(r["score"] for r in evaluated)
        iter_mean = float(np.mean([r["score"] for r in evaluated]))

        history.append({
            "iteration": iteration,
            "phase": phase,
            "relaxation": relaxation,
            "target": current_target,
            "best": iter_best,
            "mean": iter_mean,
            "valid": iter_valid,
            "good": iter_good,
            "elapsed": time.time() - t_iter,
        })

        if iter_best > best_score:
            best_score = iter_best
            stall_count = 0
        else:
            stall_count += 1

        logger.info(
            "Iter %3d  %-13s  relax=%.2f  target=%5.1f  best=%6.2f  "
            "mean=%6.2f  valid=%3d  good=%3d  [%.1fs]",
            iteration, phase, relaxation, current_target,
            iter_best, iter_mean, iter_valid, iter_good,
            history[-1]["elapsed"],
        )

        if iter_valid >= 3:
            logger.info("Sufficient valid teams found — stopping.")
            break
        if stall_count >= stall_limit:
            if best_score >= target.good_fit:
                logger.info("Good-fit threshold reached with stalled progress — stopping.")
            else:
                logger.info("Stall limit reached — stopping.")
            break

    elapsed = time.time() - t_start
    success = bool(valid_teams) or best_score >= target.good_fit

    logger.info(
        "PWS complete: success=%s  best=%.2f  valid_teams=%d  "
        "iterations=%d  elapsed=%.1fs",
        success, best_score, len(valid_teams),
        len(history), elapsed,
    )

    return PWSResult(
        success=success,
        best_score=best_score,
        total_iterations=len(history),
        elapsed_seconds=elapsed,
        valid_teams=valid_teams,
        history=history,
        phase_counts=dict(phase_counts),
    )


def main(args: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="PWS — Progressive Weighted Subsampling")
    parser.add_argument("--trips", required=True, help="Path to simulated trips CSV")
    parser.add_argument("--households", required=True, help="Path to simulated households CSV")
    parser.add_argument("--targets", required=True,
                        help="Path to JSON file with observed targets "
                             "(trip_rate, purpose_shares, mode_shares)")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--n-teams", type=int, default=1000)
    parser.add_argument("--team-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--purpose-col", default="purpose")
    parser.add_argument("--mode-col", default="trip_mode")
    parser.add_argument("--household-col", default="household_id")
    opts = parser.parse_args(args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    trips = pd.read_csv(opts.trips, low_memory=False)
    households = pd.read_csv(opts.households, low_memory=False)
    hh_ids = households[opts.household_col].unique().tolist()

    with open(opts.targets) as fh:
        targets = json.load(fh)

    objective = LogCoshObjective(
        target_trip_rate=targets["trip_rate"],
        target_purpose_shares=targets["purpose_shares"],
        target_mode_shares=targets["mode_shares"],
    )

    result = run_pws(
        trips, hh_ids, objective,
        purpose_col=opts.purpose_col,
        mode_col=opts.mode_col,
        household_col=opts.household_col,
        max_iterations=opts.max_iter,
        n_teams=opts.n_teams,
        team_size=opts.team_size,
        seed=opts.seed,
    )

    print(json.dumps({
        "success": result.success,
        "best_score": round(result.best_score, 4),
        "valid_teams": len(result.valid_teams),
        "iterations": result.total_iterations,
        "elapsed_seconds": round(result.elapsed_seconds, 1),
        "phase_counts": result.phase_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
