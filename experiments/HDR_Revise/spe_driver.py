#!/usr/bin/env python3
r"""
SPE (Segmentation-based Population Exploration) driver
======================================================

Mechanism 2 of the three-mechanism correction story:

    PWS  — post-hoc reweighting of the household SAMPLE (no learning, no re-sim)
    SPE  — optimiser-driven RESAMPLING of the household composition + re-sim   <-- this file
    HDR  — fully weighted inputs (hh/person/land-use/skim), re-sim

SPE partitions households into k hash-based groups and lets an optimiser choose
the per-group scaling weights (softmax-normalised) plus the grouping itself
(k and the hash seed are part of the search). Each candidate then:

    decode(raw) -> (k, seed, group_weights)
      -> hash_group(household_id, k, seed)
      -> scale_population(...)  REPLICATE up / SUBSAMPLE down whole household units
      -> run ActivitySim on the resampled population
      -> score trips vs VISTA targets (score_simulation.evaluate_simulation)

Whole household-person units are duplicated/removed with fresh IDs; NO individual
attribute is ever edited (defensible as materialised survey expansion weighting,
not the record-editing the reviewers rejected).

LAYERED ON THE 57 BASELINE (like HDR's --base-weights): the population is first
given the locked best_restore_faithful57 attribute weights (w_income/w_vot/cdap_*
on hh+persons, plus the fixed baseline land-use + skims), THEN resampled. At
group-weights == uniform the composition is preserved (a representative ~N/k
subsample) so the run reproduces composite ~57; SPE corrects the mix on top.

This driver REUSES hdr_driver's proven current-setup machinery (ActivitySim
invocation, score_simulation objective, baseline writers, data-safety) and the
clean spe_mechanism engines. It writes ONLY into a SEPARATE ``spe_run/`` tree so
it never collides with hdr_run/ (e.g. while the HDR forward runs are going).

Usage (asim env python):
    python spe_driver.py --smoke                       # identity weights -> ~57 check
    python spe_driver.py --optimizer cmaes --iters 40 --subsample 5000 --workers 2
    python spe_driver.py --optimizer cmaes --apply-best     # best -> FULL pop score
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hdr_driver as HD          # reuse: asim runner, scoring, baseline writers, clustering, data-safety
import hdr_mechanism as H        # HDRConfig
import spe_mechanism as SPE      # decode_parameters, hash_group, scale_population, make_optimiser

# SPE writes into its OWN run tree so it cannot collide with hdr_run/ worker dirs
# (the HDR forward optimizers may be running there in parallel).
SPE_RUN_ROOT = os.path.join(HERE, "spe_run")

DEFAULT_FAITHFUL = os.path.join(HD.RUN_ROOT, "best_restore_faithful57.npz")  # the locked 57 baseline
KMIN_DEFAULT, KMAX_DEFAULT = 4, 10
POPSIZE = 16
DEFAULT_ITERS = 40
DEFAULT_WORKERS = 2
FAIL = SPE.FAILURE_SCORE


# ---------------------------------------------------------------------------
# baseline preparation (the 57 reconstruction, materialised once)
# ---------------------------------------------------------------------------

def build_spe_base(hh_raw, lu_raw, persons, cfg, faithful_path, data_dir):
    """Materialise the locked 57 baseline into ``data_dir`` (a full ActivitySim
    data tree: baseline-weighted households/persons/land_use + weighted skims),
    and return the baseline-weighted (households_df, persons_df) for resampling.

    Reuses HDR's writers at HDR-weights == 0 composed on best_restore_faithful57,
    so the materialised population IS the 57 baseline. SPE only resamples the
    hh+persons afterwards; land_use + skims stay fixed at the baseline."""
    dk = int(np.load(faithful_path, allow_pickle=True)["k"]) if faithful_path else 8
    idc, label_by_id, k_hdr = HD.cluster_once(hh_raw, persons, dk)
    base_logw = HD._base_logw(faithful_path, k_hdr, cfg) if faithful_path else None
    dim_hdr = (k_hdr * (len(cfg.attrs) + len(cfg.person_attrs))
               + len(cfg.lu_attrs) + len(cfg.skim_attrs))
    w0 = np.zeros(dim_hdr)                       # HDR correction = 0 -> pure 57 baseline

    HD._mirror_baseline(data_dir)               # full tree minus hh/persons/land_use
    HD.write_weighted_inputs(hh_raw, lu_raw, persons, idc, label_by_id, k_hdr, w0,
                             data_dir, cfg, base_logw=base_logw)
    base_hh = pd.read_csv(os.path.join(data_dir, "households.csv"))
    base_persons = pd.read_csv(os.path.join(data_dir, "persons.csv"))
    return base_hh, base_persons


def _hh_id_name(df):
    for c in ("HHID", "household_id", "hhno", "HHHID"):
        if c in df.columns:
            return c
    raise KeyError("no household id column in households.csv")


# ---------------------------------------------------------------------------
# SPE's OWN grouping: seed/k-EXPLORED segmentation over a behaviour-aware feature
# space (demographics + each household's baseline travel signature). Preserves
# "Segmentation-based Population Exploration" — the optimiser still explores the
# partition via the seed and the number of segments via k — but the groups now
# have real leverage on the scored metrics, which random-ID hashing lacked.
# DISTINCT from PWS (fixed travel-sig KMeans + raking of FROZEN output) and HDR
# (fixed demographic clusters + continuous attribute weights): SPE *explores*
# projective segmentations, optimiser-scales them, and RE-SIMULATES.
# ---------------------------------------------------------------------------

_SEG_DEMOG = ["income", "PERSONS", "workers", "VEHICL"]
_DISC_PURP = {"shopping", "othmaint", "othdiscr", "eatout", "social"}  # the over-generated trips


def compute_seg_features(base_hh, idn, trips_path):
    """Per-household segmentation features aligned to base_hh ROW ORDER:
    standardised [demographics + baseline trip count + work/discretionary trip
    shares]. The travel signature is what lets a segmentation target the
    over-generated discretionary trips that inflate the baseline rate (9.3 vs
    8.0) — the handle the demographic-only lever lacked."""
    feats = base_hh[_SEG_DEMOG].fillna(0).to_numpy(float)
    ntrip = np.zeros(len(base_hh)); wshare = np.zeros(len(base_hh)); dshare = np.zeros(len(base_hh))
    if trips_path and os.path.exists(trips_path):
        t = pd.read_csv(trips_path, usecols=["household_id", "purpose"])
        idx = pd.Index(base_hh[idn].to_numpy())
        n = t.groupby("household_id").size().reindex(idx).fillna(0).to_numpy()
        w = t[t["purpose"] == "work"].groupby("household_id").size().reindex(idx).fillna(0).to_numpy()
        d = t[t["purpose"].isin(_DISC_PURP)].groupby("household_id").size().reindex(idx).fillna(0).to_numpy()
        ntrip = n
        wshare = w / np.clip(n, 1, None)
        dshare = d / np.clip(n, 1, None)
    X = np.column_stack([feats, ntrip, wshare, dshare])
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def feature_segment(seg_feats, k, seed):
    """k quantile bins along a SEED-selected random projection of the feature
    space. The seed is exactly what SPE's optimiser explores — different seeds
    give different (meaningful) segmentations; k sets the number of segments."""
    if k <= 1 or len(seg_feats) == 0:
        return np.zeros(len(seg_feats), dtype=int)
    rng = np.random.default_rng(int(seed) + 1)
    proj = rng.standard_normal(seg_feats.shape[1]); proj /= (np.linalg.norm(proj) + 1e-9)
    s = seg_feats @ proj
    edges = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
    return np.clip(np.digitize(s, edges), 0, k - 1)


# ---------------------------------------------------------------------------
# SPE candidate -> resampled population on disk
# ---------------------------------------------------------------------------

def write_spe_inputs(base_hh, base_persons, raw_params, kmin, kmax, data_dir,
                     seg_feats=None, grouping="feature", zone_col=None):
    """Decode the candidate, resample the baseline-weighted population by group
    weights, and overwrite households.csv + persons.csv in ``data_dir`` (whose
    land_use + skims are already the fixed 57 baseline). Returns the resampled
    household count (the honest denominator for the trip rate). Never writes
    outside the spe_run tree.

    grouping: "feature" (default) = SPE's seed/k-explored segmentation over the
    behaviour-aware feature space (compute_seg_features); "hash" = the original
    random-ID hash (kept for fidelity — but it has no leverage on the score).
    zone_col: if set, scale_population stratifies down-sampling by zone; DEFAULT
    None because the per-(zone x group) max(1,...) floor otherwise retains nearly
    everything on a sparse subsample and starves the optimiser of leverage."""
    HD._assert_run_dir(data_dir)
    k, seed, weights = SPE.decode_parameters(raw_params, kmin, kmax)

    idn = _hh_id_name(base_hh)
    # scale_population hardcodes 'household_id'; bridge raw HHID <-> household_id.
    work_hh = base_hh.rename(columns={idn: "household_id"}) if idn != "household_id" else base_hh.copy()
    if grouping == "hash" or seg_feats is None:
        groups = SPE.hash_group(work_hh["household_id"].to_numpy(), k, seed)
    else:
        groups = feature_segment(seg_feats, k, seed)
    zcol = zone_col if (zone_col and zone_col in work_hh.columns) else None
    p_out, hh_out = SPE.scale_population(base_persons, work_hh, weights, groups, zone_col=zcol)

    if idn != "household_id":
        hh_out = hh_out.rename(columns={"household_id": idn})   # back to asim raw schema
    hh_out.to_csv(os.path.join(data_dir, "households.csv"), index=False)
    p_out.to_csv(os.path.join(data_dir, "persons.csv"), index=False)
    return len(hh_out)


# ---------------------------------------------------------------------------
# parallel worker dirs (each a full copy of the 57-baseline data tree)
# ---------------------------------------------------------------------------

def setup_spe_workers(n, spe_base, tag):
    """n isolated worker data dirs, each a full copy of the materialised 57
    baseline (so land_use + skims are present and fixed). The per-candidate
    writer overwrites only households.csv + persons.csv."""
    import shutil
    workers = []
    for i in range(n):
        ddir = os.path.join(SPE_RUN_ROOT, f"{tag}_w{i}", "data")
        odir = os.path.join(SPE_RUN_ROOT, f"{tag}_w{i}", "output")
        sentinel = os.path.join(ddir, ".spe_ready")
        if not os.path.exists(sentinel):
            shutil.copytree(spe_base, ddir, dirs_exist_ok=True)
            open(sentinel, "w").close()
        workers.append({"data": ddir, "out": odir})
    return workers


def _norm_score(s):
    """Map a failed / non-finite / negative composite to the SPE failure score
    so the optimisers treat it as a dead candidate (composites are 0..100)."""
    return float(s) if (np.isfinite(s) and s >= 0) else FAIL


def evaluate_batch_spe(cands, base_hh, base_persons, kmin, kmax, workers, tagbase,
                       seg_feats=None, grouping="feature", zone_col=None):
    """Evaluate candidates, up to len(workers) ActivitySim runs at once."""
    n = len(workers)
    scores = [FAIL] * len(cands)
    metas = [None] * len(cands)
    for start in range(0, len(cands), n):
        chunk = list(range(start, min(start + n, len(cands))))
        running = []
        for slot, ci in enumerate(chunk):
            w = workers[slot]
            try:
                nhh = write_spe_inputs(base_hh, base_persons, cands[ci], kmin, kmax, w["data"],
                                       seg_feats, grouping, zone_col)
            except Exception as e:                       # degenerate resample -> dead candidate
                print(f"[spe]   candidate {ci} write failed: {e}")
                continue
            log_path = os.path.join(SPE_RUN_ROOT, f"asim_{tagbase}_c{ci}.log")
            running.append((ci, slot, nhh, HD.launch_activitysim(w["data"], w["out"], log_path)))
        for ci, slot, nhh, p in running:
            p.wait()
            p._logf.close()
            if p.returncode == 0:
                s, res = HD.score_run(workers[slot]["out"], nhh)
                scores[ci], metas[ci] = _norm_score(s), res
    return scores, metas


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SPE resampling correction driver (layered on 57 baseline)")
    ap.add_argument("--optimizer", choices=["cmaes", "ppo", "lstm"], default="cmaes")
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--popsize", type=int, default=POPSIZE)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="max concurrent ActivitySim runs")
    ap.add_argument("--kmin", type=int, default=KMIN_DEFAULT, help="min hash groups")
    ap.add_argument("--kmax", type=int, default=KMAX_DEFAULT, help="max hash groups (sets param dim = 2+kmax)")
    ap.add_argument("--subsample", type=int, default=0,
                    help="optimise on an N-household subsample of the baseline (faster)")
    ap.add_argument("--base-weights", dest="base_weights", default=DEFAULT_FAITHFUL,
                    help="npz of the locked 57 baseline to layer SPE on (default: best_restore_faithful57.npz). "
                         "Pass '' to run on the raw current data instead.")
    ap.add_argument("--smoke", action="store_true",
                    help="run ONE ActivitySim at uniform group weights and report the score (~57 check)")
    ap.add_argument("--apply-best", dest="apply_best", action="store_true",
                    help="apply saved best_spe_<optimizer>.npz to the FULL population and score")
    ap.add_argument("--zone-strat", dest="zone_strat", action="store_true",
                    help="stratify down-sampling by TAZ (preserves spatial structure but, on a "
                         "sparse subsample, the per-cell floor starves the optimiser of leverage). "
                         "Default OFF — group weights act at full strength.")
    ap.add_argument("--grouping", choices=["feature", "hash"], default="feature",
                    help="feature (default): SPE's seed/k-explored segmentation over demographics + "
                         "baseline travel signature (has leverage). hash: original random-ID hash (flat).")
    ap.add_argument("--baseline-trips", dest="baseline_trips",
                    default=os.path.join(HERE, "hdr_run", "output_best_restore_faithful57", "final_trips.csv"),
                    help="57-baseline final_trips.csv for per-household travel features (feature grouping)")
    ap.add_argument("--data", default=HD.BASELINE_DATA, help="baseline data dir")
    ap.add_argument("--configs", default=HD.CONFIGS, help="ActivitySim configs dir")
    args = ap.parse_args()
    zone_col = "TAZ" if args.zone_strat else None

    # route ALL HDR helper writes into the SPE run tree (separate from hdr_run/)
    HD.RUN_ROOT = SPE_RUN_ROOT
    HD.BASELINE_DATA = args.data
    HD.CONFIGS = args.configs
    os.makedirs(SPE_RUN_ROOT, exist_ok=True)

    cfg = H.HDRConfig()
    faithful = args.base_weights or ""
    if faithful and not os.path.isabs(faithful):
        faithful = os.path.join(HERE, "hdr_run", faithful) if not os.path.exists(faithful) else faithful
    tag = f"spe_{args.optimizer}"
    kmin, kmax = args.kmin, args.kmax
    dim = 2 + kmax                      # [k_norm, seed_norm, w_1..w_kmax]

    hh_raw, persons = HD.load_baseline()
    lu_raw = HD.load_landuse()
    print(f"[spe] baseline: {len(hh_raw)} households, {len(persons)} persons, {len(lu_raw)} zones")
    print(f"[spe] layering on {os.path.basename(faithful) if faithful else 'RAW current data (no 57 base)'}")

    # materialise the 57 baseline once; gives the baseline-weighted hh/persons to resample
    spe_base = os.path.join(SPE_RUN_ROOT, f"{tag}_base")
    print("[spe] materialising 57 baseline (HDR weights=0 on the reconstruction) ...")
    base_hh_full, base_persons_full = build_spe_base(hh_raw, lu_raw, persons, cfg, faithful, spe_base)
    idn = _hh_id_name(base_hh_full)
    seg_full = None
    if args.grouping == "feature":
        seg_full = compute_seg_features(base_hh_full, idn, args.baseline_trips)
        src = "with travel signature" if os.path.exists(args.baseline_trips) else "DEMOGRAPHICS ONLY (no baseline trips found)"
        print(f"[spe] grouping=feature: seed/k-explored segmentation over {seg_full.shape[1]} features ({src})")
    else:
        print("[spe] grouping=hash: original random-ID hashing (warning: flat, no leverage)")

    # ---- smoke: uniform group weights should reproduce ~57 (downsampled, proportions kept) ----
    if args.smoke:
        sdir = os.path.join(SPE_RUN_ROOT, "smoke_data")
        import shutil
        if not os.path.exists(os.path.join(sdir, ".spe_ready")):
            shutil.copytree(spe_base, sdir, dirs_exist_ok=True)
            open(os.path.join(sdir, ".spe_ready"), "w").close()
        raw0 = np.zeros(dim)            # weights -> uniform softmax; k -> mid; seed -> 0
        nhh = write_spe_inputs(base_hh_full, base_persons_full, raw0, kmin, kmax, sdir,
                               seg_full, args.grouping, zone_col)
        k0, seed0, w0 = SPE.decode_parameters(raw0, kmin, kmax)
        out_dir = os.path.join(SPE_RUN_ROOT, "smoke_output")
        print(f"[spe] SMOKE: k={k0} seed={seed0} uniform weights -> resampled {nhh} households; running ...")
        t0 = time.time()
        ok = HD.run_activitysim(sdir, out_dir, os.path.join(SPE_RUN_ROOT, "asim_smoke.log"))
        if not ok:
            print("[spe] SMOKE FAILED — see", os.path.join(SPE_RUN_ROOT, "asim_smoke.log"))
            sys.exit(1)
        score, res = HD.score_run(out_dir, nhh)
        print(f"[spe] SMOKE OK in {time.time()-t0:.0f}s  composite={score:.2f}  (expect ~57 = baseline)")
        if res:
            print(f"      trip_rate={res['trip_rate_actual']:.3f} (target {res['trip_rate_target']:.3f})  "
                  f"mode={res['mode_score']:.1f} purpose={res['purpose_score']:.1f} trip={res['trip_rate_score']:.1f}")
        return

    # ---- apply saved best to the FULL population ----
    if args.apply_best:
        bp = os.path.join(SPE_RUN_ROOT, f"best_{tag}.npz")
        if not os.path.exists(bp):
            print("[spe] no saved weights at", bp); sys.exit(1)
        d = np.load(bp, allow_pickle=True)
        raw = d["w"]; kmin, kmax = int(d["kmin"]), int(d["kmax"])
        adir = os.path.join(SPE_RUN_ROOT, f"best_{tag}_data")
        import shutil
        shutil.copytree(spe_base, adir, dirs_exist_ok=True)
        nhh = write_spe_inputs(base_hh_full, base_persons_full, raw, kmin, kmax, adir,
                               seg_full, args.grouping, zone_col)
        k1, seed1, w1 = SPE.decode_parameters(raw, kmin, kmax)
        out_dir = os.path.join(SPE_RUN_ROOT, f"output_best_{tag}")
        print(f"[spe] applying best {tag}: k={k1} seed={seed1} -> {nhh} households (FULL pop); running ...")
        ok = HD.run_activitysim(adir, out_dir, os.path.join(SPE_RUN_ROOT, f"asim_best_{tag}.log"))
        if not ok:
            print("[spe] full-pop run FAILED — see log"); sys.exit(1)
        score, res = HD.score_run(out_dir, nhh)
        print(f"[spe] FULL-POP best {args.optimizer}: composite={score:.2f}")
        if res:
            print(f"      trip_rate={res['trip_rate_actual']:.3f} (target {res['trip_rate_target']:.3f})  "
                  f"mode={res['mode_score']:.1f} purpose={res['purpose_score']:.1f} trip={res['trip_rate_score']:.1f}")
        return

    # ---- optimisation ----
    if args.subsample and args.subsample < len(base_hh_full):
        rng = np.random.default_rng(0)
        pick = rng.choice(len(base_hh_full), size=args.subsample, replace=False)
        base_hh = base_hh_full.iloc[pick].copy()
        seg = seg_full[pick] if seg_full is not None else None
        ids = set(base_hh[idn].tolist())
        base_persons = base_persons_full[base_persons_full["household_id"].isin(ids)].copy()
        print(f"[spe] optimising on a {len(base_hh)}-household subsample ({len(base_persons)} persons)")
    else:
        base_hh, base_persons, seg = base_hh_full, base_persons_full, seg_full

    opt = SPE.make_optimiser(args.optimizer, dim, args.popsize)
    n_workers = max(1, min(args.workers, args.popsize))
    workers = setup_spe_workers(n_workers, spe_base, tag)
    print(f"[spe] optimiser={args.optimizer} dim={dim} (2 + kmax={kmax})  "
          f"running up to {n_workers} ActivitySim evals in parallel")

    history = []
    best_score, best_w = FAIL, None
    t_start = time.time()
    for it in range(args.iters):
        cands = opt.ask()
        scores, metas = evaluate_batch_spe(cands, base_hh, base_persons, kmin, kmax,
                                           workers, tagbase=f"{tag}_it{it}",
                                           seg_feats=seg, grouping=args.grouping, zone_col=zone_col)
        opt.tell(cands, scores)
        w_best, s_best = opt.best_solution
        if s_best > best_score:
            best_score, best_w = s_best, np.asarray(w_best).copy()
            np.savez(os.path.join(SPE_RUN_ROOT, f"best_{tag}.npz"),
                     w=best_w, kmin=kmin, kmax=kmax, score=best_score)
        rec = {"iter": it, "best": float(best_score),
               "iter_mean": float(np.mean(scores)), "iter_max": float(np.max(scores))}
        history.append(rec)
        with open(os.path.join(SPE_RUN_ROOT, f"history_{tag}.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"[spe] it {it:3d}/{args.iters}  best={best_score:6.2f}  "
              f"iter_max={np.max(scores):6.2f}  iter_mean={np.mean(scores):6.2f}  "
              f"elapsed={time.time()-t_start:.0f}s")
        if opt.converged:
            print("[spe] optimiser converged."); break

    print(f"[spe] DONE  best composite = {best_score:.2f}")
    print(f"[spe] best saved -> {os.path.join(SPE_RUN_ROOT, f'best_{tag}.npz')}  "
          f"(run --apply-best for the full-pop number)")


if __name__ == "__main__":
    main()
