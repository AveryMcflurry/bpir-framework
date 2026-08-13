# Reproducing the paper's results

This folder contains the exact experiment code used in the paper, with the
parameters left at the values reported there (every driver also exposes them as
command-line flags). All commands below are run from `experiments/HDR_Revise/`
inside the conda environment defined by `../environment.yml`.

Hardware used throughout: a single desktop workstation (Windows 11, 16 cores,
64 GB RAM; peak usage 8.5 GB). No non-standard hardware. GPU not required.

| Experiment | Command | Expected result | Runtime |
|---|---|---|---|
| **Demo: Melbourne baseline** | `python reproduce_baseline.py` | 5-mode 59.8 ± 0.5 (paper full-pop 58.97); 7-mode 58.2 ± 0.5 (57.46) | **~3 min** |
| **San Francisco, small (synthetic-truth)** | see §2 | degraded 89.2 → HDR recovers; weight readout matches the injected error | ~1–2 h |
| **San Francisco, full scale** | see §3 | baseline 94.99 → PWS 98.06 → HDR 98.38 (97.12 on unseen 45k) | overnight |
| Melbourne PWS floor | see §4 | 66.25 (5-mode) from the baseline output | seconds |
| Melbourne / VITM optimisation campaigns | see §5 | 

## 1. Install (10–15 min)

```bash
conda env create -f environment.yml
conda activate bpir
```

Tested with Python 3.9.18 / ActivitySim 1.1.2 on Windows 11; the stack is
platform-independent conda-forge + pip packages.

## 2. San Francisco — small demonstration (openly downloadable data)

The 25-zone MTC example ships with ActivitySim. The synthetic-truth experiment
injects a documented input-side error and shows BPIR removing it, with the
learned weights recovering the injected channels:

```bash
python -m activitysim create -e prototype_mtc -d ../examples
python mtc_sf_demo.py --targets      # ~2 min: synthetic-truth targets + 20% hold-out
python mtc_sf_demo.py --degrade      # ~2 min: injected input error -> baseline 89.2
python mtc_sf_demo.py --correct --iters 20 --workers 3    # ~1 h
python mtc_sf_demo.py --apply-best   # scores full population + the 20% hold-out
python mtc_sf_demo.py --pws
python mtc_sf_demo.py --report       # summary + weight-recovery diagnostic
```

Each ActivitySim evaluation takes ~61 s.

## 3. San Francisco — full-scale (Supplementary Note 4)

Downloads the full 1,454-zone example (~1 GB) from the ActivitySim consortium,
then reproduces the supplementary results against the observed Bay Area
marginals in `mtc_full_run/mtc_full_targets.json` (provenance in the file:
MTC's 2012 Travel Model One calibration & validation report, Tables 55 and 8).

```bash
python -m activitysim create -e prototype_mtc_full -d ../examples
python build_sf_configs.py           # stock configs + BPIR hooks (scripted, ~1 min)
python mtc_full_demo.py --baseline   # ~6 min  -> composite 94.99 (trip 99/purpose 78/mode 92)
python mtc_full_demo.py --correct    # overnight -> 98.38 (10 CMA-ES iterations)
python mtc_full_demo.py --pws        # seconds  -> 98.06
python mtc_full_demo.py --verify     # ~15 min  -> 97.12 on a disjoint 45,000-household sample
python mtc_full_demo.py --report     # summary + learned weight readout
```

Full-scale evaluations: ~4.4 min per 15,000-household run, 8.5 GB peak RAM.
Reference outputs from the paper's own runs are included in `mtc_full_run/`
(`*_score.json`, `history_*.json`) for direct comparison.

## 4. Melbourne — deposited baseline and mechanisms

`../melbourne/data_baseline` is the paper's Melbourne baseline as ready-to-run
input data (see `../melbourne/README.md` for the integrity statement), and
`../MainTrain/configs_mel` is the complete model configuration including every
coefficient file. `reproduce_baseline.py` (the 3-minute demo above) verifies the
deposit against the paper's baseline scores. The Post-hoc Weighting Scheme can
be applied to any baseline run output (`pws_driver.py`); the full optimisation
campaigns (HDR/SPE under CMA-ES, PPO and the LSTM meta-optimiser, and the
six-lever SPSA benchmark) are the drivers in this folder, run with their
defaults as reported in Methods.

## 5. Shipped campaign artifacts

Re-running every optimisation campaign takes machine-days; the paper's own run
records are therefore included for inspection and plotting:

- `../melbourne/results/` — per-optimiser convergence histories and best weight
  vectors for HDR and SPE, the SPSA benchmark history, and the PWS trajectory.
- `../vitm_results/` — the corresponding histories for the official VITM model.
  The VITM trial model and its input data are confidential and cannot be
  redistributed (see the paper's Data Availability statement); the code that ran
  on it (`*_vitm.py`) is included in full.

## 6. Running on your own model

The mechanisms need (1) an ActivitySim-compatible model, (2) the weight hooks
added to its configuration (five files — `build_sf_configs.py` documents every
edit), and (3) observed aggregate marginals: one trip rate, a purpose
distribution and a mode-share distribution. See the top-level README.
