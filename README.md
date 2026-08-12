# BPIR - Behaviour-Preserving Input-attribute Reweighting

Reference implementation of the BPIR framework for adapting transferred
activity-based travel models (ABMs) by weighting input attributes, while the
input data records and every behavioural coefficient stay unchanged.

Companion code for: Ma, Rashidi, Najmi, "Weighting input attributes rather
than coefficients reaches travel-model errors that calibration cannot".

## What is here

High-level, simulator-agnostic implementations of the core logic. Every module
expects a user-supplied `simulate(inputs) -> trips` callable (any ABM works;
the paper uses ActivitySim) and a `score(trips) -> float` objective. Tuning
constants and engine-specific hooks are deliberately left out; see the paper's
Methods for the exact experimental settings.

| Module | Mechanism |
|---|---|
| `bpir/objective.py` | Composite log-cosh objective (trip rate, purpose, mode) |
| `bpir/pws.py` | PWS: post-hoc bounded IPF reweighting of simulated households |
| `bpir/spe.py` | SPE: group resampling of households, re-simulated |
| `bpir/hdr.py` | HDR: bounded attribute weights on population, land use, skims |
| `bpir/spsa.py` | SPSA coefficient-calibration benchmark |
| `bpir/rf_benchmark.py` | Chained Random-Forest travel-diary emulator benchmark |
| `bpir/optimizers.py` | ask/tell optimiser interface (CMA-ES; PPO/LSTM pluggable) |

## Principles enforced in code

1. Input records are never edited; weights are attached to attributes and act
   at run time inside the simulation.
2. Weights are bounded to calibration-conventional magnitudes.
3. Setting all weights to one restores the original model exactly.
4. Evaluation belongs on held-out records the optimiser never saw.

## Citation

Ma, Rashidi and Najmi (2026), BPIR framework,
https://github.com/AveryMcflurry/bpir-framework


## System requirements

Python 3.9 or later with `numpy`; `cma` is optional but recommended (it enables
the CMA-ES engine). Pure Python, no non-standard hardware; any OS. Tested with
Python 3.10 on Windows 11.

## Installation guide

```
git clone https://github.com/AveryMcflurry/bpir-framework
cd bpir-framework
pip install numpy cma
```

Typical install time on a normal desktop computer: under one minute.

## Demo

```
python examples/demo.py
```

The demo optimises bounded attribute weights against a small synthetic
simulator (no data required). Expected output: the composite score printed at
baseline, rising above 95 after 30 iterations, followed by the learned weight
factors. Expected run time: seconds on a normal desktop computer.

## Instructions for use

Each mechanism takes a `simulate(inputs) -> trips` callable and a
`score(trips) -> float` objective. To run on your own model, implement
`simulate` around your ABM (the paper uses ActivitySim), build the composite
objective from your observed targets with `bpir/objective.py`, and call the
mechanism loops in `bpir/pws.py`, `bpir/spe.py` or `bpir/hdr.py`. The
preprocessed author-built Melbourne inputs accompany this repository; the
VITM model and its inputs are confidential and not included.
