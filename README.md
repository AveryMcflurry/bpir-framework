# BPIR framework

Code for the paper "Weighting the input attributes of city travel models reaches
hidden errors that calibration cannot" (Ma, Rashidi and Najmi).

BPIR (Behaviour-Preserving Input-attribute Reweighting) implements attribute-weight
correction through three mechanisms: the Post-hoc Weighting Scheme (PWS),
Segmentation-based Population Exploration (SPE) and Hierarchical Disaggregate-input
Reweighting (HDR). The repository also holds the calibration benchmark and the
emulator benchmark.

## Files

| Path | Content |
|---|---|
| `bpir/` | the mechanisms, the objective, the search methods and the two benchmarks, written independently of the simulator |
| `demo.py` | the core package on a synthetic simulator |
| `pws_mechanism.py`, `spe_mechanism.py`, `hdr_mechanism.py` | the three mechanisms as used with ActivitySim |
| `pws_driver.py`, `spe_driver.py`, `hdr_driver.py` | drivers for the transferred model |
| `pws_vitm.py`, `spe_vitm.py`, `hdr_driver_vitm.py` | drivers for VITM |
| `spsa_mel.py`, `spsa_vitm_tour_asc.py` | calibration benchmark on the transferred model and on VITM |
| `rf_framework_train_and_simulate.py` | emulator benchmark |
| `score_simulation.py`, `score_vitm.py` | composite score |
| `run_baseline.py` | one run of a model with every weight at its neutral value |
| `build_sf_configs.py`, `mtc_full_demo.py` | San Francisco Bay Area example |
| `melb/configs/` | the configuration files through which the weights enter the model |
| `sf/mtc_full_targets.json` | observed targets for the San Francisco Bay Area example |
| `cost_utils.py` | run time and memory of each model run |

## Environment

```
conda env create -f environment.yml
conda activate bpir
```

## San Francisco Bay Area example

The example runs on public data from the ActivitySim project:

```
python -m activitysim create -e prototype_mtc_full -d download
```

Copy the `configs` and `data` folders of `download/prototype_mtc_full` into `sf/`, then:

```
python build_sf_configs.py
python mtc_full_demo.py --baseline
python mtc_full_demo.py --correct
python mtc_full_demo.py --pws
python mtc_full_demo.py --verify
python mtc_full_demo.py --report
```

## Data

No data are included. The inputs of the transferred model are not part of this
repository, and the VITM trial model and its input data are confidential.
