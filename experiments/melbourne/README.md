# The deposited Melbourne baseline

`data_baseline/` contains the author-built Melbourne model inputs exactly as
scored in the paper (composite 58.97 five-mode / 57.46 seven-mode at full
population), preprocessed and ready to run with `../MainTrain/configs_mel`.

## What the files contain

- `households.csv`, `persons.csv` — the 30,195 households / 77,428 persons
  derived from the public VISTA survey (preprocessing in Supplementary Note 1),
  carrying the framework's k-means segmentation columns (`w_vot`,
  `cdap_work_w`, `cdap_nonmand_w`). These columns are part of the BPIR input
  schema — the correction mechanisms require them on any dataset — and here
  they carry the per-segment terms of the deposited baseline state. `w_income`
  is 1.0 throughout (the income state is expressed in the values themselves).
- `land_use.csv` — the 32-zone land-use table (construction hypotheses in
  Supplementary Note 2), with neutral (`= 1.0`) weight columns.
- `omx/skims.omx` — the assembled impedance matrices (Supplementary Note 2).

No record is synthetic beyond what population synthesis from the public VISTA
survey implies, and the model configuration (`../MainTrain/configs_mel`,
including every coefficient file) is deposited unchanged and in full.

## Verification

`baseline_verification.json` is the deposit's receipt. To re-verify on your
machine (~3 minutes):

```bash
cd ../HDR_Revise
python reproduce_baseline.py
```

Expected: 5-mode composite 59.8 ± 0.5 on the 5,000-household verification
subsample (58.97 at full population), trip rate ≈ 9.1–9.2 against the observed
8.03 — the over-generating, mode-distorted baseline the paper corrects.

## Results artifacts

`results/` holds the paper's own optimisation records for this dataset:
convergence histories (`history_*.json`) and best weight vectors
(`best_*.npz`) for HDR and SPE under all three optimisers, the SPSA benchmark
trajectory, and the PWS raking trajectory. Figure 5 and Supplementary Fig. 2
plot exactly these files.
