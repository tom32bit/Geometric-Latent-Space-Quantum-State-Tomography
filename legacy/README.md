# Code and outputs of the submitted version

This folder holds the code and outputs of the first submitted version of the paper (archived on Zenodo as [10.5281/zenodo.20096033](https://doi.org/10.5281/zenodo.20096033)). The files are unchanged apart from their location.

The trained models here are still the main models of the revised paper. The scripts in [`../revision/`](../revision/) re-evaluate them on fixed held-out test sets. **Do not use the evaluation code in this folder for new results.** It has the known issues listed below, and `revision/` replaces it.

## What is here

| Path | What it holds | Use in the revised paper |
|---|---|---|
| `reviewer_experiments/run_all_experiments.ipynb` | The 2- and 3-qubit seed study (seeds 42, 123, 256, 789, 1024), the latent-dimension ablation, the original 3 × 3 loss-weight sweep, and the per-channel, circuit and float64 checks. Outputs are in `run_all_experiments/`. | The seed models (`model_{2,3}q_seed*.pt`) are the main models. Their logged validation fidelities (`seed_variance_{2,3}q_results.json`) give Table 2's data-sufficient column. `latent_dim_ablation_3q.json` gives Table 12. `float64_check_3q.json` is used in the Numerical Validation subsection. |
| `reviewer_experiments/run_4q_experiments.ipynb` | The 4-qubit seed study (seeds 42, 123, 256). Outputs are in `run_4q_experiments/`. | As above, for 4 qubits. |
| `reviewer_experiments/run_4q_exp4_exp6.ipynb` | The 4-qubit per-channel and float64 checks. Outputs are in `run_4q_experiments/notebook c/`. | `float64_check_4q.json` is used in the Numerical Validation subsection. |
| `reviewer_experiments/exp*.py` | Stand-alone versions of the notebook experiments. | — |
| `tomography.py`, `4q_5000.py` | Single training runs: the data-limited runs (`{n} Qubit Results n_train=5000/`; `4q_5000.py` for 4 qubits) and the data-sufficient runs (`{n}q saturated/`). Each folder holds the model, the training history and the log. | The validation fidelities of the data-limited runs give Table 2's 5k column. `4q saturated/training_history_4q.pkl` gives Figure 2. |
| `tomography.ipynb` | An interactive version of `tomography.py`. It also needs `pennylane` and `tqdm`. | — |
| `generate_paper_figures.py` | The figures of the submitted version, from hard-coded values. | Figure 4 only (`fig_data_sufficiency()`). |

The dataset caches that `tomography.py` writes (`dataset_*_cache.npz`) are not included; the 4-qubit one is 170 MB.

## Known issues

1. **Shot-noise evaluation.** This affects `evaluate_shot_noise` in `tomography.py`, `exp1_seed_variance.py` and the notebooks.
   - Noise is added to the model's *output* (the predicted Pauli vector), not to its input.
   - The noisy vector is turned into a density matrix without projecting it onto physical states.
   - The fidelity (`fidelity_numpy`) is not capped at 1.

   Even a perfect predictor therefore scores F > 1 at 1,000 shots or fewer. This produced the apparent fidelity *gain* at 100 shots reported in the submitted version. The corrected evaluation, with binomial noise on the encoder input, is `revision/run_analyses.py shots`. Under it, fidelity falls monotonically with fewer shots. `revision/nbcore.py` contains this function verbatim, as part of the copied notebook code, but no revision analysis calls it.
2. **Geometry metrics on the full latent vector.** The notebooks and `exp1_seed_variance.py` compute the latent–Bures correlation, PCA dimension and curvature on the full latent vector z, which includes the two purity dimensions. Their pairs are drawn from training and validation data. The revision uses the geometric part z_geom, on pairs of held-out test states.
3. **Unseeded data generation.** The joblib workers that generate states are not seeded. The training and validation sets of these runs therefore cannot be regenerated, although the model seeds fix initialization and batch order. The revision generates all new data with `nbcore.build_dataset_seeded`.
4. **`tomography.py` data generator.**
   - `purity_to_depol_param` solves the purity equation with incorrect coefficients, so its depolarized and isotropic states miss their target purity. The notebooks solve it numerically.
   - The default `n_qubits` is 5.
5. **Generator names** (all versions; described in Section 2.1 of the paper).
   - "Depolarized" and "Isotropic" are the same generator: a Haar-random pure state mixed with white noise.
   - "Werner" states are GHZ states mixed with white noise. The Werner weight comes from an approximate closed form, so Werner purities are shifted relative to the purity window.

   `revision/nbcore.py` keeps these generators unchanged, so new data matches the data the models were trained on.
6. **`exp3_pareto_validation.py`** is hard-coded to 2 qubits, although the notebook ran the 3-qubit grid. The revised paper replaces that 3 × 3 sweep with a 12-configuration grid trained on seeded data (`revision/kaggle_lambda_m_ablation.ipynb`, group `2q_pareto`).
7. **`generate_paper_figures.py`** draws the submitted figures from hard-coded numbers, several of which the revision corrects.
