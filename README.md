# Geometric Latent Space Quantum State Tomography with Metric-Preserving Autoencoders

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24108878.svg)](https://doi.org/10.5281/zenodo.24108878)

Code, held-out test sets, trained models and result files for the paper

> **Geometric Latent Space Quantum State Tomography with Metric-Preserving Autoencoders**
> S.M. Yousuf Iqbal Tomal and Abdullah Al Shafin
> *Machine Learning: Science and Technology* (IOP Publishing), under review.

This is version 2 of the code, for the revised manuscript, archived on Zenodo as [10.5281/zenodo.24108878](https://doi.org/10.5281/zenodo.24108878). The code and outputs of the first submitted version (version 1, archived as [10.5281/zenodo.20096033](https://doi.org/10.5281/zenodo.20096033)) are kept in [`legacy/`](legacy/).

**Authors**

- **S.M. Yousuf Iqbal Tomal** (corresponding author): [yousuf.iqbal.tomal@g.bracu.ac.bd](mailto:yousuf.iqbal.tomal@g.bracu.ac.bd) · [ORCID 0009-0000-3391-3824](https://orcid.org/0009-0000-3391-3824)
- **Abdullah Al Shafin**: [abdullah.al.shafin@g.bracu.ac.bd](mailto:abdullah.al.shafin@g.bracu.ac.bd) · [ORCID 0009-0007-6634-6900](https://orcid.org/0009-0007-6634-6900)

Department of Computer Science and Engineering, BRAC University, Dhaka, Bangladesh.

---

## Summary

A classical neural encoder maps the Pauli expectation values of an n-qubit state to a latent vector with a geometric part and a purity part. A parameterized quantum circuit with learned depolarizing noise decodes the latent vector into a density matrix. Training combines the Uhlmann fidelity with a Pearson-correlation metric loss on the geometric part and purity-matching losses. One architecture, scaled with n, is used with the same loss weights at 2, 3 and 4 qubits.

Results on held-out test states (7,000 per qubit count; mean ± sd over 5, 5 and 3 seeds):

| Qubits | Fidelity | Latent–Bures Pearson r, all pairs | r without Werner states |
|:------:|:-------------:|:-----:|:-----:|
| 2 | 0.979 ± 0.003 | 0.919 | 0.854 |
| 3 | 0.924 ± 0.003 | 0.795 | 0.425 |
| 4 | 0.834 ± 0.001 | 0.821 | 0.257 |

- **Werner states carry the correlation at 3 and 4 qubits.** They form a near-degenerate family, and excluding them leaves a much weaker correlation.
- **Data, not the circuit, limits fidelity.** 4-qubit fidelity rises by 0.14 when the training set grows from 20 to 300 samples per free parameter. Fitting the same circuit to each state separately reaches F = 0.996, 0.992 and 0.987.
- **The metric loss is a trade-off.** A paired ablation without it (λ_m = 0) shows that it raises the correlation at every qubit count, at a fidelity cost of at most 0.013. Without Werner states, r rises from 0.43 to 0.83 at 2 qubits but only from 0.22 to 0.26 at 4 qubits.
- **Shot noise.** With simulated shot noise on the measured input, fidelity falls monotonically with fewer shots, by at most 2% at 1,000 shots or more.
- **Scope.** The input, the training data and the training cost all grow exponentially with the number of qubits, so these results are a proof of concept for small systems.

---

## Repository layout

```
revision/                        analyses of the revised paper (everything below runs on a CPU)
├── nbcore.py                    model, data and training code of the original notebooks, copied
│                                verbatim, plus seeded data generation and model loading
├── build_nbcore.py              regenerates nbcore.py from the notebooks in legacy/
├── make_testsets.py             builds the fixed test sets in data/
├── qst_eval.py                  shared helpers: data, inference, float64 fidelity, metrics
├── run_analyses.py              the analyses of the trained models (results/*.json)
├── kaggle_lambda_m_ablation.ipynb   GPU notebook: the λ_m = 0 ablation and the loss-weight grid
├── build_kaggle_notebook.py     regenerates that notebook
├── process_kaggle.py            checks and evaluates the Kaggle runs
├── eval_table2.py               the Table 2 models on the test sets
├── acceptance_check.py          do the archived models reproduce their logged fidelities?
├── make_figures.py              the figures of the revised paper (figures/*.pdf)
├── tests/test_nbcore.py         sanity tests for nbcore.py
├── data/                        test sets (testset_{n}q.npz) and purity-bin sets (ood_{n}q.npz)
├── kaggle_outputs/              models and run records of the Kaggle runs
├── results/                     result files and run logs
└── figures/                     generated figures
legacy/                          code and outputs of the submitted version; see legacy/README.md
```

The main models are the seed models of the submitted version, in `legacy/reviewer_experiments/run_*_experiments/model_{n}q_seed*.pt`. `revision/` evaluates them again on the fixed test sets. **The evaluation code in `legacy/` has known issues, including a flawed shot-noise evaluation, and is kept only for provenance** ([details](legacy/README.md#known-issues)).

---

## Requirements

Python 3.12 with the pinned packages:

```bash
pip install -r requirements.txt
```

All analyses run on a CPU. Retraining needs a GPU: the Kaggle runs used one NVIDIA T4, and each `run_*.json` records its software versions.

---

## Reproducing the results

Run every command from the repository root.

```bash
python revision/tests/test_nbcore.py       # sanity tests, about a minute
python revision/acceptance_check.py        # archived models vs their logged fidelities, on the test sets

python revision/run_analyses.py all        # all analyses of the main models, about 30 min on a laptop CPU
python revision/process_kaggle.py          # λ_m ablation and loss-weight grid from revision/kaggle_outputs/
python revision/eval_table2.py             # data-limited vs data-sufficient models on the test sets
python revision/make_figures.py            # all figures; needs results/cache/ from run_analyses.py
```

- `run_analyses.py` writes `revision/results/<analysis>_original.json` and caches latents in `revision/results/cache/`, which is not in the repository. A single analysis can be rerun with, for example, `python revision/run_analyses.py shots`.
- `make_testsets.py` rebuilds the test sets from fixed seeds. They are included, so this step is optional.
- To retrain the λ_m ablation and the loss-weight grid, run `revision/kaggle_lambda_m_ablation.ipynb` on Kaggle, one group per session; the notebook explains the groups. All groups together take about 31 GPU-hours.

### Where each table and figure comes from

Numbers refer to the revised manuscript.

| Item | Content | Produced by | Result file(s) in `revision/` |
|---|---|---|---|
| Table 1, Table 10 | Fidelity per qubit count and per generator | `run_analyses.py fidelity` | `results/fidelity_original.json` |
| Table 2 | Data-limited vs data-sufficient models | Logged validation fidelities of the submitted runs; test-set check by `eval_table2.py` | `legacy/` logs and `seed_variance_*_results.json`; `results/table2_testset.json` |
| Tables 3, 4, 5 | Latent–Bures correlations, PCA dimension, local anisotropy ratio κ | `run_analyses.py geometry` | `results/geometry_original.json` |
| Table 6 | Loss-weight grid (3 qubits) | Kaggle notebook (group `2q_pareto`), `process_kaggle.py` | `results/pareto_grid.json` |
| Table 7 | Effect of the metric loss (paired λ_m = 0 vs 0.2 runs) | Kaggle notebook, `process_kaggle.py` | `results/ablation_summary.json`, `fidelity_ablation.json`, `geometry_ablation.json` |
| Tables 8, 9 | Shot noise | `run_analyses.py shots` | `results/shots_original.json` |
| Table 11 | Fidelity outside the training purity window | `run_analyses.py ood` | `results/ood_original.json` |
| Table 12 | Latent-dimension ablation (3 qubits) | Submitted version | `legacy/reviewer_experiments/run_all_experiments/latent_dim_ablation_3q.json` |
| Table 13 | Circuit parameters vs state-space dimension | `run_analyses.py oracle` | `results/oracle.json` (`ratio_table`) |
| Tables 14–17 | Settings, notation, components, comparison | Text | — |
| Figure 1 | Architecture | Drawn in the manuscript | — |
| Figure 2 | 4-qubit training dynamics | `make_figures.py` | `figures/fig2_training_dynamics_4q.pdf`, from `legacy/4q saturated/training_history_4q.pkl` |
| Figure 3 | Scaling | `make_figures.py` | `figures/fig3_scaling.pdf` |
| Figure 4 | Data sufficiency | Submitted version (`legacy/generate_paper_figures.py`) | — |
| Figure 5 | Latent vs Bures distance (3 qubits) | `make_figures.py` | `figures/fig5_latent_bures_3q.pdf` |
| Figures 6, 10, 11, 12 | Latent space (t-SNE, PCA) | `run_analyses.py latentviz`, `make_figures.py` | `results/latentviz_{n}q.npz`; `figures/fig_latent_*.pdf` |
| Figure 7 | Latent geometry across qubit counts | `make_figures.py` | `figures/fig6_manifold.pdf` |
| Figure 8 | Loss-weight grid | `make_figures.py` | `figures/fig7_pareto.pdf` |
| Figure 9 | Shot noise | `make_figures.py` | `figures/fig8_shot_noise.pdf` |
| Text | Per-state circuit fits; refinement from the encoder's prediction; latent-distance timing | `run_analyses.py oracle`, `warmstart`, `timing` | `results/oracle.json`, `warmstart.json`, `timing.json` |
| Text | Numerical validation (float32 vs float64) | Submitted version | `legacy/reviewer_experiments/run_*_experiments/**/float64_check_*.json` |

### Data

- **Test sets.** 7 generators × 1,000 states per qubit count, purity 0.85–0.95 (seed 303).
- **Purity-bin sets.** Depolarized states in six purity bins from 0.30 to 1.00, 500 per bin (seed 404).
- **Kaggle training data.** Generated with fixed seeds: 101 and 202 for the ablation, 505 and 606 for the grid. Each run record stores a hash of its training and validation data.
- **Original training data.** The data of the original seed models cannot be regenerated, because the original generator was not seeded ([legacy/README.md](legacy/README.md#known-issues)).

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{tomal2026geometric,
  author  = {Tomal, S. M. Yousuf Iqbal and Al Shafin, Abdullah},
  title   = {Geometric Latent Space Quantum State Tomography with Metric-Preserving Autoencoders},
  journal = {Machine Learning: Science and Technology},
  year    = {2026},
  note    = {Under review}
}
```

To cite the code itself, use [10.5281/zenodo.24108878](https://doi.org/10.5281/zenodo.24108878) (version 2.0.0); its metadata is also in [`CITATION.cff`](CITATION.cff).

## License

The code is released under the [MIT License](LICENSE).
