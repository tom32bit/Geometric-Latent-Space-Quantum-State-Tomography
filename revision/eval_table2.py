"""Table 2 on the held-out test sets: data-limited (N_train = 5,000) vs data-sufficient models.

Data-limited models: the single runs in 'legacy/{n} Qubit Results n_train=5000/best_model_{n}q.pt'
(produced by legacy/tomography.py and legacy/4q_5000.py). Data-sufficient: the archived seed
models. Both are evaluated with the same float64 fidelity on revision/data/testset_{n}q.npz.
If a run's dataset cache is present (it is not in the repository), the script also checks
that the model reproduces the validation fidelity logged in that run's log.txt.

Usage:  python revision/eval_table2.py
"""
import json
import os

import numpy as np
import torch

import qst_eval as qe
from qst_eval import nb


LOGGED_VAL_F = {2: None, 3: 0.9167, 4: 0.6939}   # 'Final val: mean=' in each run's log.txt


def load_learned_weights(n, path):
    """tomography.py stores the circuit's constant gate matrices under other buffer
    names; only the learned weights are loaded into the original HybridAE."""
    nb.ensure_basis(n)
    model = nb.HybridAE(nb.n_pauli_nonid, 10 * n, nb.n_angles, nb.n_noise_params,
                        nb.full_paulis_arr, nb.pauli_idx_nonid)
    state = torch.load(path, map_location='cpu', weights_only=True)
    learned = {k: v for k, v in state.items() if not k.startswith('circuit.')}
    missing, unexpected = model.load_state_dict(learned, strict=False)
    assert all(k.startswith('circuit.') for k in missing) and not unexpected, (missing, unexpected)
    return model.eval()


def main():
    out = {}
    fid = json.load(open(os.path.join(qe.RESULTS, 'fidelity_original.json')))
    for n in (2, 3, 4):
        ts = qe.load_testset(n)
        folder = os.path.join(qe.LEGACY, f'{n} Qubit Results n_train=5000')
        model = load_learned_weights(n, os.path.join(folder, f'best_model_{n}q.pt'))
        cache_path = os.path.join(folder, f'dataset_{n}q_cache.npz')
        own_val = None
        if os.path.exists(cache_path):
            cache = np.load(cache_path)
            own_val, _, _ = nb.evaluate_fidelity(model, cache['X_val'], cache['Rho_val'])
            if LOGGED_VAL_F[n] is not None:
                assert abs(own_val - LOGGED_VAL_F[n]) < 2e-3, f'{n}q: circuit not equivalent ({own_val} vs {LOGGED_VAL_F[n]})'
            print(f'{n}q: own validation set F {own_val:.4f} (logged {LOGGED_VAL_F[n]})')
        else:
            print(f'{n}q: no dataset cache, own-validation check skipped (logged {LOGGED_VAL_F[n]})')
        F = qe.fidelity64(ts['Rho'], qe.infer(model, ts['X'])['rho'])
        suff = fid[f'{n}q']['summary']['original']['mean']
        out[f'{n}q'] = {'data_limited_own_val_F': own_val, 'data_limited_logged_val_F': LOGGED_VAL_F[n],
                        'data_limited_F': F.mean(), 'data_limited_F_no_werner': F[ts['labels'] != qe.WERNER].mean(),
                        'data_sufficient_F_seed_mean': suff['mean'], 'data_sufficient_F_sd': suff['sd'],
                        'delta_F': suff['mean'] - F.mean(),
                        'samples_per_parameter_5k': 5000 / (4 ** n - 1),
                        'note': 'data-limited: single run (tomography.py generator), batch 64; '
                                'data-sufficient: seed study, batch 64/256/512'}
        print(f'{n}q: 5k {F.mean():.4f} | 300 s/p {suff["mean"]:.4f} ± {suff["sd"]:.4f} | ΔF {suff["mean"] - F.mean():+.4f}')
    with open(os.path.join(qe.RESULTS, 'table2_testset.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
