"""Do the archived seed models reproduce their published fidelities on the new test sets?

The original evaluate_fidelity() (complex64, as in the paper) is applied to the
13 archived seed models on revision/data/testset_{n}q.npz. Each seed-mean is
compared with the seed-mean of the models' own validation fidelities in
seed_variance_{n}q_results.json. Tolerances: 0.005 at 2-3 qubits, 0.01 at 4.

Usage:  python revision/acceptance_check.py
"""
import json
import os
import sys

REV = os.path.dirname(os.path.abspath(__file__))
LEGACY = os.path.join(os.path.dirname(REV), 'legacy')
os.environ.setdefault('QST_OUTDIR', os.path.join(REV, 'results', 'scratch'))
sys.path.insert(0, REV)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import nbcore as nb  # noqa: E402

torch.set_num_threads(8)
SEEDS = {2: [42, 123, 256, 789, 1024], 3: [42, 123, 256, 789, 1024], 4: [42, 123, 256]}
TOL = {2: 0.005, 3: 0.005, 4: 0.01}


def ckpt(n, s):
    sub = 'run_4q_experiments' if n == 4 else 'run_all_experiments'
    return os.path.join(LEGACY, 'reviewer_experiments', sub, f'model_{n}q_seed{s}.pt')


def main():
    out, ok = {}, True
    for n in (2, 3, 4):
        d = np.load(os.path.join(REV, 'data', f'testset_{n}q.npz'))
        sub = 'run_4q_experiments' if n == 4 else 'run_all_experiments'
        logged = {r['seed']: r['mean_fidelity'] for r in json.load(open(os.path.join(
            LEGACY, 'reviewer_experiments', sub, f'seed_variance_{n}q_results.json')))}
        rows = []
        for s in SEEDS[n]:
            model = nb.load_model(n, ckpt(n, s))
            mean_f, _, fids = nb.evaluate_fidelity(model, d['X'], d['Rho'])
            rows.append({'seed': s, 'test_F': mean_f, 'logged_val_F': logged[s],
                         'max_F': float(fids.max())})
        test_mean = float(np.mean([r['test_F'] for r in rows]))
        log_mean = float(np.mean([r['logged_val_F'] for r in rows]))
        passed = abs(test_mean - log_mean) <= TOL[n]
        ok &= passed
        out[f'{n}q'] = {'per_seed': rows, 'test_mean': test_mean, 'logged_mean': log_mean,
                        'difference': test_mean - log_mean, 'tolerance': TOL[n], 'passed': passed}
        print(f'{n}q: test-set mean F {test_mean:.4f} vs logged {log_mean:.4f} '
              f'(diff {test_mean - log_mean:+.4f}, tol {TOL[n]}) -> {"PASS" if passed else "FAIL"}')
        for r in rows:
            print(f'    seed {r["seed"]:5d}: test {r["test_F"]:.4f}  logged {r["logged_val_F"]:.4f}')
    os.makedirs(os.path.join(REV, 'results'), exist_ok=True)
    with open(os.path.join(REV, 'results', 'acceptance_check.json'), 'w') as f:
        json.dump({'all_passed': ok, **out}, f, indent=1)
    print('ALL PASSED' if ok else 'SOME CHECKS FAILED')


if __name__ == '__main__':
    main()
