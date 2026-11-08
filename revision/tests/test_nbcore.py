"""Sanity tests for revision/nbcore.py (CPU, about a minute).

Run from the repository root:  python revision/tests/test_nbcore.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
LEGACY = os.path.join(os.path.dirname(REV), 'legacy')
os.environ.setdefault('QST_OUTDIR', os.path.join(REV, 'results', 'scratch'))
os.environ['PYTHONPATH'] = REV + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, REV)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import nbcore as nb  # noqa: E402


def test_checkpoint_reproduces_known_fidelity():
    """Seed-42 3q model on the cached 3q validation set of legacy/tomography.py: F = 0.9255.

    The dataset caches are not in the repository; without this one the test is skipped.
    """
    path = os.path.join(LEGACY, '3q saturated', 'dataset_3q_cache.npz')
    if not os.path.exists(path):
        print("  skipped: needs legacy/3q saturated/dataset_3q_cache.npz, which is not in the repository")
        return
    d = np.load(path)
    model = nb.load_model(3, os.path.join(LEGACY, 'reviewer_experiments', 'run_all_experiments',
                                          'model_3q_seed42.pt'))
    mean_f, _, fids = nb.evaluate_fidelity(model, d['X_val'], d['Rho_val'])
    print(f'  3q seed 42 on cached validation set: F = {mean_f:.4f} (expected ~0.9255)')
    assert abs(mean_f - 0.9255) < 2e-3
    assert np.all(fids <= 1.0 + 1e-6)


def test_seeded_generation_is_worker_independent():
    a = nb.build_dataset_seeded(2, 70, seed=7, n_jobs=1, verbose=False)
    b = nb.build_dataset_seeded(2, 70, seed=7, n_jobs=4, verbose=False)
    c = nb.build_dataset_seeded(2, 70, seed=8, n_jobs=1, verbose=False)
    ha, hb, hc = nb.dataset_hash(*a), nb.dataset_hash(*b), nb.dataset_hash(*c)
    print(f'  hashes: 1 worker {ha}, 4 workers {hb}, other seed {hc}')
    assert ha == hb and ha != hc
    purity = np.einsum('nij,nji->n', a[1], a[1]).real
    assert purity.min() >= 0.84 and purity.max() <= 0.96
    assert np.bincount(a[2]).tolist() == [10] * 7


def test_history_wrapper_trains_identically():
    nb.ensure_basis(2)
    Xtr, Rtr, _ = nb.build_dataset_seeded(2, 140, seed=11, n_jobs=1, verbose=False)
    Xv, Rv, _ = nb.build_dataset_seeded(2, 70, seed=12, n_jobs=1, verbose=False)
    kw = dict(latent_dim=20, batch_size=64, max_epochs=6, n_geodesic_pairs=50)
    _, s1 = nb.train_one_seed(42, Xtr, Rtr, Xv, Rv, **kw)
    _, s2, hist, final = nb.train_with_history(42, Xtr, Rtr, Xv, Rv, **kw)
    same = all(torch.equal(s1[k], s2[k]) for k in s1)
    print(f'  identical best_state: {same}; history epochs {[h[0] for h in hist]}; '
          f'final F {final:.4f}')
    assert same
    assert [h[0] for h in hist] == [1, 5]


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            print(name)
            fn()
    print('all tests passed')
