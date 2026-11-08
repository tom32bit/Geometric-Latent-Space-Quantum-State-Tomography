"""Build the frozen held-out test sets used by every revision analysis.

For n = 2, 3, 4 qubits this writes, under revision/data/:
  testset_{n}q.npz  7 families x 1,000 states from the original generators
                    (purity window 0.85-0.95), with family labels.
  ood_{n}q.npz      Haar + white-noise states ("Depolarized" generator) in six
                    purity bins from 0.30 to 1.00, 500 per bin.

Both use nbcore.build_dataset_seeded with fixed seeds, so they are reproducible
and are not the data the models were trained on.

Usage:  python revision/make_testsets.py [--n 2 3 4] [--jobs 8]
"""
import argparse
import os
import sys
import time

REV = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('QST_OUTDIR', os.path.join(REV, 'results', 'scratch'))
os.environ['PYTHONPATH'] = REV + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, REV)

import numpy as np  # noqa: E402

import nbcore as nb  # noqa: E402

DATA = os.path.join(REV, 'data')
TEST_SEED = 303
OOD_SEED = 404
N_PER_FAMILY = 1000
OOD_BINS = [(0.30, 0.50), (0.50, 0.70), (0.70, 0.85), (0.85, 0.95), (0.95, 0.99), (0.99, 1.00)]
OOD_PER_BIN = 500


def purity(rhos):
    return np.einsum('nij,nji->n', rhos, rhos).real.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, nargs='+', default=[2, 3, 4])
    ap.add_argument('--jobs', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    for n in args.n:
        t0 = time.time()
        print(f'[{n}q] test set', flush=True)
        X, R, L = nb.build_dataset_seeded(n, 7 * N_PER_FAMILY, TEST_SEED, n_jobs=args.jobs)
        np.savez_compressed(os.path.join(DATA, f'testset_{n}q.npz'), X=X, Rho=R, labels=L,
                            purity=purity(R), seed=TEST_SEED, families=np.array(nb.FAMILIES),
                            hash=nb.dataset_hash(X, R, L))
        print(f'[{n}q] test set done in {time.time() - t0:.0f}s, hash {nb.dataset_hash(X, R, L)}',
              flush=True)
        t0 = time.time()
        Xo, Ro, Bo = [], [], []
        for b, (lo, hi) in enumerate(OOD_BINS):
            x, r, _ = nb.build_dataset_seeded(n, OOD_PER_BIN, OOD_SEED + 10 * b, min_purity=lo,
                                              max_purity=hi, n_jobs=args.jobs, families=[0],
                                              verbose=False)
            Xo.append(x); Ro.append(r); Bo.append(np.full(len(x), b))
        Xo, Ro, Bo = np.concatenate(Xo), np.concatenate(Ro), np.concatenate(Bo)
        np.savez_compressed(os.path.join(DATA, f'ood_{n}q.npz'), X=Xo, Rho=Ro, bin=Bo,
                            purity=purity(Ro), bins=np.array(OOD_BINS), seed=OOD_SEED,
                            hash=nb.dataset_hash(Xo, Ro, Bo))
        print(f'[{n}q] OOD set done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
