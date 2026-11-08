"""Build revision/nbcore.py from the original Kaggle notebooks.

nbcore.py holds the exact model, data and training code that produced the
submitted seed models: cells 1-4 and 8 of
legacy/reviewer_experiments/run_all_experiments.ipynb, which are byte-identical to the
same cells of run_4q_experiments.ipynb (checked below). The only edit to those
cells is that OUTDIR can be overridden with the QST_OUTDIR environment variable.

A clearly separated block of additions follows the copied cells. The additions
add seeded data generation, model loading and history logging; none of them
changes how a model is built or trained.

Usage:  python revision/build_nbcore.py
"""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_ALL = os.path.join(ROOT, 'legacy', 'reviewer_experiments', 'run_all_experiments.ipynb')
NB_4Q = os.path.join(ROOT, 'legacy', 'reviewer_experiments', 'run_4q_experiments.ipynb')
OUT = os.path.join(ROOT, 'revision', 'nbcore.py')
CELLS = [1, 2, 3, 4, 8]
OUTDIR_OLD = 'OUTDIR = "/kaggle/working"'
OUTDIR_NEW = 'OUTDIR = os.environ.get("QST_OUTDIR", "/kaggle/working")'

HEADER = '''"""Model, data and training code of the original experiments (generated file).

Do not edit by hand: run revision/build_nbcore.py instead.

Sections 1-5 are copied verbatim from cells 1, 2, 3, 4 and 8 of
reviewer_experiments/run_all_experiments.ipynb (identical in
run_4q_experiments.ipynb). The only edit is that OUTDIR can be overridden with
the QST_OUTDIR environment variable. The final section adds seeded data
generation, model loading and history logging without changing how a model is
built or trained.

Cell md5 (first 10 hex digits): {md5s}
"""
'''

ADDITIONS = r'''
# =============================================================================
# ADDITIONS FOR THE MLST REVISION
# Everything above is copied verbatim from the original notebooks. Nothing below
# changes how a model is built or trained.
# =============================================================================
import contextlib as _contextlib
import hashlib as _hashlib
import io as _io

FAMILIES = ['Depolarized', 'Werner', 'Isotropic', 'AmpDamped',
            'PhaseDamped', 'Thermal', 'Separable']
WERNER = FAMILIES.index('Werner')


def ensure_basis(n_qubits):
    """Activate the Pauli basis for n_qubits, quietly, unless it is active."""
    if globals().get('N_QUBITS') != n_qubits:
        with _contextlib.redirect_stdout(_io.StringIO()):
            setup_pauli_basis(n_qubits)


def _task_seed(seed, n_qubits, family, k):
    """Seed of generation task k of one family; fixed by the arguments alone."""
    ss = np.random.SeedSequence(entropy=[seed, n_qubits, family], spawn_key=(k,))
    return int(ss.generate_state(1)[0])


def _gen_task(n_qubits, family, task_seed, min_p, max_p):
    """One call of the original _gen_one() for one family, from a fixed seed."""
    ensure_basis(n_qubits)
    np.random.seed(task_seed)
    fn = _make_generators(min_p, max_p)[family][1]
    return _gen_one(fn, min_p, max_p)


def build_dataset_seeded(n_qubits, n_samples, seed, min_purity=0.85,
                         max_purity=0.95, n_jobs=None, families=None,
                         verbose=True):
    """Reproducible version of build_dataset().

    Same generators, per-family counts, acceptance rule (_gen_one) and family
    order as build_dataset(). The only change is seeding: generation task k of
    a family draws from its own seed, so the result does not depend on the
    number of workers or on how tasks are chunked. Tasks are taken in order and
    the first n_this valid results are kept, as in the original.

    Returns X (float32), Rhos (complex64) and the family index of each state.
    """
    ensure_basis(n_qubits)
    n_jobs = N_JOBS_DATA if n_jobs is None else n_jobs
    fams = list(range(len(FAMILIES))) if families is None else list(families)
    n_per, rem = divmod(n_samples, len(fams))
    X_list, R_list, L_list = [], [], []
    for j, fam in enumerate(fams):
        n_this = n_per + (1 if j < rem else 0)
        valid, k = [], 0
        while len(valid) < n_this:
            chunk = max(32, int(1.3 * (n_this - len(valid))) + 16)
            seeds = [_task_seed(seed, n_qubits, fam, k + t) for t in range(chunk)]
            if HAS_JOBLIB and n_jobs > 1:
                res = Parallel(n_jobs=n_jobs)(
                    delayed(_gen_task)(n_qubits, fam, s, min_purity, max_purity)
                    for s in seeds)
            else:
                res = [_gen_task(n_qubits, fam, s, min_purity, max_purity)
                       for s in seeds]
            valid.extend(r for r in res if r is not None)
            k += chunk
            if k > 40 * n_this + 2000:
                raise RuntimeError(f'{FAMILIES[fam]}: only {len(valid)} valid '
                                   f'states after {k} tasks')
        for exps, rho in valid[:n_this]:
            X_list.append(exps); R_list.append(rho); L_list.append(fam)
        if verbose:
            print(f'  {FAMILIES[fam]}: {n_this} states ({k} tasks)', flush=True)
    ensure_basis(n_qubits)
    X = np.stack(X_list); Rhos = np.stack(R_list)
    return X, Rhos, np.array(L_list, dtype=np.int64)


def dataset_hash(*arrays):
    """Short SHA-256 of the arrays' bytes, to check that two runs saw the same data."""
    h = _hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def load_model(n_qubits, path, latent_dim=None):
    """Load a saved best_state checkpoint into the original HybridAE (eval mode)."""
    ensure_basis(n_qubits)
    latent_dim = 10 * n_qubits if latent_dim is None else latent_dim
    model = HybridAE(n_pauli_nonid, latent_dim, n_angles, n_noise_params,
                     full_paulis_arr, pauli_idx_nonid)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.to(device_torch).eval()


def train_with_history(*args, **kwargs):
    """Run the original train_one_seed() unchanged and record its validation curve.

    train_one_seed() calls evaluate_fidelity() at epoch 1, every eval_every
    epochs, and once more after restoring the best state. A logging wrapper is
    swapped in for the duration of the call; it returns exactly what the
    original returns and draws no random numbers, so training is unaffected.

    Returns (results, best_state, history, final_eval). history holds one
    (epoch, validation mean F, seconds) entry per in-training evaluation, and
    final_eval is the validation mean F of the restored best state.
    """
    global evaluate_fidelity
    original = evaluate_fidelity
    every = kwargs.get('eval_every', 5)
    calls, t0 = [], time.time()

    def _logged(model, X_np, Rho_np, bs=256):
        out = original(model, X_np, Rho_np, bs)
        calls.append((out[0], time.time() - t0))
        return out

    evaluate_fidelity = _logged
    try:
        results, best_state = train_one_seed(*args, **kwargs)
    finally:
        evaluate_fidelity = original
    final_eval = calls.pop()[0]
    history = [(1 if i == 0 else every * i, f, s) for i, (f, s) in enumerate(calls)]
    return results, best_state, history, final_eval


def pareto_metrics_e3(model, X_tr, Rho_tr, X_v, Rho_v, seed=42):
    """Table 6 metrics exactly as the original Pareto cell computed them.

    Validation mean F, purity MAE, and Pearson r over 500 pairs drawn from
    training + validation data with the full latent z (evaluate_geometry_e3).
    """
    model.eval()
    val_mean, val_median, _ = evaluate_fidelity(model, X_v, Rho_v)
    with torch.no_grad():
        xb = torch.tensor(X_v, dtype=torch.float32, device=device_torch)
        rt = torch.tensor(Rho_v, dtype=torch.complex64, device=device_torch)
        preds_b, *_ = model(xb)
        rho_pred = pauli_exps_to_rho_torch(preds_b)
        pur_pred = torch.einsum('bij,bji->b', rho_pred, rho_pred).real.cpu().numpy()
        pur_true = torch.einsum('bij,bji->b', rt, rt).real.cpu().numpy()
        purity_mae = float(np.mean(np.abs(pur_pred - pur_true)))
        _enc = model.encoder
        X_all = torch.cat([torch.tensor(X_tr, dtype=torch.float32, device=device_torch),
                           torch.tensor(X_v, dtype=torch.float32, device=device_torch)], dim=0)
        Rho_all_np = np.concatenate([Rho_tr, Rho_v], axis=0)
        Z = np.vstack([_enc(X_all[i:i+256])[0].cpu().numpy()
                       for i in range(0, len(X_all), 256)])
    np.random.seed(seed)
    n_pairs = min(500, len(Z) * (len(Z) - 1) // 2)
    pair_idx = np.array([np.random.choice(len(Z), 2, replace=False) for _ in range(n_pairs)])
    ii, jj = pair_idx[:, 0], pair_idx[:, 1]
    geo_dists = np.linalg.norm(Z[ii] - Z[jj], axis=1)
    Ri = torch.tensor(Rho_all_np[ii], dtype=torch.complex64, device=device_torch)
    Rj = torch.tensor(Rho_all_np[jj], dtype=torch.complex64, device=device_torch)
    F_vals = np.concatenate([uhlmann_fidelity_batch(Ri[k:k+50], Rj[k:k+50]).cpu().numpy()
                             for k in range(0, len(ii), 50)])
    bures = np.arccos(np.sqrt(np.clip(F_vals, 0.0, 1.0)))
    valid = bures > 1e-6
    pr, _ = pearsonr(geo_dists[valid], bures[valid])
    return {'mean_fidelity': val_mean, 'median_fidelity': val_median,
            'pearson_r': float(pr), 'purity_mae': purity_mae}
'''


def main():
    cells = {}
    for tag, path in (('all', NB_ALL), ('4q', NB_4Q)):
        nb = json.load(open(path, encoding='utf-8'))
        cells[tag] = {i: ''.join(nb['cells'][i]['source']) for i in CELLS}
        for i in CELLS:
            assert nb['cells'][i]['cell_type'] == 'code', (path, i)
    for i in CELLS:
        assert cells['all'][i] == cells['4q'][i], f'cell {i} differs between notebooks'
    src1 = cells['all'][1]
    assert src1.count(OUTDIR_OLD) == 1, 'OUTDIR line not found exactly once'
    cells['all'][1] = src1.replace(OUTDIR_OLD, OUTDIR_NEW)
    md5s = ', '.join(f'cell{i}={hashlib.md5(cells["4q"][i].encode()).hexdigest()[:10]}'
                     for i in CELLS)
    parts = [HEADER.format(md5s=md5s)]
    for n, i in enumerate(CELLS, 1):
        parts.append(f'\n# {"=" * 77}\n# SECTION {n}: notebook cell {i} (verbatim)\n# {"=" * 77}\n')
        parts.append(cells['all'][i].rstrip() + '\n')
    parts.append(ADDITIONS)
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        f.write(''.join(parts))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
