"""Build revision/kaggle_lambda_m_ablation.ipynb: the λ_m = 0 ablation and the loss-weight grid.

The notebook writes the current revision/nbcore.py into the Kaggle session and
trains with it, so Kaggle and the local evaluation use identical code.

Usage:  python revision/build_kaggle_notebook.py
"""
import hashlib
import json
import os

REV = os.path.dirname(os.path.abspath(__file__))
NBCORE = os.path.join(REV, 'nbcore.py')
OUT = os.path.join(REV, 'kaggle_lambda_m_ablation.ipynb')

INTRO = """# λ_m = 0 ablation of the metric loss

This notebook trains the **identical architecture and settings** as the original seed study, with only the metric-loss weight λ_m changed. The training code is the original notebook code (`nbcore.py`, written by the next cell), unchanged.

**Why both arms are trained.** The original seed models cannot be compared directly with new runs, because their training data cannot be regenerated: the original data generator was unseeded. Here both arms, λ_m = 0 and λ_m = 0.2, train on the same new, seeded dataset. With the same seed, the two arms also start from identical weights and see identical batches. The only difference between them is λ_m.

## How to run
1. Settings: **Accelerator = GPU T4 x2** (one GPU is used); Internet is not needed.
2. In the next-but-one cell, set `RUN_GROUP`. Run `'smoke'` first (about 5 minutes) to check that everything works.
3. Use **Save Version → Save & Run All (Commit)**, one group per commit. Each group fits within Kaggle's 12-hour limit.
4. When a version finishes, download `lambda_m_ablation_<group>.zip` from its **Output** tab and put it in `revision/kaggle_outputs/` in the repository.

| RUN_GROUP | Runs | Approx. GPU time |
|---|---|---|
| `smoke` | 2q, tiny data, 3 epochs | 5 min |
| `4q_lm0` | 4q, λ_m = 0, seeds 42 / 123 / 256 | 9 h |
| `4q_lm02` | 4q, λ_m = 0.2, seeds 42 / 123 / 256 | 9 h |
| `3q` | 3q, λ_m = 0 and 0.2, 5 seeds each | 7 h |
| `2q_pareto` | 2q, λ_m = 0 and 0.2, 5 seeds each; then the Table 6 grid at 3q (λ_m ∈ {0, 0.1, 0.2, 0.5} × λ_p ∈ {1, 2, 5}) | 6 h |

A finished run is never repeated within a session. Each run saves its checkpoint and a JSON file as soon as it ends. The JSON records the full validation curve and a hash of the training data, so the two arms can be checked for having seen identical data.
"""

CONFIG = """# Choose one group per commit: 'smoke', '4q_lm0', '4q_lm02', '3q', '2q_pareto'
RUN_GROUP = 'smoke'"""

RUN = r'''import hashlib, json, os, platform, shutil, sys, time
os.environ['QST_OUTDIR'] = '/kaggle/working'
os.environ['PYTHONPATH'] = os.getcwd() + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import nbcore as nb

NBCORE_SHA = hashlib.sha256(open('nbcore.py', 'rb').read().rstrip(b'\n')).hexdigest()[:16]
assert NBCORE_SHA == '__NBCORE_SHA__', 'nbcore.py differs from the version this notebook was built with'
OUT = '/kaggle/working/lambda_m_ablation'
os.makedirs(OUT, exist_ok=True)

TRAIN_SEED, VAL_SEED = 101, 202                 # main-configuration data (both arms)
PARETO_TRAIN_SEED, PARETO_VAL_SEED = 505, 606   # Table 6 grid data
SEEDS = {2: [42, 123, 256, 789, 1024], 3: [42, 123, 256, 789, 1024], 4: [42, 123, 256]}
GROUPS = {
    'smoke':     [('main', 2, [0.0], [42])],
    '4q_lm0':    [('main', 4, [0.0], SEEDS[4])],
    '4q_lm02':   [('main', 4, [0.2], SEEDS[4])],
    '3q':        [('main', 3, [0.0, 0.2], SEEDS[3])],
    '2q_pareto': [('main', 2, [0.0, 0.2], SEEDS[2]), ('pareto', 3, None, [42])],
}
assert RUN_GROUP in GROUPS, RUN_GROUP
SMOKE = RUN_GROUP == 'smoke'
if not SMOKE:
    assert torch.cuda.is_available(), 'Turn on the GPU accelerator before running this group.'
ENV = {'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version(),
       'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
       'nbcore_sha': NBCORE_SHA, 'group': RUN_GROUP}
print(ENV)


def make_data(n, nt, nv, seed_tr, seed_v):
    t0 = time.time()
    Xtr, Rtr, Ltr = nb.build_dataset_seeded(n, nt, seed_tr)
    Xv, Rv, Lv = nb.build_dataset_seeded(n, nv, seed_v)
    info = {'n_train': nt, 'n_val': nv, 'train_seed': seed_tr, 'val_seed': seed_v,
            'train_hash': nb.dataset_hash(Xtr, Rtr, Ltr), 'val_hash': nb.dataset_hash(Xv, Rv, Lv),
            'generation_s': round(time.time() - t0, 1)}
    print('data', info, flush=True)
    return Xtr, Rtr, Xv, Rv, info


def save_run(tag, record, state):
    torch.save(state, os.path.join(OUT, f'model_{tag}.pt'))
    with open(os.path.join(OUT, f'run_{tag}.json'), 'w') as f:
        json.dump(record, f, indent=1)


def run_main(n, lambdas, seeds):
    nt = max(5000, 300 * (4 ** n - 1)); nv = max(800, nt // 6)
    bs = 512 if nt >= 50000 else 256 if nt >= 10000 else 64
    epochs = 600
    if SMOKE:
        nt, nv, epochs = 350, 140, 3
    Xtr, Rtr, Xv, Rv, info = make_data(n, nt, nv, TRAIN_SEED, VAL_SEED)
    for lm in lambdas:
        for s in seeds:
            tag = f'{n}q_lm{lm:g}_seed{s}'
            if os.path.exists(os.path.join(OUT, f'run_{tag}.json')):
                print('skip', tag); continue
            t0 = time.time()
            res, state, hist, final = nb.train_with_history(
                s, Xtr, Rtr, Xv, Rv, latent_dim=10 * n, batch_size=bs, max_epochs=epochs,
                lambda_metric=lm, lambda_purity=2.0, lambda_purity_enc=0.10)
            record = {'kind': 'main', 'n_qubits': n, 'lambda_m': lm, 'lambda_p': 2.0,
                      'lambda_e': 0.10, 'seed': s, 'batch_size': bs, 'max_epochs': epochs,
                      'data': info, 'env': ENV, 'wall_s': round(time.time() - t0, 1),
                      'final_val_fidelity': final, 'history': hist, 'train_one_seed_results': res,
                      'note': 'shot_* entries in train_one_seed_results use the original '
                              '(flawed) shot-noise evaluation and are not used.'}
            save_run(tag, record, state)
            print(f'done {tag}: val F {final:.4f}, epochs {res["final_epoch"]}, '
                  f'{record["wall_s"] / 60:.1f} min', flush=True)


def run_pareto(n=3):
    nt, nv, bs, epochs = 5000, 800, 64, 200
    Xtr, Rtr, Xv, Rv, info = make_data(n, nt, nv, PARETO_TRAIN_SEED, PARETO_VAL_SEED)
    for lm in [0.0, 0.10, 0.20, 0.50]:
        for lp in [1.0, 2.0, 5.0]:
            tag = f'pareto_{n}q_lm{lm:g}_lp{lp:g}'
            if os.path.exists(os.path.join(OUT, f'run_{tag}.json')):
                print('skip', tag); continue
            t0 = time.time()
            res, state, hist, final = nb.train_with_history(
                42, Xtr, Rtr, Xv, Rv, latent_dim=30, batch_size=bs, max_epochs=epochs,
                lambda_metric=lm, lambda_purity=lp, lambda_purity_enc=0.10)
            model = nb.HybridAE(nb.n_pauli_nonid, 30, nb.n_angles, nb.n_noise_params,
                                nb.full_paulis_arr, nb.pauli_idx_nonid).to(nb.device_torch)
            model.load_state_dict(state)
            table6 = nb.pareto_metrics_e3(model, Xtr, Rtr, Xv, Rv, seed=42)
            record = {'kind': 'pareto', 'n_qubits': n, 'lambda_m': lm, 'lambda_p': lp,
                      'lambda_e': 0.10, 'seed': 42, 'batch_size': bs, 'max_epochs': epochs,
                      'data': info, 'env': ENV, 'wall_s': round(time.time() - t0, 1),
                      'final_val_fidelity': final, 'history': hist, 'table6_metrics': table6,
                      'train_one_seed_results': res}
            save_run(tag, record, state)
            print(f'done {tag}: F {table6["mean_fidelity"]:.4f}, r {table6["pearson_r"]:.4f}, '
                  f'{record["wall_s"] / 60:.1f} min', flush=True)


t_all = time.time()
for job in GROUPS[RUN_GROUP]:
    if job[0] == 'main':
        run_main(job[1], job[2], job[3])
    else:
        run_pareto(job[1])
print(f'group {RUN_GROUP} finished in {(time.time() - t_all) / 3600:.2f} h')
shutil.make_archive(f'/kaggle/working/lambda_m_ablation_{RUN_GROUP}', 'zip', OUT)
print('saved', f'/kaggle/working/lambda_m_ablation_{RUN_GROUP}.zip')'''


def cell(kind, src):
    lines = src.split('\n')
    source = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    c = {'cell_type': kind, 'metadata': {}, 'source': source}
    if kind == 'code':
        c.update({'execution_count': None, 'outputs': []})
    return c


def main():
    nbcore = open(NBCORE, encoding='utf-8').read()
    sha = hashlib.sha256(nbcore.rstrip('\n').encode('utf-8')).hexdigest()[:16]
    cells = [cell('markdown', INTRO),
             cell('code', '%%writefile nbcore.py\n' + nbcore.rstrip('\n')),
             cell('code', CONFIG),
             cell('code', RUN.replace('__NBCORE_SHA__', sha))]
    notebook = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f'wrote {OUT} (nbcore sha {sha})')


if __name__ == '__main__':
    main()
