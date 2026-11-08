"""Process the Kaggle outputs: the λ_m ablation and the loss-weight grid.

1. Unpacks every lambda_m_ablation_<group>.zip in revision/kaggle_outputs/.
2. Checks that, for each qubit count, all runs (both λ_m arms) saw identical
   training and validation data (same data hashes) and the same nbcore.py.
3. Evaluates the ablation models on the frozen test sets (fidelity and geometry,
   via run_analyses.py) and computes paired differences λ_m = 0.2 minus λ_m = 0
   per seed, with t-based 95 % CIs (results/ablation_summary.json).
4. Collects the Table 6 grid (results/pareto_grid.json).

Usage:  python revision/process_kaggle.py
"""
import glob
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
from scipy.stats import t as tdist

REV = os.path.dirname(os.path.abspath(__file__))
KO = os.path.join(REV, 'kaggle_outputs')
RES = os.path.join(REV, 'results')


def unpack():
    for z in glob.glob(os.path.join(KO, 'lambda_m_ablation_*.zip')):
        dest = os.path.join(KO, os.path.splitext(os.path.basename(z))[0])
        if not os.path.isdir(dest):
            with zipfile.ZipFile(z) as f:
                f.extractall(dest)
            print('unpacked', os.path.basename(z))


def load_runs():
    runs = []
    for p in glob.glob(os.path.join(KO, '**', 'run_*.json'), recursive=True):
        r = json.load(open(p))
        if r['env'].get('group') == 'smoke':           # 3-epoch check run, not part of the study
            continue
        r['_path'] = p
        runs.append(r)
    return runs


def check_pairing(runs):
    ok = True
    by_key = {}
    for r in runs:
        key = (r['kind'], r['n_qubits'])
        by_key.setdefault(key, set()).add((r['data']['train_hash'], r['data']['val_hash'], r['env']['nbcore_sha']))
    for key, hashes in sorted(by_key.items()):
        status = 'OK' if len(hashes) == 1 else 'MISMATCH'
        ok &= len(hashes) == 1
        print(f'{key[0]} {key[1]}q: {len(hashes)} distinct (train, val, nbcore) hash set(s) -> {status}')
    return ok


def paired_summary():
    fid = json.load(open(os.path.join(RES, 'fidelity_ablation.json')))
    geo = json.load(open(os.path.join(RES, 'geometry_ablation.json')))
    out = {}
    for n in ('2q', '3q', '4q'):
        if n not in fid or 'lambda_m=0' not in fid[n]['per_model'] or 'lambda_m=0.2' not in fid[n]['per_model']:
            continue
        f0 = {r['seed']: r for r in fid[n]['per_model']['lambda_m=0']}
        f2 = {r['seed']: r for r in fid[n]['per_model']['lambda_m=0.2']}
        g0 = {r['seed']: r for r in geo[n]['per_model']['lambda_m=0']}
        g2 = {r['seed']: r for r in geo[n]['per_model']['lambda_m=0.2']}
        seeds = sorted(set(f0) & set(f2) & set(g0) & set(g2))
        metrics = {
            'F': lambda F, G: F['mean'], 'F_no_werner': lambda F, G: F['mean_no_werner'],
            'pearson_all': lambda F, G: G['all']['pearson'], 'spearman_all': lambda F, G: G['all']['spearman'],
            'pearson_no_werner': lambda F, G: G['no_werner']['pearson'],
            'spearman_no_werner': lambda F, G: G['no_werner']['spearman'],
            'pearson_within': lambda F, G: G['within_family_no_werner']['pearson'],
            'pearson_between': lambda F, G: G['between_family_no_werner']['pearson']}
        res = {'seeds': seeds}
        for name, get in metrics.items():
            a0 = np.array([get(f0[s], g0[s]) for s in seeds])
            a2 = np.array([get(f2[s], g2[s]) for s in seeds])
            d = a2 - a0
            k = len(d)
            half = float(tdist.ppf(0.975, k - 1) * d.std(ddof=1) / np.sqrt(k)) if k > 1 else float('nan')
            res[name] = {'lambda_m_0': [float(a0.mean()), float(a0.std(ddof=1)) if k > 1 else 0.0],
                         'lambda_m_0.2': [float(a2.mean()), float(a2.std(ddof=1)) if k > 1 else 0.0],
                         'paired_difference': float(d.mean()), 'paired_ci95': [float(d.mean()) - half, float(d.mean()) + half],
                         'per_seed_difference': d.tolist()}
        out[n] = res
        print(f'{n}: ' + '; '.join(f'{m} {res[m]["lambda_m_0"][0]:.3f} -> {res[m]["lambda_m_0.2"][0]:.3f} '
                                    f'(diff {res[m]["paired_difference"]:+.3f} [{res[m]["paired_ci95"][0]:+.3f}, {res[m]["paired_ci95"][1]:+.3f}])'
                                    for m in ('F', 'pearson_all', 'pearson_no_werner', 'spearman_no_werner')))
    with open(os.path.join(RES, 'ablation_summary.json'), 'w') as f:
        json.dump(out, f, indent=1)


def pareto_grid(runs):
    rows = [{'lambda_m': r['lambda_m'], 'lambda_p': r['lambda_p'], **r['table6_metrics'],
             'final_epoch': r['train_one_seed_results']['final_epoch']}
            for r in runs if r['kind'] == 'pareto']
    rows.sort(key=lambda r: (r['lambda_m'], r['lambda_p']))
    with open(os.path.join(RES, 'pareto_grid.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        print(f'  lambda_m {r["lambda_m"]:<4} lambda_p {r["lambda_p"]:<4} F {r["mean_fidelity"]:.3f} '
              f'r {r["pearson_r"]:.3f} purity MAE {r["purity_mae"]:.4f} epochs {r["final_epoch"]}')
    return rows


def main():
    unpack()
    runs = load_runs()
    print(f'{len(runs)} run files found')
    if not runs:
        return
    if not check_pairing(runs):
        print('WARNING: the two arms did not see identical data; do not report paired differences.')
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    if any(r['kind'] == 'main' for r in runs):
        for cmd in ('fidelity', 'geometry'):
            subprocess.run([sys.executable, os.path.join(REV, 'run_analyses.py'), cmd, '--models', 'ablation'],
                           check=True, env=env)
        paired_summary()
    if any(r['kind'] == 'pareto' for r in runs):
        pareto_grid(runs)


if __name__ == '__main__':
    main()
