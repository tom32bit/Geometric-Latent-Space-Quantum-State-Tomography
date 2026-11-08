"""Analyses of the revised manuscript. Each command writes revision/results/<command>[_<models>].json.

  fidelity   fidelity per seed and per generator, with and without Werner states, CIs
  geometry   latent-Bures correlations on held-out pairs with state-level bootstrap CIs,
             cross-n differences, local anisotropy ratio κ, PCA dimension
  shots      shot noise on the encoder input (binomial sampling): fidelity, bias/variance
             split, purity and Bures-angle diagnostics, and the submitted evaluation
             applied to a perfect predictor
  oracle     per-state circuit fits and the parameter-ratio table
  warmstart  per-state refinement against the measured Pauli vector, started from the
             encoder's prediction or from random parameters
  latentviz  PCA / t-SNE coordinates of the latent space
  timing     latent distance vs exact fidelity, per pair
  ood        fidelity outside the training purity window

Usage:  python revision/run_analyses.py <command|all> [--models original|ablation]
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

import qst_eval as qe
from qst_eval import nb

CACHE = os.path.join(qe.RESULTS, 'cache')
QUBITS = (2, 3, 4)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def save(name, obj):
    os.makedirs(qe.RESULTS, exist_ok=True)
    meta = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'testset_hashes': {f'{n}q': str(np.load(os.path.join(qe.REV, 'data', f'testset_{n}q.npz'))['hash'])
                               for n in QUBITS}}
    path = os.path.join(qe.RESULTS, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(jsonable({'meta': meta, **obj}), f, indent=1)
    print(f'wrote {path}', flush=True)


def cached_inference(which, n, label, seed, path, X):
    """Latents and test-set fidelities of one model, cached under results/cache/."""
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, f'{which}_{n}q_{label.replace("=", "")}_seed{seed}.npz')
    if os.path.exists(fn):
        d = np.load(fn)
        return {k: d[k] for k in d.files}
    model = nb.load_model(n, path)
    inf = qe.infer(model, X)
    ts = qe.load_testset(n)
    F = qe.fidelity64(ts['Rho'], inf['rho'])
    out = {'z': inf['z'], 'zg': inf['zg'], 'zp': inf['zp'], 'F': F,
           'purity_pred': qe.purity(inf['rho']),
           'max_trace_dev': inf['checks']['max_trace_dev'],
           'min_eigenvalue': inf['checks']['min_eigenvalue']}
    np.savez_compressed(fn, **out)
    return out


# --------------------------------------------------------------------- fidelity
def run_fidelity(which, B=2000):
    out = {}
    rng = np.random.default_rng(11)
    for n in QUBITS:
        ts = qe.load_testset(n)
        lab, labm = ts['labels'], qe.merged_labels(ts['labels'])
        rows, Fs = {}, {}
        for label, seed, path in qe.model_list(which, n):
            c = cached_inference(which, n, label, seed, path, ts['X'])
            F = c['F']
            rows.setdefault(label, []).append({
                'seed': seed, 'mean': F.mean(), 'median': np.median(F),
                'mean_no_werner': F[lab != qe.WERNER].mean(),
                'per_family': {fam: F[lab == k].mean() for k, fam in enumerate(qe.FAMILIES)},
                'haar_plus_noise_merged': F[labm == 0].mean(),
                'max_F': F.max(), 'max_trace_dev': float(c['max_trace_dev']),
                'min_eigenvalue': float(c['min_eigenvalue'])})
            Fs.setdefault(label, []).append(F)
        summary = {}
        for label, rs in rows.items():
            Fbar = np.mean(Fs[label], axis=0)            # seed-mean fidelity of each state
            nw = lab != qe.WERNER
            idx = rng.integers(0, len(Fbar), size=(B, len(Fbar)))
            boot_all = Fbar[idx].mean(axis=1)
            idx_nw = rng.integers(0, nw.sum(), size=(B, nw.sum()))
            boot_nw = Fbar[nw][idx_nw].mean(axis=1)
            summary[label] = {
                'mean': qe.summarize([r['mean'] for r in rs]),
                'median': qe.summarize([r['median'] for r in rs]),
                'mean_no_werner': qe.summarize([r['mean_no_werner'] for r in rs]),
                'haar_plus_noise_merged': qe.summarize([r['haar_plus_noise_merged'] for r in rs]),
                'per_family': {fam: qe.summarize([r['per_family'][fam] for r in rs])
                               for fam in qe.FAMILIES},
                'state_bootstrap_ci95_of_seed_mean': {'all': qe.percentile_ci(boot_all),
                                                      'no_werner': qe.percentile_ci(boot_nw)}}
            s = summary[label]
            print(f'{n}q {label}: F {s["mean"]["mean"]:.4f} ± {s["mean"]["sd"]:.4f} '
                  f'(no Werner {s["mean_no_werner"]["mean"]:.4f})', flush=True)
        out[f'{n}q'] = {'per_model': rows, 'summary': summary}
    save(f'fidelity_{which}', out)


# --------------------------------------------------------------------- geometry
def run_geometry(which, B=1000, n_pairs=20000):
    out, reps_all = {}, {}
    for n in QUBITS:
        t0 = time.time()
        ts = qe.load_testset(n)
        N = len(ts['X'])
        lab, labm = ts['labels'], qe.merged_labels(ts['labels'])
        rng = np.random.default_rng(2026 + n)
        i = rng.integers(0, N, n_pairs); j = rng.integers(0, N, n_pairs)
        keep = i != j; i, j = i[keep], j[keep]
        db = qe.bures_angle(qe.fidelity64(ts['Rho'][i], ts['Rho'][j]))
        valid = db > 1e-6
        wern = (lab[i] == qe.WERNER) | (lab[j] == qe.WERNER)
        same = labm[i] == labm[j]
        subsets = {'all': valid, 'no_werner': valid & ~wern,
                   'within_family_no_werner': valid & ~wern & same,
                   'between_family_no_werner': valid & ~wern & ~same,
                   'werner_werner': valid & (lab[i] == qe.WERNER) & (lab[j] == qe.WERNER)}
        counts = rng.multinomial(N, np.full(N, 1.0 / N), size=B).astype(np.float64)
        rows, reps = {}, {}
        for label, seed, path in qe.model_list(which, n):
            c = cached_inference(which, n, label, seed, path, ts['X'])
            zg, z = c['zg'].astype(np.float64), c['z'].astype(np.float64)
            dl = np.linalg.norm(zg[i] - zg[j], axis=1)
            res = {'seed': seed}
            rep = reps.setdefault(label, {})
            for sname, m in subsets.items():
                x, y = db[m], dl[m]
                entry = {'n_pairs': int(m.sum()), 'fraction_of_pairs': float(m.mean())}
                if m.sum() >= 10 and sname != 'werner_werner':
                    entry.update({'pearson': pearsonr(y, x)[0], 'spearman': spearmanr(y, x)[0],
                                  'r2': qe.r2_linfit(x, y)})
                    ox, oy = np.argsort(x, kind='stable'), np.argsort(y, kind='stable')
                    w_all = counts[:, i[m]] * counts[:, j[m]]
                    bp = np.array([qe.wpearson(y, x, w) for w in w_all])
                    bs = np.array([qe.wpearson(qe.wranks(oy, w), qe.wranks(ox, w), w) for w in w_all])
                    entry.update({'pearson_ci95': qe.percentile_ci(bp), 'spearman_ci95': qe.percentile_ci(bs)})
                    rep.setdefault(f'{sname}.pearson', []).append(bp)
                    rep.setdefault(f'{sname}.spearman', []).append(bs)
                if sname == 'werner_werner':
                    entry.update({'mean_bures_angle': float(x.mean()) if len(x) else None,
                                  'mean_latent_distance': float(y.mean()) if len(y) else None,
                                  'mean_latent_distance_all_pairs': float(dl[valid].mean())})
                res[sname] = entry
            kap = qe.kappa_per_state(zg)
            ok = ~np.isnan(kap)
            kb = (counts[:, ok] @ kap[ok]) / counts[:, ok].sum(axis=1)
            res['kappa'] = {'mean': float(kap[ok].mean()), 'ci95': qe.percentile_ci(kb),
                            'k_neighbours': 25, 'latent': 'z_geom',
                            'n_singular_values': int(min(25, zg.shape[1]))}
            rep.setdefault('kappa', []).append(kb)
            res['pca_dim95_full_z'] = qe.pca_dim95(z)
            res['pearson_minus_spearman_all'] = res['all']['pearson'] - res['all']['spearman']
            rows.setdefault(label, []).append(res)
        summary = {}
        for label, rs in rows.items():
            s = {}
            for sname in ('all', 'no_werner', 'within_family_no_werner', 'between_family_no_werner'):
                s[sname] = {met: qe.summarize([r[sname][met] for r in rs]) for met in ('pearson', 'spearman', 'r2')}
                for met in ('pearson', 'spearman'):
                    seed_mean_reps = np.mean(reps[label][f'{sname}.{met}'], axis=0)
                    s[sname][met]['state_bootstrap_ci95_of_seed_mean'] = qe.percentile_ci(seed_mean_reps)
                    reps_all.setdefault(label, {}).setdefault(n, {})[f'{sname}.{met}'] = seed_mean_reps
                s[sname]['n_pairs'] = rs[0][sname]['n_pairs']
            s['werner_werner_fraction'] = rs[0]['werner_werner']['fraction_of_pairs']
            s['kappa'] = qe.summarize([r['kappa']['mean'] for r in rs])
            s['kappa']['state_bootstrap_ci95_of_seed_mean'] = qe.percentile_ci(np.mean(reps[label]['kappa'], axis=0))
            reps_all.setdefault(label, {}).setdefault(n, {})['kappa'] = np.mean(reps[label]['kappa'], axis=0)
            s['pca_dim95_full_z'] = qe.summarize([r['pca_dim95_full_z'] for r in rs])
            summary[label] = s
            print(f'{n}q {label}: r all {s["all"]["pearson"]["mean"]:.3f}, no-Werner '
                  f'{s["no_werner"]["pearson"]["mean"]:.3f}; rho_s all {s["all"]["spearman"]["mean"]:.3f}; '
                  f'kappa {s["kappa"]["mean"]:.4f}  [{time.time() - t0:.0f}s]', flush=True)
        out[f'{n}q'] = {'pairs': {'requested': n_pairs, 'used': int(len(i)), 'valid': int(valid.sum()),
                                  'seed': 2026 + n},
                        'bootstrap': {'replicates': B, 'unit': 'states (multinomial counts; pair weight = c_i c_j)',
                                      'kappa_note': 'per-state κ fixed; neighbourhoods not recomputed'},
                        'per_model': rows, 'summary': summary}
    diffs = {}
    for label, by_n in reps_all.items():
        for a, b in ((2, 3), (3, 4), (2, 4)):
            if a in by_n and b in by_n:
                for key in by_n[a]:
                    d = by_n[a][key] - by_n[b][key]
                    diffs.setdefault(label, {})[f'{key}: {a}q-{b}q'] = {
                        'point': float(np.mean(by_n[a][key]) - np.mean(by_n[b][key])),
                        'ci95': qe.percentile_ci(d), 'excludes_zero': bool(np.all(np.sign(np.percentile(d, [2.5, 97.5])) == np.sign(np.mean(d))))}
    out['cross_n_differences'] = diffs
    save(f'geometry_{which}', out)


# ------------------------------------------------------------------- shot noise
def binomial_paulis(X, shots, rng):
    p = np.clip((1.0 + X.astype(np.float64)) / 2.0, 0.0, 1.0)
    return (2.0 * rng.binomial(shots, p) / shots - 1.0).astype(np.float32)


def spectrum_stats(rho, d):
    ev = np.clip(np.linalg.eigvalsh(rho), 0, None)
    pur = np.sum(ev ** 2, axis=-1)
    F_mm = np.sum(np.sqrt(ev), axis=-1) ** 2 / d        # fidelity with I/d
    return pur, qe.bures_angle(F_mm)


def describe(v):
    return {'mean': float(np.mean(v)), 'sd': float(np.std(v)), 'q05': float(np.percentile(v, 5)),
            'q50': float(np.percentile(v, 50)), 'q95': float(np.percentile(v, 95))}


def old_pipeline_perfect_predictor(n, X, Rho, n_states=200):
    """The submitted evaluation, fed the true Pauli vectors (a perfect predictor)."""
    nb.ensure_basis(n)
    rng = np.random.default_rng(99)
    S = rng.choice(len(X), n_states, replace=False)
    res = {}
    for shots in (10_000, 1_000, 100):
        np.random.seed(1234 + shots)
        F, min_eigs = [], []
        for k in S:
            pred = X[k].astype(np.float32)
            variance = (1.0 - pred ** 2) / shots
            noisy = np.clip(pred + np.random.randn(len(pred)) * np.sqrt(np.maximum(variance, 0.0)), -1.0, 1.0)
            rho_pred = nb.pauli_exps_to_rho(noisy)
            F.append(nb.fidelity_numpy(Rho[k], rho_pred))
            min_eigs.append(np.linalg.eigvalsh(rho_pred).min())
        F, min_eigs = np.array(F), np.array(min_eigs)
        res[str(shots)] = {'mean_F': F.mean(), 'max_F': F.max(), 'fraction_F_above_1': float((F > 1).mean()),
                           'fraction_non_physical': float((min_eigs < -1e-9).mean())}
    return res


def run_shots(which, R=10, sigmas=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3), R_sigma=3):
    out = {}
    for n in QUBITS:
        t0 = time.time()
        ts = qe.load_testset(n)
        X, Rho, d = ts['X'], ts['Rho'], 2 ** n
        rng = np.random.default_rng(5000 + n)
        pur_true, bang_true = spectrum_stats(Rho.astype(np.complex128), d)
        rows = {}
        for label, seed, path in qe.model_list(which, n):
            model = nb.load_model(n, path)
            base = qe.infer(model, X)
            F_inf = qe.fidelity64(Rho, base['rho'])
            pur0, bang0 = spectrum_stats(base['rho'], d)
            rec = {'seed': seed, 'infinite': {'F_mean': F_inf.mean(), 'purity_pred': describe(pur0),
                                              'bures_angle_to_maxmixed_pred': describe(bang0)}}
            for shots in (10_000, 1_000, 100):
                sum_rho = np.zeros_like(base['rho']); sum_dev2 = np.zeros(len(X))
                Fm, purs, bangs, tr_dev, min_eig, maxF = [], [], [], 0.0, 1.0, 0.0
                for _ in range(R):
                    inf = qe.infer(model, binomial_paulis(X, shots, rng))
                    F = qe.fidelity64(Rho, inf['rho'])
                    Fm.append(F.mean()); maxF = max(maxF, float(F.max()))
                    p, b = spectrum_stats(inf['rho'], d); purs.append(p); bangs.append(b)
                    sum_rho += inf['rho']
                    sum_dev2 += np.sum(np.abs(inf['rho'] - base['rho']) ** 2, axis=(1, 2))
                    tr_dev = max(tr_dev, inf['checks']['max_trace_dev'])
                    min_eig = min(min_eig, inf['checks']['min_eigenvalue'])
                rbar = sum_rho / R
                F_bar = qe.fidelity64(Rho, rbar).mean()
                bias2 = np.sum(np.abs(rbar - base['rho']) ** 2, axis=(1, 2))
                Fm = np.array(Fm)
                rec[str(shots)] = {
                    'F_mean': Fm.mean(), 'F_sd_across_realizations': Fm.std(ddof=1),
                    'relative_loss_vs_infinite': (F_inf.mean() - Fm.mean()) / F_inf.mean(),
                    'decomposition': {'total_drop': F_inf.mean() - Fm.mean(),
                                      'bias_part': F_inf.mean() - F_bar,
                                      'variance_part': F_bar - Fm.mean(),
                                      'hs_bias2': bias2.mean(),
                                      'hs_variance': (sum_dev2 / R - bias2).mean(),
                                      'note': f'mean reconstruction from {R} realizations'},
                    'purity_pred': describe(np.concatenate(purs)),
                    'bures_angle_to_maxmixed_pred': describe(np.concatenate(bangs)),
                    'checks': {'max_trace_dev': tr_dev, 'min_eigenvalue': min_eig, 'max_F': maxF}}
            inj = {}
            for sg in sigmas:
                Fs = [qe.fidelity64(Rho, qe.infer(model, np.clip(X + sg * rng.standard_normal(X.shape),
                                                                  -1, 1).astype(np.float32))['rho']).mean()
                      for _ in range(R_sigma)]
                inj[str(sg)] = {'F_mean': float(np.mean(Fs)), 'F_sd': float(np.std(Fs, ddof=1))}
            rec['gaussian_injection'] = inj
            rows.setdefault(label, []).append(rec)
            print(f'{n}q {label} seed {seed}: F inf {F_inf.mean():.4f} | 10k {rec["10000"]["F_mean"]:.4f} | '
                  f'1k {rec["1000"]["F_mean"]:.4f} | 100 {rec["100"]["F_mean"]:.4f}  [{time.time() - t0:.0f}s]',
                  flush=True)
        summary = {}
        for label, rs in rows.items():
            s = {'infinite': qe.summarize([r['infinite']['F_mean'] for r in rs])}
            for shots in ('10000', '1000', '100'):
                s[shots] = {k: qe.summarize([r[shots][k] for r in rs])
                            for k in ('F_mean', 'relative_loss_vs_infinite')}
                s[shots]['decomposition'] = {k: qe.summarize([r[shots]['decomposition'][k] for r in rs])
                                             for k in ('total_drop', 'bias_part', 'variance_part',
                                                       'hs_bias2', 'hs_variance')}
                s[shots]['purity_pred_mean'] = qe.summarize([r[shots]['purity_pred']['mean'] for r in rs])
                s[shots]['bures_to_maxmixed_mean'] = qe.summarize(
                    [r[shots]['bures_angle_to_maxmixed_pred']['mean'] for r in rs])
            s['gaussian_injection'] = {sg: qe.summarize([r['gaussian_injection'][sg]['F_mean'] for r in rs])
                                       for sg in rs[0]['gaussian_injection']}
            summary[label] = s
        eff_sigma = {str(sh): float(np.sqrt(np.mean((1 - X.astype(np.float64) ** 2) / sh)))
                     for sh in (10_000, 1_000, 100)}
        out[f'{n}q'] = {'per_model': rows, 'summary': summary,
                        'true_states': {'purity': describe(pur_true),
                                        'bures_angle_to_maxmixed': describe(bang_true)},
                        'binomial_effective_sigma': eff_sigma,
                        'old_pipeline_perfect_predictor': old_pipeline_perfect_predictor(n, X, Rho),
                        'protocol': 'Binomial sampling of each Pauli expectation on the encoder input (Eq. 27); '
                                    f'{R} realizations per shot count; full test set.'}
    save(f'shots_{which}', out)


# ------------------------------------------------------- per-state circuit fits
def check_circuit64(n, trials=6):
    nb.ensure_basis(n)
    circ32 = nb.TorchDMCircuit(n, nb.n_layers, nb.n_angles, nb.n_noise_params,
                               nb.full_paulis_arr, nb.pauli_idx_nonid)
    g = torch.Generator().manual_seed(n)
    ang = (torch.rand(trials, nb.n_angles, generator=g, dtype=torch.float64) * 2 - 1) * np.pi
    noise = torch.rand(trials, nb.n_noise_params, generator=g, dtype=torch.float64) * 0.15
    with torch.no_grad():
        r32 = circ32.forward_rho(torch.cat([ang, noise], -1).float()).numpy()
        r64 = qe.Circuit64(n).rho(ang, noise).numpy()
    return float(np.abs(r32 - r64).max())


def run_oracle(per_family=30, restarts=3, steps=400, lr=0.03):
    out = {'ratio_table': [{'n': n, 'circuit_params': (n + 2) * (3 * n + 1), 'state_dim': 4 ** n - 1,
                            'pure_state_dim': 2 ** (n + 1) - 2,
                            'params_over_state_dim': (n + 2) * (3 * n + 1) / (4 ** n - 1)}
                           for n in range(2, 11)]}
    for n in QUBITS:
        t0 = time.time()
        dev = check_circuit64(n)
        assert dev < 1e-5, f'Circuit64 disagrees with the original simulator at {n}q: {dev}'
        ts = qe.load_testset(n)
        rng = np.random.default_rng(77 + n)
        S = np.concatenate([rng.choice(np.where(ts['labels'] == k)[0], per_family, replace=False)
                            for k in range(len(qe.FAMILIES))])
        model = nb.load_model(n, qe.orig_ckpt(n, 42))
        with torch.no_grad():
            z, _, _ = model.encoder(torch.tensor(ts['X'][S], dtype=torch.float32))
            ang0 = model.latent2angles(z).double()
            p0 = (torch.sigmoid(model.latent2noise(z)) * (nb.NOISE_MAX - nb.NOISE_MIN) + nb.NOISE_MIN).double()
        circ = qe.Circuit64(n)
        Rt = torch.tensor(ts['Rho'][S], dtype=torch.complex128)
        with torch.no_grad():
            F_model = qe.fidelity_torch64(Rt, circ.rho(ang0, p0)).numpy()
        K, Bn = 1 + restarts, len(S)
        g = torch.Generator().manual_seed(100 + n)
        ang = torch.cat([ang0] + [torch.rand(Bn, ang0.shape[1], generator=g, dtype=torch.float64) * 2 * np.pi - np.pi
                                  for _ in range(restarts)]).clone().requires_grad_(True)
        u0 = torch.logit(torch.clamp((p0 - nb.NOISE_MIN) / (nb.NOISE_MAX - nb.NOISE_MIN), 1e-4, 1 - 1e-4))
        u = torch.cat([u0] + [torch.zeros(Bn, p0.shape[1], dtype=torch.float64)
                              for _ in range(restarts)]).clone().requires_grad_(True)
        RtK = Rt.repeat(K, 1, 1)
        opt = torch.optim.Adam([ang, u], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            F = qe.fidelity_torch64(RtK, circ.rho(ang, torch.sigmoid(u) * (nb.NOISE_MAX - nb.NOISE_MIN) + nb.NOISE_MIN))
            (-F.mean()).backward()
            opt.step()
        with torch.no_grad():
            F = qe.fidelity_torch64(RtK, circ.rho(ang, torch.sigmoid(u) * (nb.NOISE_MAX - nb.NOISE_MIN)
                                                  + nb.NOISE_MIN)).reshape(K, Bn).numpy()
        F_or = F.max(axis=0)
        lab = ts['labels'][S]
        out[f'{n}q'] = {'n_states': int(Bn), 'circuit_params': int(ang0.shape[1] + p0.shape[1]),
                        'circuit64_max_abs_deviation_from_original': dev,
                        'settings': {'model': 'seed 42', 'restarts': restarts, 'steps': steps, 'lr': lr,
                                     'optimizer': 'Adam', 'objective': 'Uhlmann fidelity to the true state'},
                        'model_F': F_model.mean(), 'per_state_fit_F': F_or.mean(),
                        'warm_start_only_F': F[0].mean(), 'random_init_only_F': F[1:].max(axis=0).mean(),
                        'per_family': {fam: {'model_F': F_model[lab == k].mean(), 'per_state_fit_F': F_or[lab == k].mean()}
                                       for k, fam in enumerate(qe.FAMILIES)}}
        print(f'{n}q: model F {F_model.mean():.4f} -> per-state fit {F_or.mean():.4f} '
              f'(warm {F[0].mean():.4f}, random {F[1:].max(axis=0).mean():.4f})  [{time.time() - t0:.0f}s]', flush=True)
    save('oracle', out)


# -------------------------------------------------------------------- warm start
def run_warmstart(per_family=30, checkpoints=(0, 10, 25, 50, 100, 200), restarts=3, lr=0.03):
    """Refine circuit parameters per state against the measured Pauli vector only.

    Loss: ||x_hat(theta, p) - x_measured||^2 (no access to the true state). Starts:
    the encoder's prediction (seed-42 model) or `restarts` random initializations,
    of which the one with the lowest loss is reported at every checkpoint.
    Fidelity to the true state is recorded along the way. Noiseless and 1k-shot data.
    """
    out = {}
    span = nb.NOISE_MAX - nb.NOISE_MIN
    for n in QUBITS:
        t0 = time.time()
        ts = qe.load_testset(n)
        rng = np.random.default_rng(77 + n)
        S = np.concatenate([rng.choice(np.where(ts['labels'] == k)[0], per_family, replace=False)
                            for k in range(len(qe.FAMILIES))])
        model = nb.load_model(n, qe.orig_ckpt(n, 42))
        nb.ensure_basis(n)
        P = torch.tensor(nb.full_paulis_arr[nb.pauli_idx_nonid], dtype=torch.complex128)
        Rt = torch.tensor(ts['Rho'][S], dtype=torch.complex128)
        circ = qe.Circuit64(n)
        res_n = {}
        for data_kind in ('noiseless', '1000_shots'):
            X = ts['X'][S].astype(np.float32)
            if data_kind == '1000_shots':
                X = binomial_paulis(X, 1000, np.random.default_rng(500 + n))
            with torch.no_grad():
                z, _, _ = model.encoder(torch.tensor(X, dtype=torch.float32))
                ang0 = model.latent2angles(z).double()
                p0 = (torch.sigmoid(model.latent2noise(z)) * span + nb.NOISE_MIN).double()
            Bn = len(S)
            g = torch.Generator().manual_seed(200 + n)
            starts = {'encoder': (ang0, torch.logit(torch.clamp((p0 - nb.NOISE_MIN) / span, 1e-4, 1 - 1e-4))),
                      'random': (torch.rand(restarts * Bn, ang0.shape[1], generator=g, dtype=torch.float64) * 2 * np.pi - np.pi,
                                 torch.zeros(restarts * Bn, p0.shape[1], dtype=torch.float64))}
            xm = torch.tensor(X, dtype=torch.float64)
            curves = {}
            for name, (a0, u0) in starts.items():
                K = a0.shape[0] // Bn
                ang = a0.clone().requires_grad_(True); u = u0.clone().requires_grad_(True)
                opt = torch.optim.Adam([ang, u], lr=lr)
                xmK, RtK = xm.repeat(K, 1), Rt.repeat(K, 1, 1)
                curve = {}
                for step in range(max(checkpoints) + 1):
                    rho = circ.rho(ang, torch.sigmoid(u) * span + nb.NOISE_MIN)
                    xh = torch.einsum('bij,kji->bk', rho, P).real
                    loss_each = ((xh - xmK) ** 2).sum(-1)
                    if step in checkpoints:
                        with torch.no_grad():
                            F = qe.fidelity_torch64(RtK, rho).reshape(K, Bn)
                            best = loss_each.detach().reshape(K, Bn).argmin(0)
                            curve[step] = float(F[best, torch.arange(Bn)].mean())
                    if step == max(checkpoints):
                        break
                    opt.zero_grad(); loss_each.mean().backward(); opt.step()
                curves[name] = curve
            res_n[data_kind] = curves
            print(f'{n}q {data_kind}: encoder start {curves["encoder"]} | random {curves["random"]}  '
                  f'[{time.time() - t0:.0f}s]', flush=True)
        out[f'{n}q'] = {'n_states': int(len(S)), 'settings': {'model': 'seed 42', 'lr': lr, 'optimizer': 'Adam',
                        'random_restarts': restarts, 'objective': 'squared error to the measured Pauli vector'},
                        'curves': res_n}
    save('warmstart', out)


# --------------------------------------------------------- latent visualisation
def run_latentviz():
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    out = {}
    for n in QUBITS:
        ts = qe.load_testset(n)
        c = cached_inference('original', n, 'original', 42, qe.orig_ckpt(n, 42), ts['X'])
        zg = c['zg'].astype(np.float64)
        pca = PCA(2).fit(zg)
        tsne = TSNE(2, perplexity=30, init='pca', random_state=0).fit_transform(zg)
        neg = qe.negativity(ts['Rho'].astype(np.complex128), n)
        np.savez_compressed(os.path.join(qe.RESULTS, f'latentviz_{n}q.npz'), pca=pca.transform(zg),
                            tsne=tsne, labels=ts['labels'], purity=ts['purity'], negativity=neg)
        out[f'{n}q'] = {'model': 'seed 42', 'n_states': len(zg),
                        'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                        'tsne': {'perplexity': 30, 'init': 'pca', 'random_state': 0},
                        'negativity_cut': f'first {n // 2} qubit(s) | rest'}
        print(f'{n}q latent visualization done', flush=True)
    save('latentviz', out)


# ----------------------------------------------------------------------- timing
def run_timing(n_pairs=10000, repeats=7):
    out = {}
    for n in QUBITS:
        ts = qe.load_testset(n)
        model = nb.load_model(n, qe.orig_ckpt(n, 42))
        rng = np.random.default_rng(3)
        i = rng.integers(0, len(ts['X']), n_pairs); j = rng.integers(0, len(ts['X']), n_pairs)
        X = torch.tensor(ts['X'], dtype=torch.float32)
        with torch.no_grad():
            model.encoder(X[:100])                      # warm-up
            t = time.perf_counter(); zg = model.encoder(X)[1].numpy().astype(np.float64)
            t_enc = (time.perf_counter() - t) / len(X)
        A, Bm = ts['Rho'][i].astype(np.complex128), ts['Rho'][j].astype(np.complex128)

        def best(fn):
            ts_ = []
            for _ in range(repeats):
                t = time.perf_counter(); fn(); ts_.append(time.perf_counter() - t)
            return min(ts_) / n_pairs
        t_lat = best(lambda: np.linalg.norm(zg[i] - zg[j], axis=1))
        t_fid = best(lambda: qe.fidelity64(A, Bm))
        out[f'{n}q'] = {'latent_distance_s_per_pair': t_lat, 'exact_fidelity_s_per_pair': t_fid,
                        'speedup': t_fid / t_lat, 'encoder_s_per_state': t_enc,
                        'latent_dim': int(zg.shape[1]), 'matrix_dim': 2 ** n,
                        'note': 'CPU, 8 threads, batched numpy over 10,000 pairs, best of 7'}
        print(f'{n}q: latent {t_lat * 1e9:.0f} ns/pair, fidelity {t_fid * 1e6:.1f} µs/pair, '
              f'speedup {t_fid / t_lat:.0f}x, encoder {t_enc * 1e6:.1f} µs/state', flush=True)
    save('timing', out)


# ------------------------------------------- purity outside the training window
def run_ood(which):
    out = {}
    for n in QUBITS:
        od = qe.load_ood(n)
        rows = {}
        for label, seed, path in qe.model_list(which, n):
            model = nb.load_model(n, path)
            F = qe.fidelity64(od['Rho'], qe.infer(model, od['X'])['rho'])
            rows.setdefault(label, []).append({'seed': seed, 'per_bin': [F[od['bin'] == b].mean()
                                                                         for b in range(len(od['bins']))]})
        summary = {label: [qe.summarize([r['per_bin'][b] for r in rs]) for b in range(len(od['bins']))]
                   for label, rs in rows.items()}
        out[f'{n}q'] = {'bins': od['bins'].tolist(), 'generator': 'Depolarized (Haar + white noise)',
                        'per_model': rows, 'summary': summary}
        print(f'{n}q OOD: ' + ', '.join(f'{lo:.2f}-{hi:.2f}: {s["mean"]:.3f}' for (lo, hi), s in
                                         zip(od['bins'], summary['original' if which == 'original' else list(summary)[0]])),
              flush=True)
    save(f'ood_{which}', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['fidelity', 'geometry', 'shots', 'oracle', 'warmstart', 'latentviz',
                                        'timing', 'ood', 'all'])
    ap.add_argument('--models', default='original', choices=['original', 'ablation'])
    a = ap.parse_args()
    cmds = ['fidelity', 'geometry', 'shots', 'oracle', 'latentviz', 'timing', 'ood'] if a.command == 'all' else [a.command]
    for c in cmds:
        t0 = time.time()
        print(f'=== {c} ({a.models}) ===', flush=True)
        if c in ('oracle', 'warmstart', 'latentviz', 'timing'):
            globals()[f'run_{c}']()
        else:
            globals()[f'run_{c}'](a.models)
        print(f'=== {c} finished in {time.time() - t0:.0f}s ===', flush=True)


if __name__ == '__main__':
    main()
