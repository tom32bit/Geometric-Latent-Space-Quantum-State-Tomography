"""Regenerate the revised manuscript's figures from result files only.

Inputs: revision/results/*.json and *.npz, revision/data/testset_*.npz, the cached
latents in revision/results/cache/ (written by `run_analyses.py fidelity`), and
'legacy/4q saturated/training_history_4q.pkl'.
Outputs: revision/figures/*.pdf (optionally copied to --copy-to).

Colours follow a validated categorical palette (blue, orange, aqua: all pairs pass
the colour-vision checks) with a marker shape per series as secondary encoding.

Usage:  python revision/make_figures.py [--copy-to DIR]
"""
import argparse
import json
import os
import pickle
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REV = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(REV)
RES = os.path.join(REV, 'results')
OUT = os.path.join(REV, 'figures')

INK, INK2, MUTED, GRID, BASE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'          # categorical slots 1-3
QCOL = {2: C1, 3: C2, 4: C3}
QMARK = {2: 'o', 3: 's', 4: '^'}
BLUES = LinearSegmentedColormap.from_list('blues', ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
FAMILY_PANELS = [('Werner', [1]), ('Haar + white noise\n(Depolarized, Isotropic)', [0, 2]),
                 ('Amplitude damping', [3]), ('Phase damping', [4]), ('Thermal (GUE)', [5]),
                 ('Separable', [6])]

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'legend.frameon': False,
    'axes.edgecolor': BASE, 'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'grid.linestyle': '-',
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.4,
    'lines.markersize': 5, 'savefig.dpi': 300, 'pdf.fonttype': 42})


def load(name):
    return json.load(open(os.path.join(RES, name)))


def panel_label(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=9, va='bottom')


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def series(ax, xs, ys, es, n, label=None, **kw):
    ax.errorbar(xs, ys, yerr=es, color=QCOL[n], marker=QMARK[n], capsize=2, elinewidth=0.8,
                markeredgecolor='white', markeredgewidth=0.6, label=label or f'{n} qubits', **kw)


# ---------------------------------------------------------------- Figure 2
def fig2():
    h = pickle.load(open(os.path.join(ROOT, 'legacy', '4q saturated', 'training_history_4q.pkl'), 'rb'))
    ep = np.array(h['epoch'])
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.3))
    a.plot(ep, h['val_fid'], color=C1)
    a.set_xlabel('Epoch'); a.set_ylabel('Validation fidelity'); a.set_ylim(0.4, 0.9)
    best = int(np.argmax(h['val_fid']))
    a.annotate(f'best {h["val_fid"][best]:.3f} (epoch {ep[best]})', (ep[best], h['val_fid'][best]),
               xytext=(-95, -28), textcoords='offset points', color=INK2, fontsize=7,
               arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
    for key, lab, col, mk in (('loss_total', 'total', INK2, None), ('loss_recon', 'reconstruction', C1, None),
                              ('loss_metric', 'metric', C2, None)):
        b.plot(ep, h[key], color=col, label=lab)
    b.set_xlabel('Epoch'); b.set_ylabel('Training loss'); b.legend(loc='upper right')
    panel_label(a, 'A'); panel_label(b, 'B')
    save(fig, 'fig2_training_dynamics_4q.pdf')


# ---------------------------------------------------------------- Figure 3
def fig3():
    fid, geo = load('fidelity_original.json'), load('geometry_original.json')
    ns = [2, 3, 4]
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2))
    F = [fid[f'{n}q']['summary']['original']['mean'] for n in ns]
    Fn = [fid[f'{n}q']['summary']['original']['mean_no_werner'] for n in ns]
    axs[0].errorbar(ns, [f['mean'] for f in F], [f['sd'] for f in F], color=C1, marker='o', capsize=2,
                    label='all states', markeredgecolor='white')
    axs[0].errorbar(ns, [f['mean'] for f in Fn], [f['sd'] for f in Fn], color=C2, marker='s', capsize=2,
                    label='excl. Werner', markeredgecolor='white')
    axs[0].set_ylabel('Mean fidelity'); axs[0].legend(loc='lower left')
    for key, lab, col, mk in (('all', 'all pairs', C1, 'o'), ('no_werner', 'excl. Werner', C2, 's')):
        v = [geo[f'{n}q']['summary']['original'][key]['pearson'] for n in ns]
        axs[1].errorbar(ns, [x['mean'] for x in v], [x['sd'] for x in v], color=col, marker=mk, capsize=2,
                        label=lab, markeredgecolor='white')
    axs[1].set_ylabel('Pearson $r$ (latent vs Bures)'); axs[1].set_ylim(0, 1); axs[1].legend(loc='lower left')
    axs[2].plot(ns, [1 - f['mean'] for f in F], color=C1, marker='o', markeredgecolor='white')
    axs[2].set_yscale('log'); axs[2].set_ylabel('Infidelity $1-F$')
    axs[2].set_yticks([0.02, 0.05, 0.1, 0.2]); axs[2].set_yticklabels(['0.02', '0.05', '0.1', '0.2'])
    axs[2].minorticks_off()
    for v, n in zip([1 - f['mean'] for f in F], ns):
        axs[2].annotate(f'{v:.3f}', (n, v), xytext=(4, -9), textcoords='offset points', fontsize=7, color=INK2)
    for ax, lab in zip(axs, 'ABC'):
        ax.set_xticks(ns); ax.set_xlabel('Qubits'); panel_label(ax, lab)
    fig.tight_layout(w_pad=1.5)
    save(fig, 'fig3_scaling.pdf')


# ---------------------------------------------------------------- Figure 5
def fig5(n=3, n_show=2000):
    import qst_eval as qe
    geo = load('geometry_original.json')
    rows = geo[f'{n}q']['per_model']['original']
    mean_r = geo[f'{n}q']['summary']['original']['all']['pearson']['mean']
    pick = min(rows, key=lambda r: abs(r['all']['pearson'] - mean_r))
    seed = pick['seed']
    ts = qe.load_testset(n); lab = ts['labels']; N = len(lab)
    zg = np.load(os.path.join(RES, 'cache', f'original_{n}q_original_seed{seed}.npz'))['zg'].astype(float)
    rng = np.random.default_rng(2026 + n)
    i = rng.integers(0, N, 20000); j = rng.integers(0, N, 20000)
    keep = i != j; i, j = i[keep], j[keep]
    db = qe.bures_angle(qe.fidelity64(ts['Rho'][i], ts['Rho'][j]))
    dl = np.linalg.norm(zg[i] - zg[j], axis=1)
    wi, wj = lab[i] == qe.WERNER, lab[j] == qe.WERNER
    show = np.random.default_rng(1).choice(len(i), n_show, replace=False)
    groups = [('neither state Werner', ~wi & ~wj, BASE, 6), ('one Werner state', wi ^ wj, C1, 8),
              ('Werner–Werner', wi & wj, C2, 14)]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for name, m, col, s in groups:
        sel = show[m[show]]
        ax.scatter(db[sel], dl[sel], s=s, color=col, edgecolors='white', linewidths=0.3,
                   label=f'{name} ({m.mean() * 100:.0f}% of pairs)', zorder=3 if col != BASE else 2)
    a, b = np.polyfit(db, dl, 1)
    xx = np.array([0, db.max()])
    ax.plot(xx, a * xx + b, color=INK2, lw=1.0, label='least-squares line (all pairs)')
    ax.set_xlabel('Bures distance $d_B$ (rad)'); ax.set_ylabel('Latent distance $d_L$')
    ax.legend(loc='lower right', fontsize=6.8, handletextpad=0.3)
    ax.set_title(f'{n} qubits, seed {seed}: r = {pick["all"]["pearson"]:.3f} (all pairs), '
                 f'{pick["no_werner"]["pearson"]:.3f} (excl. Werner)', fontsize=7.5, color=INK2)
    save(fig, f'fig5_latent_bures_{n}q.pdf')
    return {'seed': seed, 'r_all': pick['all']['pearson'], 'r_no_werner': pick['no_werner']['pearson'],
            'n_points_shown': n_show}


# ---------------------------------------------------------------- Figure 6
def fig6():
    geo = load('geometry_original.json')
    ns = [2, 3, 4]
    S = [geo[f'{n}q']['summary']['original'] for n in ns]
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2))
    axs[0].errorbar(ns, [s['pca_dim95_full_z']['mean'] for s in S], [s['pca_dim95_full_z']['sd'] for s in S],
                    color=C1, marker='o', capsize=2, label='$d_{\\mathrm{PCA}}(95\\%)$', markeredgecolor='white')
    axs[0].plot(ns, [10 * n for n in ns], color=MUTED, lw=1.0, marker='_', label='$d_z = 10n$')
    axs[0].set_ylabel('Latent dimensions'); axs[0].legend(loc='upper left')
    axs[1].errorbar(ns, [s['kappa']['mean'] for s in S], [s['kappa']['sd'] for s in S], color=C1, marker='o',
                    capsize=2, markeredgecolor='white')
    axs[1].set_ylabel(r'Mean anisotropy ratio $\bar{\kappa}$')
    for key, lab, col, mk in (('all', 'all pairs', C1, 'o'), ('no_werner', 'excl. Werner', C2, 's')):
        axs[2].errorbar(ns, [s[key]['spearman']['mean'] for s in S], [s[key]['spearman']['sd'] for s in S],
                        color=col, marker=mk, capsize=2, label=lab, markeredgecolor='white')
    axs[2].set_ylabel('Spearman $r_s$ (latent vs Bures)'); axs[2].set_ylim(0, 1); axs[2].legend(loc='upper right')
    for ax, lab in zip(axs, 'ABC'):
        ax.set_xticks(ns); ax.set_xlabel('Qubits'); panel_label(ax, lab)
    fig.tight_layout(w_pad=1.5)
    save(fig, 'fig6_manifold.pdf')


# ---------------------------------------------------------------- Figure 8
def fig8():
    sh = load('shots_original.json')
    ns = [2, 3, 4]
    shots = [100, 1000, 10000]
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.3))
    xinf = 10 ** 4.7
    for n in ns:
        su = sh[f'{n}q']['summary']['original']
        y = [su[str(s)]['F_mean']['mean'] for s in shots] + [su['infinite']['mean']]
        e = [su[str(s)]['F_mean']['sd'] for s in shots] + [su['infinite']['sd']]
        series(axs[0], shots + [xinf], y, e, n)
        sig = sorted(su['gaussian_injection'], key=float)
        axs[1].errorbar([float(s) for s in sig], [su['gaussian_injection'][s]['mean'] for s in sig],
                        [su['gaussian_injection'][s]['sd'] for s in sig], color=QCOL[n], marker=QMARK[n],
                        capsize=2, markeredgecolor='white', label=f'{n} qubits')
        eff = sh[f'{n}q']['binomial_effective_sigma']
        axs[1].scatter([eff[str(s)] for s in shots], [su[str(s)]['F_mean']['mean'] for s in shots], s=40,
                       facecolors='none', edgecolors=QCOL[n], linewidths=1.0, zorder=4)
        p = [su[str(s)]['purity_pred_mean']['mean'] for s in shots]
        p_inf = np.mean([r['infinite']['purity_pred']['mean'] for r in sh[f'{n}q']['per_model']['original']])
        axs[2].plot(shots + [xinf], p + [p_inf], color=QCOL[n], marker=QMARK[n], markeredgecolor='white',
                    label=f'{n} qubits')
        axs[2].axhline(sh[f'{n}q']['true_states']['purity']['mean'], color=QCOL[n], lw=0.8, ls='--')
    axs[0].set_xscale('log'); axs[0].set_xticks(shots + [xinf]); axs[0].set_xticklabels(['$10^2$', '$10^3$', '$10^4$', '$\\infty$'])
    axs[0].set_xlabel('Shots per Pauli expectation'); axs[0].set_ylabel('Mean fidelity'); axs[0].legend(loc='lower right')
    axs[1].set_xscale('log'); axs[1].set_xlabel('Input noise s.d. $\\sigma$'); axs[1].set_ylabel('Mean fidelity')
    axs[1].legend(loc='lower left')
    axs[2].set_xscale('log'); axs[2].set_xticks(shots + [xinf]); axs[2].set_xticklabels(['$10^2$', '$10^3$', '$10^4$', '$\\infty$'])
    axs[2].set_xlabel('Shots per Pauli expectation'); axs[2].set_ylabel('Mean purity of reconstructions')
    axs[2].set_ylim(0.895, 0.915); axs[2].text(0.03, 0.04, 'dashed: true states', transform=axs[2].transAxes,
                                               fontsize=6.8, color=INK2)
    for ax, lab in zip(axs, 'ABC'):
        ax.minorticks_off(); panel_label(ax, lab)
    fig.tight_layout(w_pad=1.5)
    save(fig, 'fig8_shot_noise.pdf')


# ---------------------------------------------------------------- Figure 7
LAMBDA_M_RAMP = {0.0: '#86b6ef', 0.1: '#3987e5', 0.2: '#1c5cab', 0.5: '#0d366b'}   # validated ordinal ramp
LAMBDA_P_MARK = {1.0: 'o', 2.0: 's', 5.0: '^'}


def fig7():
    path = os.path.join(RES, 'pareto_grid.json')
    if not os.path.exists(path):
        print('fig7: results/pareto_grid.json not found yet (needs the Kaggle 2q_pareto run); skipped')
        return
    rows = json.load(open(path))
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for r in rows:
        sel = r['lambda_m'] == 0.2 and r['lambda_p'] == 2.0
        ax.scatter(r['pearson_r'], r['mean_fidelity'], s=46 if sel else 30, marker=LAMBDA_P_MARK[r['lambda_p']],
                   color=LAMBDA_M_RAMP[r['lambda_m']], edgecolors=INK if sel else 'white',
                   linewidths=1.2 if sel else 0.5, zorder=3)
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], ls='', marker='o', color=c, label=f'$\\lambda_m$ = {lm:g}') for lm, c in LAMBDA_M_RAMP.items()]
    h2 = [Line2D([], [], ls='', marker=m, color=MUTED, label=f'$\\lambda_p$ = {lp:g}') for lp, m in LAMBDA_P_MARK.items()]
    ax.legend(handles=h1 + h2, loc='lower left', fontsize=6.5, ncol=2, handletextpad=0.2, columnspacing=0.8)
    ax.set_xlabel('Pearson $r$ (latent vs Bures)'); ax.set_ylabel('Validation fidelity')
    ax.set_title('outlined: selected point ($\\lambda_m$ = 0.2, $\\lambda_p$ = 2)', fontsize=7, color=INK2)
    save(fig, 'fig7_pareto.pdf')


# ------------------------------------------------------- latent visualization
def fig_latentviz(n=4, coords='tsne', name=None):
    d = np.load(os.path.join(RES, f'latentviz_{n}q.npz'))
    P, lab = d[coords], d['labels']
    fig, axs = plt.subplots(2, 4, figsize=(6.5, 3.5))
    axs = axs.ravel()
    for ax, (title, fams) in zip(axs[:6], FAMILY_PANELS):
        m = np.isin(lab, fams)
        ax.scatter(P[~m, 0], P[~m, 1], s=1.2, color=GRID, rasterized=True, lw=0)
        ax.scatter(P[m, 0], P[m, 1], s=1.6, color=C1, rasterized=True, lw=0)
        ax.set_title(title, fontsize=7, color=INK)
    for ax, key, title, cmap in ((axs[6], 'purity', 'Purity', BLUES),
                                 (axs[7], 'negativity', 'Negativity (half–half cut)', 'Oranges')):
        order = np.argsort(d[key])
        sc = ax.scatter(P[order, 0], P[order, 1], c=d[key][order], s=1.4, cmap=cmap, rasterized=True, lw=0)
        cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
        cb.ax.tick_params(labelsize=6, width=0.5); cb.outline.set_linewidth(0.4)
        ax.set_title(title, fontsize=7, color=INK)
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(GRID)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, name or f'fig_latent_{coords}_{n}q.pdf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--copy-to', default=None)
    a = ap.parse_args()
    fig2(); fig3(); info5 = fig5(); fig6(); fig7(); fig8()
    for n in (2, 3, 4):
        for coords in ('tsne', 'pca'):
            fig_latentviz(n, coords)
    with open(os.path.join(OUT, 'figure_info.json'), 'w') as f:
        json.dump({'fig5': info5}, f, indent=1)
    print('fig5:', info5)
    if a.copy_to:
        # Copy only the figures that the manuscript in that folder actually includes.
        import re
        tex = open(os.path.join(a.copy_to, 'main.tex'), encoding='utf-8').read()
        used = set(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', tex))
        copied = [fn for fn in os.listdir(OUT) if fn in used]
        for fn in copied:
            shutil.copy(os.path.join(OUT, fn), a.copy_to)
        print(f'copied {len(copied)} figure(s) used by main.tex to', a.copy_to)


if __name__ == '__main__':
    main()
