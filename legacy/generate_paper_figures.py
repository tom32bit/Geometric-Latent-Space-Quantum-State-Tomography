"""
generate_paper_figures.py — Composite figures for the paper update
=================================================================
Generates figures that show cross-scale comparisons (2q, 3q, 4q)
for both data-limited and data-sufficient regimes.

Run locally (not on Kaggle) — only needs the result folders.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ============================================================================
# Data — manually extracted from logs
# ============================================================================

# Data-limited regime (n_train=5000 for all)
limited = {
    'qubits':       [2, 3, 4],
    'n_train':      [5000, 5000, 5000],
    'samples_param':[333, 79, 20],
    'fidelity':     [0.9584, 0.9176, 0.6939],
    'fid_median':   [None, None, 0.7163],  # 2q/3q median not in logs
    'pearson_r':    [0.9247, 0.8200, 0.8368],
    'spearman_rho': [0.8386, 0.5743, 0.4212],
    'R2':           [0.8551, 0.6724, 0.7002],
    'pca_95':       [7, 21, 32],
    'curvature':    [0.0039, 0.0129, 0.0438],
    'shot_inf':     [0.9584, 0.9176, 0.6939],   # analytical
    'shot_10k':     [None, None, 0.6985],
    'shot_1k':      [None, None, 0.7244],
    'shot_100':     [None, None, 0.7496],
}

# Data-sufficient regime (~300 samples/param)
saturated = {
    'qubits':       [2, 3, 4],
    'n_train':      [5000, 18900, 76500],
    'samples_param':[333, 300, 300],
    'fidelity':     [0.9584, 0.9234, 0.8366],
    'fid_median':   [None, 0.9353, 0.8408],
    'pearson_r':    [0.9247, 0.8103, 0.8096],
    'spearman_rho': [0.8386, 0.5415, 0.4277],
    'R2':           [0.8551, 0.6566, 0.6554],
    'pca_95':       [7, 20, 31],
    'curvature':    [0.0039, 0.0114, 0.0494],
    'latent_dim':   [20, 30, 40],
    'shot_inf':     [0.9584, 0.9234, 0.8366],
    'shot_10k':     [None, 0.9199, 0.8328],
    'shot_1k':      [None, 0.9201, 0.8324],
    'shot_100':     [None, 0.9398, 0.8876],
    'fid_std':      [None, 0.0666, 0.1069],
}

OUTDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================================
# Figure A — Fidelity Scaling: Data-limited vs Data-sufficient
# ============================================================================

def fig_fidelity_scaling():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), gridspec_kw={'width_ratios': [1, 1, 1]})
    fig.suptitle('Multi-Qubit Scaling: Data-Limited vs Data-Sufficient', fontsize=14, fontweight='bold', y=1.02)

    qubits = [2, 3, 4]
    x = np.arange(len(qubits))
    w = 0.30

    # (A) Fidelity comparison
    ax = axes[0]
    bars1 = ax.bar(x - w/2, limited['fidelity'], w, label='Data-limited\n(n=5,000)', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + w/2, saturated['fidelity'], w, label='Data-sufficient\n(300 s/p)', color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Mean Fidelity F', fontsize=11)
    ax.set_xlabel('Qubits', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(['2', '3', '4'])
    ax.set_ylim(0.55, 1.0)
    ax.legend(fontsize=8, loc='lower left')
    ax.set_title('(A) Reconstruction Fidelity', fontsize=11)
    ax.axhline(y=0.90, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.text(2.4, 0.905, 'F=0.90', fontsize=7, color='gray')
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7.5)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7.5)

    # (B) Infidelity scaling (log)
    ax = axes[1]
    inf_lim = [1 - f for f in limited['fidelity']]
    inf_sat = [1 - f for f in saturated['fidelity']]
    ax.semilogy(qubits, inf_lim, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Data-limited', zorder=5)
    ax.semilogy(qubits, inf_sat, 's-', color='#2ecc71', linewidth=2, markersize=8, label='Data-sufficient', zorder=5)
    # O(4^n) reference line (normalized to 2q data-sufficient)
    ref_2q = inf_sat[0]
    ref_line = [ref_2q * (4**(q-2)) for q in qubits]
    ax.semilogy(qubits, ref_line, '--', color='gray', linewidth=1.5, alpha=0.6, label=r'$O(4^n)$ scaling')
    ax.set_ylabel('Infidelity (1 - F)', fontsize=11)
    ax.set_xlabel('Qubits', fontsize=11)
    ax.set_xticks(qubits)
    ax.legend(fontsize=8)
    ax.set_title('(B) Infidelity Scaling', fontsize=11)
    ax.grid(True, alpha=0.3)

    # (C) Pearson r across scales
    ax = axes[2]
    ax.bar(x - w/2, limited['pearson_r'], w, label='Data-limited', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, saturated['pearson_r'], w, label='Data-sufficient', color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Pearson r', fontsize=11)
    ax.set_xlabel('Qubits', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(['2', '3', '4'])
    ax.set_ylim(0.70, 1.0)
    ax.axhline(y=0.80, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.text(2.4, 0.805, 'r=0.80', fontsize=7, color='gray')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_title('(C) Metric Preservation', fontsize=11)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'fig_scaling_comparison.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


# ============================================================================
# Figure B — Manifold Geometry Across Scales
# ============================================================================

def fig_manifold_scaling():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('Manifold Geometry Across Qubit Scales (Data-Sufficient)', fontsize=14, fontweight='bold', y=1.02)

    qubits = [2, 3, 4]

    # (A) PCA dimensionality vs latent dim
    ax = axes[0]
    latent = saturated['latent_dim']
    pca = saturated['pca_95']
    x = np.arange(len(qubits))
    w = 0.30
    ax.bar(x - w/2, latent, w, label='Latent dim', color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, pca, w, label='PCA 95% dim', color='#e67e22', alpha=0.85, edgecolor='black', linewidth=0.5)
    for i, (l, p) in enumerate(zip(latent, pca)):
        headroom = (l - p) / l * 100
        ax.text(i, max(l, p) + 1, f'{headroom:.0f}%\nfree', ha='center', fontsize=7, color='gray')
    ax.set_ylabel('Dimensions', fontsize=11)
    ax.set_xlabel('Qubits', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(['2', '3', '4'])
    ax.legend(fontsize=8)
    ax.set_title('(A) Latent Compression', fontsize=11)

    # (B) Curvature scaling
    ax = axes[1]
    curv = saturated['curvature']
    ax.semilogy(qubits, curv, 'o-', color='#9b59b6', linewidth=2, markersize=10, zorder=5)
    for i, (q, c) in enumerate(zip(qubits, curv)):
        ax.annotate(f'{c:.4f}', (q, c), textcoords='offset points', xytext=(12, 5), fontsize=9)
    # Show ~4x per qubit reference
    ref = [curv[0] * (4**(q-2)) for q in qubits]
    ax.semilogy(qubits, ref, '--', color='gray', alpha=0.5, linewidth=1.5, label=r'$\sim 4\times$ per qubit')
    ax.set_ylabel(r'Mean curvature $\kappa$', fontsize=11)
    ax.set_xlabel('Qubits', fontsize=11)
    ax.set_xticks(qubits)
    ax.legend(fontsize=8)
    ax.set_title('(B) Curvature Scaling', fontsize=11)
    ax.grid(True, alpha=0.3)

    # (C) Pearson vs Spearman gap
    ax = axes[2]
    ax.plot(qubits, saturated['pearson_r'], 'o-', color='#2ecc71', linewidth=2, markersize=8, label='Pearson r (global)')
    ax.plot(qubits, saturated['spearman_rho'], 's-', color='#e74c3c', linewidth=2, markersize=8, label=r'Spearman $\rho$ (local)')
    # Shade the gap
    ax.fill_between(qubits, saturated['pearson_r'], saturated['spearman_rho'],
                     alpha=0.15, color='orange', label='Curvature distortion')
    for i, q in enumerate(qubits):
        gap = saturated['pearson_r'][i] - saturated['spearman_rho'][i]
        mid = (saturated['pearson_r'][i] + saturated['spearman_rho'][i]) / 2
        ax.text(q + 0.08, mid, f'gap={gap:.2f}', fontsize=8, color='#e67e22')
    ax.set_ylabel('Correlation', fontsize=11)
    ax.set_xlabel('Qubits', fontsize=11)
    ax.set_xticks(qubits)
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=8, loc='lower left')
    ax.set_title('(C) Global vs Local Geometry', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'fig_manifold_scaling.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


# ============================================================================
# Figure C — Shot Noise Robustness (data-sufficient, 3q + 4q)
# ============================================================================

def fig_shot_noise_summary():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle('Shot Noise Robustness (Data-Sufficient Regime)', fontsize=14, fontweight='bold', y=1.02)

    conditions = [r'$\infty$', '10,000', '1,000', '100']
    x = np.arange(len(conditions))
    w = 0.30

    # 3q
    ax = axes[0]
    vals_3q = [0.9234, 0.9199, 0.9201, 0.9398]
    colors_3q = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']
    bars = ax.bar(x, vals_3q, 0.6, color=colors_3q, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, vals_3q):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Mean Fidelity F', fontsize=11)
    ax.set_xlabel('Shot Count', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0.88, 0.96)
    ax.set_title('(A) 3-Qubit', fontsize=11)

    # 4q
    ax = axes[1]
    vals_4q = [0.8366, 0.8328, 0.8324, 0.8876]
    bars = ax.bar(x, vals_4q, 0.6, color=colors_3q, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, vals_4q):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Mean Fidelity F', fontsize=11)
    ax.set_xlabel('Shot Count', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0.78, 0.92)
    ax.set_title('(B) 4-Qubit', fontsize=11)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'fig_shot_noise_summary.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


# ============================================================================
# Figure D — Data Sufficiency: samples/param vs fidelity
# ============================================================================

def fig_data_sufficiency():
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Data Sufficiency: Samples per Parameter vs Fidelity', fontsize=13, fontweight='bold')

    # Each qubit has two points: limited and saturated
    colors = {'2': '#3498db', '3': '#e67e22', '4': '#e74c3c'}
    markers = {'2': 'o', '3': 's', '4': 'D'}

    for i, q in enumerate([2, 3, 4]):
        sp_lim = limited['samples_param'][i]
        sp_sat = saturated['samples_param'][i]
        f_lim = limited['fidelity'][i]
        f_sat = saturated['fidelity'][i]
        c = colors[str(q)]
        m = markers[str(q)]

        # Plot both points
        ax.scatter(sp_lim, f_lim, color=c, marker=m, s=120, zorder=5, edgecolors='black', linewidth=0.8)
        ax.scatter(sp_sat, f_sat, color=c, marker=m, s=120, zorder=5, edgecolors='black', linewidth=0.8)

        # Connect with arrow
        if sp_lim != sp_sat:
            ax.annotate('', xy=(sp_sat, f_sat), xytext=(sp_lim, f_lim),
                        arrowprops=dict(arrowstyle='->', color=c, linewidth=1.5, alpha=0.6))

        # Label
        ax.text(sp_sat + 10, f_sat + 0.008, f'{q}q: F={f_sat:.3f}', fontsize=9, color=c, fontweight='bold')
        if sp_lim != sp_sat:
            ax.text(sp_lim + 5, f_lim - 0.025, f'{q}q limited: F={f_lim:.3f}', fontsize=8, color=c, alpha=0.7)

    ax.set_xlabel('Samples per Free Parameter', fontsize=12)
    ax.set_ylabel('Mean Fidelity F', fontsize=12)
    ax.set_xlim(0, 370)
    ax.set_ylim(0.60, 1.0)
    ax.axhline(y=0.90, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.2)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, markeredgecolor='black', label='2-qubit'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#e67e22', markersize=10, markeredgecolor='black', label='3-qubit'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#e74c3c', markersize=10, markeredgecolor='black', label='4-qubit'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='lower right')

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'fig_data_sufficiency.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


# ============================================================================
# Run all
# ============================================================================

if __name__ == '__main__':
    fig_fidelity_scaling()
    fig_manifold_scaling()
    fig_shot_noise_summary()
    fig_data_sufficiency()
    print(f'\nAll figures saved to {OUTDIR}')
