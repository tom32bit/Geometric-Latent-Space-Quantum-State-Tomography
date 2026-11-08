"""Shared helpers for the revision analyses: data, inference, metrics.

Models are always the original HybridAE from nbcore.py. Fidelities are computed
in float64 and asserted to be at most 1.
"""
import glob
import os
import sys

REV = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(REV)
LEGACY = os.path.join(ROOT, 'legacy')                      # code and outputs of the submitted version
os.environ.setdefault('QST_OUTDIR', os.path.join(REV, 'results', 'scratch'))
os.environ['PYTHONPATH'] = REV + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, REV)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import nbcore as nb  # noqa: E402

RESULTS = os.path.join(REV, 'results')
SEEDS = {2: [42, 123, 256, 789, 1024], 3: [42, 123, 256, 789, 1024], 4: [42, 123, 256]}
FAMILIES = nb.FAMILIES
WERNER = nb.WERNER
torch.set_num_threads(8)


# ----------------------------------------------------------------------------- data
def load_testset(n):
    d = np.load(os.path.join(REV, 'data', f'testset_{n}q.npz'))
    return {k: d[k] for k in ('X', 'Rho', 'labels', 'purity')}


def load_ood(n):
    d = np.load(os.path.join(REV, 'data', f'ood_{n}q.npz'))
    return {k: d[k] for k in ('X', 'Rho', 'bin', 'purity', 'bins')}


def merged_labels(labels):
    """Depolarized (0) and Isotropic (2) use the same generator: treat them as one family."""
    out = np.array(labels, copy=True)
    out[out == 2] = 0
    return out


# --------------------------------------------------------------------------- models
def orig_ckpt(n, seed):
    sub = 'run_4q_experiments' if n == 4 else 'run_all_experiments'
    return os.path.join(LEGACY, 'reviewer_experiments', sub, f'model_{n}q_seed{seed}.pt')


def ablation_ckpts(n, lambda_m):
    """Checkpoints of the Kaggle λ_m-ablation runs, found under revision/kaggle_outputs/."""
    pat = os.path.join(REV, 'kaggle_outputs', '**', f'model_{n}q_lm{lambda_m:g}_seed*.pt')
    found = {int(p.rsplit('seed', 1)[1].split('.')[0]): p for p in glob.glob(pat, recursive=True)
             if 'smoke' not in p}                      # the smoke test is a 3-epoch check run
    return dict(sorted(found.items()))


def model_list(which, n):
    """(label, seed, path) for 'original' (archived seed study) or 'ablation' models."""
    if which == 'original':
        return [('original', s, orig_ckpt(n, s)) for s in SEEDS[n]]
    out = []
    for lm in (0.0, 0.2):
        out += [(f'lambda_m={lm:g}', s, p) for s, p in ablation_ckpts(n, lm).items()]
    return out


@torch.no_grad()
def infer(model, X, bs=500):
    """Latents and decoder density matrices (complex128) for inputs X.

    The density matrix is the circuit's own output (forward_rho); for a complete
    Pauli basis it equals the paper's Pauli-vector route up to rounding.
    """
    zs, zgs, zps, rhos = [], [], [], []
    for i in range(0, len(X), bs):
        xb = torch.tensor(np.asarray(X[i:i + bs]), dtype=torch.float32, device=nb.device_torch)
        z, zg, zp = model.encoder(xb)
        noise = torch.sigmoid(model.latent2noise(z)) * (nb.NOISE_MAX - nb.NOISE_MIN) + nb.NOISE_MIN
        rho = model.circuit.forward_rho(torch.cat([model.latent2angles(z), noise], dim=-1))
        rhos.append(rho.cpu().numpy()); zs.append(z.cpu().numpy())
        zgs.append(zg.cpu().numpy()); zps.append(zp.cpu().numpy())
    rho = np.concatenate(rhos).astype(np.complex128)
    trace_dev = float(np.abs(np.trace(rho, axis1=1, axis2=2) - 1).max())
    rho = herm(rho)
    min_eig = float(np.linalg.eigvalsh(rho).min())
    rho /= np.trace(rho, axis1=1, axis2=2).real[:, None, None]
    return {'z': np.concatenate(zs), 'zg': np.concatenate(zgs), 'zp': np.concatenate(zps),
            'rho': rho, 'checks': {'max_trace_dev': trace_dev, 'min_eigenvalue': min_eig}}


# -------------------------------------------------------------------- float64 metrics
def herm(A):
    return 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))


def sqrtm_psd(A):
    w, V = np.linalg.eigh(herm(A))
    return (V * np.sqrt(np.clip(w, 0, None))[..., None, :]) @ np.conj(np.swapaxes(V, -1, -2))


def fidelity64(A, B, chunk=4000):
    """Uhlmann fidelity of batched density matrices (float64). Asserts F <= 1 + 1e-6."""
    A = np.asarray(A, dtype=np.complex128)
    B = np.broadcast_to(np.asarray(B, dtype=np.complex128), A.shape)
    out = np.empty(len(A))
    for i in range(0, len(A), chunk):
        sA = sqrtm_psd(A[i:i + chunk])
        mu = np.clip(np.linalg.eigvalsh(herm(sA @ B[i:i + chunk] @ sA)), 0, None)
        out[i:i + chunk] = np.sum(np.sqrt(mu), axis=-1) ** 2
    assert out.max() <= 1 + 1e-6, f'fidelity above 1: {out.max()}'
    return np.minimum(out, 1.0)


def bures_angle(F):
    return np.arccos(np.sqrt(np.clip(F, 0.0, 1.0)))


def purity(rho):
    return np.einsum('nij,nji->n', rho, rho).real


def negativity(rho, n):
    """Negativity across the cut between the first n//2 qubits and the rest."""
    dA = 2 ** (n // 2); dB = 2 ** n // dA
    r = rho.reshape(-1, dA, dB, dA, dB).transpose(0, 3, 2, 1, 4).reshape(rho.shape)
    ev = np.linalg.eigvalsh(herm(r))
    return -np.sum(np.clip(ev, None, 0), axis=-1)


def r2_linfit(x, y):
    """R² of the linear fit y ≈ a·x + b, as in the paper (latent distance on Bures angle)."""
    a, b = np.polyfit(x, y, 1)
    return float(1 - np.sum((y - (a * x + b)) ** 2) / np.sum((y - y.mean()) ** 2))


def wpearson(x, y, w):
    w = w / w.sum()
    dx = x - w @ x; dy = y - w @ y
    return float((w @ (dx * dy)) / np.sqrt((w @ (dx * dx)) * (w @ (dy * dy))))


def wranks(order, w):
    """Weighted mid-ranks for a fixed sort order (bootstrap Spearman)."""
    cw = np.cumsum(w[order])
    r = np.empty(len(order)); r[order] = cw - 0.5 * w[order]
    return r


def kappa_per_state(Z, k=25):
    """σ_min/σ_max of each state's k-nearest-neighbour displacements (the paper's κ)."""
    from sklearn.neighbors import NearestNeighbors
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    s = np.linalg.svd(Z[idx[:, 1:]] - Z[:, None, :], compute_uv=False)
    keep = s > 1e-10
    last = keep.sum(axis=1) - 1
    ok = last >= 1
    kap = np.full(len(Z), np.nan)
    kap[ok] = s[ok, last[ok]] / s[ok, 0]
    return kap


def pca_dim95(Z):
    from sklearn.decomposition import PCA
    cum = np.cumsum(PCA().fit(Z).explained_variance_ratio_)
    return int(np.argmax(cum >= 0.95)) + 1


def summarize(values, ddof=1):
    """Seed mean, sample sd and a t-based 95 % CI across seeds."""
    from scipy.stats import t
    v = np.asarray(values, dtype=float)
    k = len(v)
    sd = float(v.std(ddof=ddof)) if k > 1 else 0.0
    half = float(t.ppf(0.975, k - 1) * sd / np.sqrt(k)) if k > 1 else float('nan')
    return {'mean': float(v.mean()), 'sd': sd, 't_ci95': [float(v.mean()) - half, float(v.mean()) + half],
            'n_seeds': k, 'values': v.tolist()}


def percentile_ci(reps, axis=0):
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=axis)
    return [float(lo), float(hi)] if np.ndim(lo) == 0 else [lo.tolist(), hi.tolist()]


# ------------------------------------------------- float64 circuit (per-state fits)
class Circuit64:
    """float64 copy of nbcore.TorchDMCircuit, used to fit circuit parameters per state.

    Same gate order and conventions: per layer RZ·RY·RX on every qubit (qubit 0 most
    significant), a CNOT chain (q, q+1), then single-qubit depolarizing noise of
    strength p_l on every qubit. run_analyses.py checks it against the original.
    """

    def __init__(self, n):
        self.n, self.d, self.L = n, 2 ** n, n + 2
        c128 = torch.complex128
        I2 = torch.eye(2, dtype=c128)
        P = {'x': torch.tensor([[0, 1], [1, 0]], dtype=c128),
             'y': torch.tensor([[0, -1j], [1j, 0]], dtype=c128),
             'z': torch.tensor([[1, 0], [0, -1]], dtype=c128)}

        def embed(G, q):
            out = None
            for j in range(n):
                f = G if j == q else I2
                out = f if out is None else torch.kron(out, f)
            return out
        self.paulis = {k: [embed(v, q) for q in range(n)] for k, v in P.items()}
        C4 = torch.zeros(4, 4, dtype=c128)
        C4[0, 0] = C4[1, 1] = C4[2, 3] = C4[3, 2] = 1
        self.cnots = []
        for c in range(n - 1):
            g = C4
            if c > 0:
                g = torch.kron(torch.eye(2 ** c, dtype=c128), g)
            if c < n - 2:
                g = torch.kron(g, torch.eye(2 ** (n - c - 2), dtype=c128))
            self.cnots.append(g)

    @staticmethod
    def _rot(th, kind):
        c = torch.cos(th / 2).to(torch.complex128); s = torch.sin(th / 2).to(torch.complex128)
        R = torch.zeros(th.shape[0], 2, 2, dtype=torch.complex128)
        if kind == 'x':
            R[:, 0, 0] = c; R[:, 1, 1] = c; R[:, 0, 1] = -1j * s; R[:, 1, 0] = -1j * s
        elif kind == 'y':
            R[:, 0, 0] = c; R[:, 1, 1] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s
        else:
            R[:, 0, 0] = c - 1j * s; R[:, 1, 1] = c + 1j * s
        return R

    def rho(self, angles, noise):
        B, n = angles.shape[0], self.n
        rho = torch.zeros(B, self.d, self.d, dtype=torch.complex128)
        rho[:, 0, 0] = 1
        for layer in range(self.L):
            U = None
            for q in range(n):
                b = (layer * n + q) * 3
                Uq = (self._rot(angles[:, b + 2], 'z') @ self._rot(angles[:, b + 1], 'y')
                      @ self._rot(angles[:, b], 'x'))
                U = Uq if U is None else torch.einsum('bij,bkl->bikjl', U, Uq).reshape(
                    B, U.shape[1] * 2, U.shape[2] * 2)
            rho = U @ rho @ U.conj().transpose(-1, -2)
            for C in self.cnots:
                rho = C @ rho @ C
            p = noise[:, layer].to(torch.complex128).view(-1, 1, 1)
            for q in range(n):
                X, Y, Z = self.paulis['x'][q], self.paulis['y'][q], self.paulis['z'][q]
                rho = (1 - p) * rho + (p / 3) * (X @ rho @ X + Y @ rho @ Y + Z @ rho @ Z)
        return rho


def fidelity_torch64(rt, rp):
    """Differentiable float64 Uhlmann fidelity (rt: targets, rp: predictions)."""
    w, V = torch.linalg.eigh(rt)
    sA = (V * torch.sqrt(torch.clamp(w, min=0)).unsqueeze(-2).to(V.dtype)) @ V.conj().transpose(-1, -2)
    M = sA @ rp @ sA
    M = 0.5 * (M + M.conj().transpose(-1, -2))
    return torch.sum(torch.sqrt(torch.clamp(torch.linalg.eigvalsh(M), min=1e-12)), -1) ** 2
