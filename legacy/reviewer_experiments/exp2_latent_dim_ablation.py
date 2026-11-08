"""
Experiment 2: Latent Dimension Ablation
=======================================
Sweeps d_z in {10, 20, 30, 40, 50} for 3-qubit system to justify d_z = 10n.
Uses n_train=5000, 200 epochs each to keep it fast.
Reports fidelity and Pearson r vs d_z.

Estimated time: 5 runs x ~30 min = ~2.5 hours.

Usage:
    python exp2_latent_dim_ablation.py
    python exp2_latent_dim_ablation.py --latent_dims 10 20 30 40 50 60
"""

import os
import sys
import argparse
import json
import time
import random

import numpy as np
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# ============================================================================
# Arguments
# ============================================================================
parser = argparse.ArgumentParser(description="Latent dimension ablation")
parser.add_argument("--latent_dims", type=int, nargs="+", default=[10, 20, 30, 40, 50],
                    help="Latent dimensions to sweep")
parser.add_argument("--n_train", type=int, default=5000,
                    help="Training set size (fixed, not data-sufficient)")
parser.add_argument("--max_epochs", type=int, default=200,
                    help="Max training epochs per run")
parser.add_argument("--outdir", type=str, default="/kaggle/working")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

LATENT_DIMS = args.latent_dims
OUTDIR = args.outdir
SEED = args.seed
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================================
# Fixed Configuration: 3-qubit
# ============================================================================
N_QUBITS        = 3
dim             = 2 ** N_QUBITS  # 8
n_pauli_nonid   = 4 ** N_QUBITS - 1  # 63
n_layers        = N_QUBITS + 2  # 5
n_angles        = N_QUBITS * 3 * n_layers  # 45
n_noise_params  = n_layers  # 5
NOISE_MAX       = 0.15
NOISE_MIN       = 0.0

n_train         = args.n_train
n_val           = max(800, n_train // 6)
batch_size      = 64 if n_train < 10000 else 256
max_epochs      = args.max_epochs
N_JOBS_DATA     = 4

# Training hyperparams (fixed from Pareto)
lr              = 3e-3
weight_decay    = 1e-4
lambda_metric   = 0.20
lambda_purity   = 2.0
lambda_purity_enc = 0.10
eval_every      = 5
early_stop_patience = 15

device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Latent dim ablation: 3-qubit, dims={LATENT_DIMS}")
print(f"n_train={n_train}, max_epochs={max_epochs}, seed={SEED}")

# ============================================================================
# Pauli Basis
# ============================================================================
norm_factor = 1.0 / dim
I_m = np.array([[1, 0], [0, 1]],   dtype=complex)
X_m = np.array([[0, 1], [1, 0]],   dtype=complex)
Y_m = np.array([[0,-1j],[1j, 0]],  dtype=complex)
Z_m = np.array([[1, 0], [0,-1]],   dtype=complex)
_paulis_1q = [I_m, X_m, Y_m, Z_m]
_labels_1q = ['I', 'X', 'Y', 'Z']

full_paulis, full_labels = [], []
for index in range(4 ** N_QUBITS):
    digits, tmp = [], index
    for _ in range(N_QUBITS):
        digits.append(tmp % 4); tmp //= 4
    digits = digits[::-1]
    P = _paulis_1q[digits[0]]
    lab = _labels_1q[digits[0]]
    for d in digits[1:]:
        P = np.kron(P, _paulis_1q[d])
        lab = lab + _labels_1q[d]
    full_paulis.append(P)
    full_labels.append(lab)

identity_label  = 'I' * N_QUBITS
pauli_idx_nonid = [i for i, l in enumerate(full_labels) if l != identity_label]
pauli_idx_ident = [i for i, l in enumerate(full_labels) if l == identity_label][0]
full_paulis_arr = np.stack(full_paulis, axis=0)
pauli_torch     = torch.tensor(full_paulis_arr, dtype=torch.complex64, device=device_torch)
nonid_idx_torch = torch.tensor(pauli_idx_nonid, dtype=torch.long, device=device_torch)

# ============================================================================
# State Generation (same as exp1 / tomography.py)
# ============================================================================
from scipy.linalg import sqrtm
import scipy.linalg

def validate_density_matrix(rho, tol=1e-6):
    rho = 0.5 * (rho + rho.conj().T)
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0)
    rho = (eigvecs * eigvals) @ eigvecs.conj().T
    rho /= np.trace(rho)
    return rho

def compute_purity(rho):
    return float(np.real(np.trace(rho @ rho)))

def random_haar_statevector(n_q):
    d = 2 ** n_q
    v = np.random.randn(d) + 1j * np.random.randn(d)
    return v / np.linalg.norm(v)

def rho_to_pauli_exps(rho):
    return np.array([np.real(np.trace(rho @ full_paulis_arr[k]))
                     for k in pauli_idx_nonid], dtype=np.float32)

def fidelity_numpy(rho_true, rho_pred):
    sq = sqrtm(rho_true)
    if np.any(np.isnan(sq)):
        sq = scipy.linalg.sqrtm(rho_true + 1e-12 * np.eye(dim))
    M = sq @ rho_pred @ sq
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.maximum(eigvals.real, 0)
    return float(np.sum(np.sqrt(eigvals)) ** 2)

def purity_to_depol_param(target_purity, n_q):
    d = 2 ** n_q
    from scipy.optimize import brentq
    def _f(p):
        rho0 = np.zeros((d, d), dtype=complex); rho0[0, 0] = 1.0
        rho = (1 - p) * rho0 + p * np.eye(d, dtype=complex) / d
        return compute_purity(rho) - target_purity
    try:
        return brentq(_f, 0.0, 1.0)
    except ValueError:
        return 0.5

def _amplitude_damped(gamma):
    psi = random_haar_statevector(N_QUBITS)
    rho = np.outer(psi, psi.conj())
    K0_1q = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1_1q = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    result = np.zeros((dim, dim), dtype=complex)
    for bits in range(2 ** N_QUBITS):
        K = np.array([[1]], dtype=complex)
        for q in range(N_QUBITS):
            K = np.kron(K, K1_1q if (bits >> q) & 1 else K0_1q)
        result += K @ rho @ K.conj().T
    return validate_density_matrix(result)

def _phase_damped(gamma):
    psi = random_haar_statevector(N_QUBITS)
    rho = np.outer(psi, psi.conj())
    K0_1q = np.sqrt(1 - gamma) * np.eye(2, dtype=complex)
    K1_1q = np.sqrt(gamma) * np.array([[1, 0], [0, 0]], dtype=complex)
    K2_1q = np.sqrt(gamma) * np.array([[0, 0], [0, 1]], dtype=complex)
    kraus_1q = [K0_1q, K1_1q, K2_1q]
    n_k = len(kraus_1q)
    result = np.zeros((dim, dim), dtype=complex)
    for bits in range(n_k ** N_QUBITS):
        K = np.array([[1]], dtype=complex); tmp = bits
        for q in range(N_QUBITS):
            K = np.kron(K, kraus_1q[tmp % n_k]); tmp //= n_k
        result += K @ rho @ K.conj().T
    return validate_density_matrix(result)

def _thermal(beta):
    H = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H = (H + H.conj().T) / 2
    eigvals, eigvecs = np.linalg.eigh(H)
    rho = eigvecs @ np.diag(np.exp(-beta * eigvals)) @ eigvecs.conj().T
    rho /= np.trace(rho)
    return validate_density_matrix(rho)

def _make_generators(min_p, max_p):
    def _sp(): return np.random.uniform(min_p, max_p)
    def depolarized():
        p = purity_to_depol_param(_sp(), N_QUBITS)
        psi = random_haar_statevector(N_QUBITS); rho = np.outer(psi, psi.conj())
        return validate_density_matrix((1-p)*rho + p*np.eye(dim, dtype=complex)/dim)
    def werner():
        tp = _sp(); d = dim
        a = 1+1/d; bw = -2/d; cw = 1/d - tp
        disc = bw**2 - 4*a*cw
        p = np.clip((-bw + np.sqrt(max(disc, 0)))/(2*a), 0, 1) if disc >= 0 else 0.5
        psi = np.zeros(dim, dtype=complex); psi[0] = psi[-1] = 1/np.sqrt(2)
        pr = np.outer(psi, psi.conj())
        return validate_density_matrix(p*pr + (1-p)*np.eye(dim, dtype=complex)/dim)
    def isotropic():
        p = purity_to_depol_param(_sp(), N_QUBITS)
        psi = random_haar_statevector(N_QUBITS); pr = np.outer(psi, psi.conj())
        return validate_density_matrix((1-p)*pr + p*np.eye(dim, dtype=complex)/dim)
    def _bisect(fn, target, lo, hi, flipped=False, iters=20):
        for _ in range(iters):
            mid = (lo + hi) / 2
            pur = compute_purity(fn(mid))
            if (pur > target) ^ flipped: lo = mid
            else: hi = mid
        return fn((lo + hi) / 2)
    def amp_damped(): return _bisect(_amplitude_damped, _sp(), 0.0, 0.99)
    def phase_damped(): return _bisect(_phase_damped, _sp(), 0.0, 0.99)
    def thermal(): return _bisect(_thermal, _sp(), 0.01, 10.0, flipped=True)
    def separable():
        tp = _sp(); pq = tp ** (1.0 / N_QUBITS); rho = None
        for _ in range(N_QUBITS):
            p_this = float(np.clip(np.random.uniform(pq-0.03, pq+0.03), 0.5, 1.0))
            p_dep = purity_to_depol_param(p_this, 1)
            psi_q = random_haar_statevector(1); rq = np.outer(psi_q, psi_q.conj())
            rq = (1-p_dep)*rq + p_dep*np.eye(2, dtype=complex)/2
            rho = rq if rho is None else np.kron(rho, rq)
        return validate_density_matrix(rho)
    return [('Depolarized', depolarized), ('Werner', werner),
            ('Isotropic', isotropic), ('AmpDamped', amp_damped),
            ('PhaseDamped', phase_damped), ('Thermal', thermal),
            ('Separable', separable)]

def _gen_one(gen_fn, min_p, max_p, max_tries=120):
    for _ in range(max_tries):
        try:
            rho = gen_fn(); p = compute_purity(rho)
            if min_p - 0.01 <= p <= max_p + 0.01:
                return rho_to_pauli_exps(rho), rho.astype(np.complex64)
        except Exception: pass
    return None

def build_dataset(n_samples, min_purity=0.85, max_purity=0.95):
    generators = _make_generators(min_purity, max_purity)
    n_per_type = n_samples // len(generators)
    remainder = n_samples % len(generators)
    X_list, Rho_list = [], []
    for gi, (name, fn) in enumerate(generators):
        n_this = n_per_type + (1 if gi < remainder else 0)
        over = n_this * 4
        print(f"  {name}: {n_this} ...", end='', flush=True)
        if HAS_JOBLIB and N_JOBS_DATA > 1:
            results = Parallel(n_jobs=N_JOBS_DATA)(
                delayed(_gen_one)(fn, min_purity, max_purity) for _ in range(over))
        else:
            results = [_gen_one(fn, min_purity, max_purity) for _ in range(over)]
        valid = [r for r in results if r is not None][:n_this]
        for exps, rho in valid:
            X_list.append(exps); Rho_list.append(rho)
        print(f" got {len(valid)}")
    return np.stack(X_list), np.stack(Rho_list)


# ============================================================================
# Model Classes (parameterized by latent_dim)
# ============================================================================

class TorchDMCircuit(nn.Module):
    """Batched GPU density-matrix circuit simulator (3-qubit)."""
    def __init__(self, nq, nl, na, nn_p, paulis_arr, pidx):
        super().__init__()
        self.n_qubits = nq; self.n_layers = nl
        self.n_angles = na; self.n_noise = nn_p
        self.dim = 2 ** nq
        P_nonid = paulis_arr[pidx]
        self.register_buffer('paulis', torch.tensor(P_nonid, dtype=torch.complex64))
        rho0 = np.zeros((self.dim, self.dim), dtype=np.complex64); rho0[0, 0] = 1.0
        self.register_buffer('rho0', torch.tensor(rho0, dtype=torch.complex64))
        # CNOT gates
        cnots = []
        for q in range(nq - 1):
            cnots.append(self._build_cnot(q, nq))
        if cnots:
            self.register_buffer('cnot_gates',
                torch.tensor(np.stack(cnots), dtype=torch.complex64))
        else:
            self.cnot_gates = None
        X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
        self.register_buffer('pauli_x', torch.tensor(X, dtype=torch.complex64))
        self.register_buffer('pauli_y', torch.tensor(Y, dtype=torch.complex64))
        self.register_buffer('pauli_z', torch.tensor(Z, dtype=torch.complex64))

    @staticmethod
    def _build_cnot(ctrl, nq):
        I2 = np.eye(2, dtype=np.complex64)
        P0 = np.array([[1, 0], [0, 0]], dtype=np.complex64)
        P1 = np.array([[0, 0], [0, 1]], dtype=np.complex64)
        Xg = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        term0 = np.array([[1]], dtype=np.complex64)
        term1 = np.array([[1]], dtype=np.complex64)
        for q in range(nq):
            if q == ctrl:
                term0 = np.kron(term0, P0); term1 = np.kron(term1, P1)
            elif q == ctrl + 1:
                term0 = np.kron(term0, I2); term1 = np.kron(term1, Xg)
            else:
                term0 = np.kron(term0, I2); term1 = np.kron(term1, I2)
        return term0 + term1

    def _rx(self, theta):
        c = torch.cos(theta / 2).unsqueeze(-1).unsqueeze(-1)
        s = torch.sin(theta / 2).unsqueeze(-1).unsqueeze(-1)
        return torch.cat([torch.cat([c, -1j*s], dim=-1),
                          torch.cat([-1j*s, c], dim=-1)], dim=-2).to(torch.complex64)

    def _ry(self, theta):
        c = torch.cos(theta / 2).unsqueeze(-1).unsqueeze(-1)
        s = torch.sin(theta / 2).unsqueeze(-1).unsqueeze(-1)
        return torch.cat([torch.cat([c, -s], dim=-1),
                          torch.cat([s,  c], dim=-1)], dim=-2).to(torch.complex64)

    def _rz(self, theta):
        ep = torch.exp(1j * theta / 2).unsqueeze(-1).unsqueeze(-1)
        em = torch.exp(-1j * theta / 2).unsqueeze(-1).unsqueeze(-1)
        z = torch.zeros_like(ep)
        return torch.cat([torch.cat([em, z], dim=-1),
                          torch.cat([z, ep], dim=-1)], dim=-2).to(torch.complex64)

    def _kron_batch(self, A, B):
        N, m, n = A.shape; _, p, q = B.shape
        return torch.einsum('bij,bkl->bikjl', A, B).reshape(N, m * p, n * q)

    def _kron_1q(self, G, qubit, nq):
        B = G.shape[0]
        result = G if qubit == 0 else torch.eye(
            2**qubit, dtype=torch.complex64, device=G.device).unsqueeze(0).expand(B,-1,-1)
        if qubit > 0:
            result = self._kron_batch(result, G)
        remaining = nq - qubit - 1
        if remaining > 0:
            I_rest = torch.eye(2**remaining, dtype=torch.complex64,
                               device=G.device).unsqueeze(0).expand(B,-1,-1)
            result = self._kron_batch(result, I_rest)
        return result

    def _depol_1q(self, rho, p, q):
        nq = self.n_qubits; I2 = torch.eye(2, dtype=torch.complex64, device=rho.device)
        def _full_pauli(P_1q):
            parts = [P_1q if j == q else I2 for j in range(nq)]
            result = parts[0]
            for j in range(1, nq): result = torch.kron(result, parts[j])
            return result.unsqueeze(0)
        Px = _full_pauli(self.pauli_x); Py = _full_pauli(self.pauli_y)
        Pz = _full_pauli(self.pauli_z)
        p_3 = p.view(-1, 1, 1) / 3.0
        one_minus_p = (1.0 - p).view(-1, 1, 1)
        return one_minus_p * rho + p_3 * (
            Px @ rho @ Px.conj().transpose(-2,-1) +
            Py @ rho @ Py.conj().transpose(-2,-1) +
            Pz @ rho @ Pz.conj().transpose(-2,-1))

    def forward(self, params):
        B = params.shape[0]
        angles = params[:, :self.n_angles]; noise = params[:, self.n_angles:]
        rho = self.rho0.unsqueeze(0).expand(B, -1, -1).clone()
        idx = 0
        for layer in range(self.n_layers):
            U_layer = None
            for q in range(self.n_qubits):
                rx = self._rx(angles[:, idx]); ry = self._ry(angles[:, idx+1])
                rz = self._rz(angles[:, idx+2]); U_q = rz @ ry @ rx; idx += 3
                U_q_full = self._kron_1q(U_q, q, self.n_qubits)
                U_layer = U_q_full if U_layer is None else U_q_full @ U_layer
            rho = U_layer @ rho @ U_layer.conj().transpose(-2, -1)
            if self.cnot_gates is not None:
                for ci in range(self.cnot_gates.shape[0]):
                    C = self.cnot_gates[ci].unsqueeze(0)
                    rho = C @ rho @ C.conj().transpose(-2, -1)
            p_layer = noise[:, layer]
            for q in range(self.n_qubits):
                rho = self._depol_1q(rho, p_layer, q)
        return torch.einsum('bij,pji->bp', rho, self.paulis).real


class Encoder(nn.Module):
    def __init__(self, input_dim, ldim):
        super().__init__()
        self.latent_geom = ldim - 2; self.latent_pur = 2
        h1 = max(128, 2 * input_dim); h2 = max(64, h1 // 2)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, h1), nn.LayerNorm(h1), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(h1, h2),        nn.LayerNorm(h2), nn.ReLU(), nn.Dropout(0.15))
        self.head_geom = nn.Linear(h2, self.latent_geom)
        self.head_pur  = nn.Linear(h2, self.latent_pur)

    def forward(self, x):
        h = self.trunk(x)
        z_geom = self.head_geom(h); z_pur = torch.sigmoid(self.head_pur(h))
        return torch.cat([z_geom, z_pur], dim=-1), z_geom, z_pur


class HybridAE(nn.Module):
    def __init__(self, input_dim, ldim, n_ang, n_noise, paulis_arr, pidx):
        super().__init__()
        self.encoder = Encoder(input_dim, ldim)
        self.latent2angles = nn.Linear(ldim, n_ang)
        self.latent2noise  = nn.Linear(ldim, n_noise)
        self.circuit = TorchDMCircuit(N_QUBITS, n_layers, n_ang, n_noise, paulis_arr, pidx)
        nn.init.constant_(self.latent2noise.bias, 0.0)

    def forward(self, x):
        z, z_geom, z_pur = self.encoder(x)
        angles = self.latent2angles(z)
        noise_p = torch.sigmoid(self.latent2noise(z)) * (NOISE_MAX - NOISE_MIN) + NOISE_MIN
        params = torch.cat([angles, noise_p], dim=-1)
        return self.circuit(params), z, z_geom, z_pur


# ============================================================================
# GPU Loss Functions
# ============================================================================

def pauli_exps_to_rho_torch(exps_nonid):
    B = exps_nonid.shape[0]
    exps_full = torch.zeros(B, 4**N_QUBITS, dtype=torch.float32, device=exps_nonid.device)
    exps_full[:, pauli_idx_ident] = 1.0
    exps_full[:, nonid_idx_torch] = exps_nonid
    exps_c = exps_full.to(torch.complex64).unsqueeze(-1).unsqueeze(-1)
    rho = (exps_c * pauli_torch.unsqueeze(0)).sum(dim=1) * norm_factor
    return 0.5 * (rho + rho.conj().permute(0, 2, 1))

def sqrtm_psd_torch(A, eps=1e-12):
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag_embed(torch.sqrt(eigvals)) @ eigvecs.conj().transpose(-2,-1)

def uhlmann_fidelity_batch(rho_a, rho_b, eps=1e-12):
    sqrt_a = sqrtm_psd_torch(rho_a, eps)
    M = sqrt_a @ rho_b @ sqrt_a
    eigvals = torch.linalg.eigvalsh(M)
    eigvals = torch.clamp(eigvals.real, min=0.0)
    return torch.clamp(torch.sum(torch.sqrt(eigvals), dim=-1) ** 2, 0.0, 1.0)

def metric_preservation_loss(z_geom, rho_batch, n_pairs=50):
    B = z_geom.shape[0]
    if B < 2: return torch.tensor(0.0, device=z_geom.device)
    n_pairs = min(n_pairs, B * (B - 1) // 2)
    idx = torch.randint(0, B, (n_pairs, 2), device=z_geom.device)
    mask = idx[:, 0] != idx[:, 1]; idx = idx[mask]
    if len(idx) < 2: return torch.tensor(0.0, device=z_geom.device)
    d_latent = torch.norm(z_geom[idx[:, 0]] - z_geom[idx[:, 1]], dim=-1)
    F_pairs = uhlmann_fidelity_batch(rho_batch[idx[:, 0]], rho_batch[idx[:, 1]])
    d_bures = torch.acos(torch.clamp(torch.sqrt(torch.clamp(F_pairs, 1e-8, 1.0)), -1+1e-7, 1-1e-7))
    valid = d_bures > 1e-6; d_latent = d_latent[valid]; d_bures = d_bures[valid]
    if len(d_latent) < 2: return torch.tensor(0.0, device=z_geom.device)
    dl_c = d_latent - d_latent.mean(); db_c = d_bures - d_bures.mean()
    cov = (dl_c * db_c).mean()
    return 1.0 - cov / (torch.sqrt((dl_c**2).mean() + 1e-8) * torch.sqrt((db_c**2).mean() + 1e-8))

def evaluate_fidelity(model, X_np, Rho_np):
    model.eval(); fids = []
    with torch.no_grad():
        for i in range(0, len(X_np), 256):
            xb = torch.tensor(X_np[i:i+256], dtype=torch.float32, device=device_torch)
            rho_true = torch.tensor(Rho_np[i:i+256], dtype=torch.complex64, device=device_torch)
            preds_b, *_ = model(xb)
            rho_pred = pauli_exps_to_rho_torch(preds_b)
            F_b = uhlmann_fidelity_batch(rho_pred, rho_true)
            fids.extend(F_b.cpu().tolist())
    fids = np.array(fids)
    return float(np.mean(fids)), float(np.median(fids)), fids


# ============================================================================
# Single Run Function
# ============================================================================

def train_one_config(ldim, X_train_np, Rho_train_np, X_val_np, Rho_val_np):
    print(f"\n{'='*60}")
    print(f"LATENT DIM = {ldim} (3-qubit)")
    print(f"{'='*60}")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    model = HybridAE(n_pauli_nonid, ldim, n_angles, n_noise_params,
                      full_paulis_arr, pauli_idx_nonid).to(device_torch)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, min_lr=1e-5)

    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device_torch)
    Rho_train = torch.tensor(Rho_train_np, dtype=torch.complex64, device=device_torch)
    train_loader = DataLoader(TensorDataset(X_train, Rho_train),
                              batch_size=batch_size, shuffle=True)

    best_val_fid = 0.0; best_state = None
    patience_baseline = 0.0; no_improve_evals = 0
    t0 = time.time(); final_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train(); run_loss = 0.0; iters = 0
        for x_batch, rho_true_batch in train_loader:
            if model.training:
                x_batch = torch.clamp(x_batch + 0.005 * torch.randn_like(x_batch), -1, 1)
            optimizer.zero_grad()
            preds, z, z_geom, z_pur = model(x_batch)
            rho_pred = pauli_exps_to_rho_torch(preds)
            F_batch = uhlmann_fidelity_batch(rho_pred, rho_true_batch)
            loss_recon = 1.0 - F_batch.mean()
            purity_pred = torch.einsum('bij,bji->b', rho_pred, rho_pred).real
            purity_true = torch.einsum('bij,bji->b', rho_true_batch, rho_true_batch).real
            loss_purity = torch.mean(torch.abs(purity_pred - purity_true))
            purity_true_norm = torch.clamp((purity_true - 0.85) / 0.10, 0.0, 1.0)
            loss_purity_enc = torch.mean((z_pur.mean(dim=-1) - purity_true_norm) ** 2)
            loss_metric = metric_preservation_loss(z_geom, rho_true_batch, n_pairs=50)
            loss = (loss_recon + lambda_purity * loss_purity
                    + lambda_metric * loss_metric + lambda_purity_enc * loss_purity_enc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            run_loss += loss.item(); iters += 1

        if epoch % eval_every == 0 or epoch == 1:
            val_mean, _, _ = evaluate_fidelity(model, X_val_np, Rho_val_np)
            scheduler.step(1.0 - val_mean)
            if epoch % 50 == 0:
                print(f"  Ep {epoch:03d} | val F {val_mean:.4f} | {time.time()-t0:.0f}s")
            if val_mean > best_val_fid:
                best_val_fid = val_mean
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if val_mean > patience_baseline + 1e-3:
                patience_baseline = val_mean; no_improve_evals = 0
            else:
                no_improve_evals += 1
            if no_improve_evals >= early_stop_patience:
                print(f"  Early stop at epoch {epoch}"); final_epoch = epoch; break

    if final_epoch == 0: final_epoch = max_epochs
    if best_state: model.load_state_dict(best_state); model.to(device_torch)

    train_time = time.time() - t0
    val_mean, val_median, all_fids = evaluate_fidelity(model, X_val_np, Rho_val_np)

    # Geometric analysis
    model.eval(); _enc = model.encoder
    with torch.no_grad():
        X_all = torch.cat([
            torch.tensor(X_train_np, dtype=torch.float32, device=device_torch),
            torch.tensor(X_val_np, dtype=torch.float32, device=device_torch)], dim=0)
        Rho_all_np = np.concatenate([Rho_train_np, Rho_val_np], axis=0)
        z_chunks = [_enc(X_all[i:i+256])[0].cpu().numpy() for i in range(0, len(X_all), 256)]
        Z = np.vstack(z_chunks)

    # Pearson/Spearman
    np.random.seed(SEED)
    n_pairs = min(600, len(Z) * (len(Z) - 1) // 2)
    pair_idx = np.array([np.random.choice(len(Z), 2, replace=False) for _ in range(n_pairs)])
    ii, jj = pair_idx[:, 0], pair_idx[:, 1]
    geo_dists = np.linalg.norm(Z[ii] - Z[jj], axis=1)
    Ri = torch.tensor(Rho_all_np[ii], dtype=torch.complex64, device=device_torch)
    Rj = torch.tensor(Rho_all_np[jj], dtype=torch.complex64, device=device_torch)
    F_chunks = []
    for k in range(0, len(ii), 50):
        F_chunks.append(uhlmann_fidelity_batch(Ri[k:k+50], Rj[k:k+50]).cpu().numpy())
    F_vals = np.concatenate(F_chunks)
    bures_dists = np.arccos(np.sqrt(np.clip(F_vals, 0.0, 1.0)))
    valid_mask = bures_dists > 1e-6
    geo_v = geo_dists[valid_mask]; bures_v = bures_dists[valid_mask]
    pr, _ = pearsonr(geo_v, bures_v)
    sr, _ = spearmanr(geo_v, bures_v)

    # PCA
    pca_full = PCA().fit(Z)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    dim_95 = int(np.argmax(cumvar >= 0.95)) + 1

    result = {
        'latent_dim': ldim, 'n_params': n_params, 'final_epoch': final_epoch,
        'train_time_s': train_time,
        'mean_fidelity': val_mean, 'median_fidelity': val_median,
        'pearson_r': float(pr), 'spearman_rho': float(sr),
        'pca_dim_95': dim_95,
    }
    print(f"  Result: F={val_mean:.4f}, r={pr:.4f}, rho={sr:.4f}, PCA95={dim_95}")

    del model, X_all, Ri, Rj; torch.cuda.empty_cache()
    return result


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    random.seed(42); np.random.seed(42)
    print("Generating dataset...")
    X_train_np, Rho_train_np = build_dataset(n_train)
    X_val_np, Rho_val_np = build_dataset(n_val)
    print(f"Dataset: train={X_train_np.shape}, val={X_val_np.shape}")

    all_results = []
    for ldim in LATENT_DIMS:
        result = train_one_config(ldim, X_train_np, Rho_train_np, X_val_np, Rho_val_np)
        all_results.append(result)
        outfile = os.path.join(OUTDIR, 'latent_dim_ablation_3q.json')
        with open(outfile, 'w') as f:
            json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("LATENT DIM ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'d_z':>5} | {'F':>7} | {'r':>7} | {'rho':>7} | {'PCA95':>5} | {'Epoch':>5} | {'Params':>8}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['latent_dim']:5d} | {r['mean_fidelity']:.4f} | {r['pearson_r']:.4f} | "
              f"{r['spearman_rho']:.4f} | {r['pca_dim_95']:5d} | {r['final_epoch']:5d} | {r['n_params']:8,}")
    print(f"\nSaved to: {outfile}")
