"""Model, data and training code of the original experiments (generated file).

Do not edit by hand: run revision/build_nbcore.py instead.

Sections 1-5 are copied verbatim from cells 1, 2, 3, 4 and 8 of
reviewer_experiments/run_all_experiments.ipynb (identical in
run_4q_experiments.ipynb). The only edit is that OUTDIR can be overridden with
the QST_OUTDIR environment variable. The final section adds seeded data
generation, model loading and history logging without changing how a model is
built or trained.

Cell md5 (first 10 hex digits): cell1=d9c6b19732, cell2=3a0e73cb34, cell3=2e3180f443, cell4=4df68e622f, cell8=5fc23687fa
"""

# =============================================================================
# SECTION 1: notebook cell 1 (verbatim)
# =============================================================================
# ============================================================================
# Cell 1: Imports & Global Setup
# ============================================================================
import os, sys, time, json, random, itertools, pickle
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import sqrtm
import scipy.linalg
from scipy.optimize import brentq

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

OUTDIR = os.environ.get("QST_OUTDIR", "/kaggle/working")
os.makedirs(OUTDIR, exist_ok=True)
device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_JOBS_DATA = 4

# Noise bounds (fixed across all experiments)
NOISE_MAX = 0.15
NOISE_MIN = 0.0

print(f"Device: {device_torch}")
print(f"Output: {OUTDIR}")
timings = {}  # track wall time per experiment

# =============================================================================
# SECTION 2: notebook cell 2 (verbatim)
# =============================================================================
# ============================================================================
# Cell 2: Pauli Basis Generator (parameterized by n_qubits)
# ============================================================================

def setup_pauli_basis(n_qubits):
    """Build Pauli basis arrays and return a config dict.
    Must be called before using any model/eval code for a given n_qubits.
    Sets global variables used by model classes and loss functions.
    """
    global N_QUBITS, dim, n_pauli_nonid, n_layers, n_angles, n_noise_params
    global norm_factor, full_paulis_arr, pauli_idx_nonid, pauli_idx_ident
    global full_labels, pauli_torch, nonid_idx_torch

    N_QUBITS = n_qubits
    dim = 2 ** n_qubits
    n_pauli_nonid = 4 ** n_qubits - 1
    n_layers = n_qubits + 2
    n_angles = n_qubits * 3 * n_layers
    n_noise_params = n_layers
    norm_factor = 1.0 / dim

    I_m = np.array([[1, 0], [0, 1]], dtype=complex)
    X_m = np.array([[0, 1], [1, 0]], dtype=complex)
    Y_m = np.array([[0,-1j],[1j, 0]], dtype=complex)
    Z_m = np.array([[1, 0], [0,-1]], dtype=complex)
    _paulis_1q = [I_m, X_m, Y_m, Z_m]
    _labels_1q = ['I', 'X', 'Y', 'Z']

    full_paulis, full_labels_local = [], []
    for index in range(4 ** n_qubits):
        digits, tmp = [], index
        for _ in range(n_qubits):
            digits.append(tmp % 4); tmp //= 4
        digits = digits[::-1]
        P = _paulis_1q[digits[0]]; lab = _labels_1q[digits[0]]
        for d in digits[1:]:
            P = np.kron(P, _paulis_1q[d]); lab = lab + _labels_1q[d]
        full_paulis.append(P); full_labels_local.append(lab)

    full_labels = full_labels_local
    identity_label = 'I' * n_qubits
    pauli_idx_nonid = [i for i, l in enumerate(full_labels) if l != identity_label]
    pauli_idx_ident = [i for i, l in enumerate(full_labels) if l == identity_label][0]
    full_paulis_arr = np.stack(full_paulis, axis=0)
    pauli_torch = torch.tensor(full_paulis_arr, dtype=torch.complex64, device=device_torch)
    nonid_idx_torch = torch.tensor(pauli_idx_nonid, dtype=torch.long, device=device_torch)

    print(f"Pauli basis: {n_qubits}q, dim={dim}, n_pauli_nonid={n_pauli_nonid}, "
          f"n_layers={n_layers}, n_angles={n_angles}")

print("setup_pauli_basis() defined.")

# =============================================================================
# SECTION 3: notebook cell 3 (verbatim)
# =============================================================================
# ============================================================================
# Cell 3: TorchDMCircuit + Encoder + HybridAE (shared model classes)
# ============================================================================

class TorchDMCircuit(nn.Module):
    """Batched GPU density-matrix circuit simulator."""
    def __init__(self, nq, nl, na, nn_p, paulis_arr, pidx):
        super().__init__()
        self.n_qubits = nq; self.n_layers = nl; self.n_angles = na; self.n_noise = nn_p
        self.dim = 2 ** nq
        self.register_buffer('paulis', torch.tensor(paulis_arr[pidx], dtype=torch.complex64))
        rho0 = np.zeros((self.dim, self.dim), dtype=np.complex64); rho0[0, 0] = 1.0
        self.register_buffer('rho0', torch.tensor(rho0, dtype=torch.complex64))
        cnots = [self._build_cnot(q, nq) for q in range(nq - 1)]
        if cnots:
            self.register_buffer('cnot_gates', torch.tensor(np.stack(cnots), dtype=torch.complex64))
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
        P0 = np.array([[1,0],[0,0]], dtype=np.complex64)
        P1 = np.array([[0,0],[0,1]], dtype=np.complex64)
        Xg = np.array([[0,1],[1,0]], dtype=np.complex64)
        t0 = np.array([[1]], dtype=np.complex64); t1 = np.array([[1]], dtype=np.complex64)
        for q in range(nq):
            if q == ctrl: t0 = np.kron(t0, P0); t1 = np.kron(t1, P1)
            elif q == ctrl+1: t0 = np.kron(t0, I2); t1 = np.kron(t1, Xg)
            else: t0 = np.kron(t0, I2); t1 = np.kron(t1, I2)
        return t0 + t1

    def _rx(self, theta):
        c = torch.cos(theta/2).unsqueeze(-1).unsqueeze(-1)
        s = torch.sin(theta/2).unsqueeze(-1).unsqueeze(-1)
        return torch.cat([torch.cat([c, -1j*s], dim=-1),
                          torch.cat([-1j*s, c], dim=-1)], dim=-2).to(torch.complex64)
    def _ry(self, theta):
        c = torch.cos(theta/2).unsqueeze(-1).unsqueeze(-1)
        s = torch.sin(theta/2).unsqueeze(-1).unsqueeze(-1)
        return torch.cat([torch.cat([c, -s], dim=-1),
                          torch.cat([s,  c], dim=-1)], dim=-2).to(torch.complex64)
    def _rz(self, theta):
        ep = torch.exp(1j*theta/2).unsqueeze(-1).unsqueeze(-1)
        em = torch.exp(-1j*theta/2).unsqueeze(-1).unsqueeze(-1)
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
        if qubit > 0: result = self._kron_batch(result, G)
        remaining = nq - qubit - 1
        if remaining > 0:
            result = self._kron_batch(result, torch.eye(
                2**remaining, dtype=torch.complex64, device=G.device).unsqueeze(0).expand(B,-1,-1))
        return result

    def _depol_1q(self, rho, p, q):
        nq = self.n_qubits; I2 = torch.eye(2, dtype=torch.complex64, device=rho.device)
        def _fp(P):
            parts = [P if j == q else I2 for j in range(nq)]
            r = parts[0]
            for j in range(1, nq): r = torch.kron(r, parts[j])
            return r.unsqueeze(0)
        Px = _fp(self.pauli_x); Py = _fp(self.pauli_y); Pz = _fp(self.pauli_z)
        p3 = p.view(-1,1,1)/3.0; omp = (1.0-p).view(-1,1,1)
        return omp*rho + p3*(Px@rho@Px.conj().transpose(-2,-1) +
                              Py@rho@Py.conj().transpose(-2,-1) +
                              Pz@rho@Pz.conj().transpose(-2,-1))

    def forward(self, params):
        B = params.shape[0]; angles = params[:, :self.n_angles]; noise = params[:, self.n_angles:]
        rho = self.rho0.unsqueeze(0).expand(B,-1,-1).clone(); idx = 0
        for layer in range(self.n_layers):
            U_layer = None
            for q in range(self.n_qubits):
                U_q = self._rz(angles[:,idx+2]) @ self._ry(angles[:,idx+1]) @ self._rx(angles[:,idx])
                idx += 3; U_q_full = self._kron_1q(U_q, q, self.n_qubits)
                U_layer = U_q_full if U_layer is None else U_q_full @ U_layer
            rho = U_layer @ rho @ U_layer.conj().transpose(-2,-1)
            if self.cnot_gates is not None:
                for ci in range(self.cnot_gates.shape[0]):
                    C = self.cnot_gates[ci].unsqueeze(0)
                    rho = C @ rho @ C.conj().transpose(-2,-1)
            for q in range(self.n_qubits):
                rho = self._depol_1q(rho, noise[:, layer], q)
        return torch.einsum('bij,pji->bp', rho, self.paulis).real

    def forward_rho(self, params):
        """Return final density matrix (not expectations)."""
        B = params.shape[0]; angles = params[:, :self.n_angles]; noise = params[:, self.n_angles:]
        rho = self.rho0.unsqueeze(0).expand(B,-1,-1).clone(); idx = 0
        for layer in range(self.n_layers):
            U_layer = None
            for q in range(self.n_qubits):
                U_q = self._rz(angles[:,idx+2]) @ self._ry(angles[:,idx+1]) @ self._rx(angles[:,idx])
                idx += 3; U_q_full = self._kron_1q(U_q, q, self.n_qubits)
                U_layer = U_q_full if U_layer is None else U_q_full @ U_layer
            rho = U_layer @ rho @ U_layer.conj().transpose(-2,-1)
            if self.cnot_gates is not None:
                for ci in range(self.cnot_gates.shape[0]):
                    C = self.cnot_gates[ci].unsqueeze(0)
                    rho = C @ rho @ C.conj().transpose(-2,-1)
            for q in range(self.n_qubits):
                rho = self._depol_1q(rho, noise[:, layer], q)
        return rho


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim_):
        super().__init__()
        self.latent_geom = latent_dim_ - 2; self.latent_pur = 2
        h1 = max(128, 2 * input_dim); h2 = max(64, h1 // 2)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, h1), nn.LayerNorm(h1), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(h1, h2),        nn.LayerNorm(h2), nn.ReLU(), nn.Dropout(0.15))
        self.head_geom = nn.Linear(h2, self.latent_geom)
        self.head_pur  = nn.Linear(h2, self.latent_pur)
    def forward(self, x):
        h = self.trunk(x); z_geom = self.head_geom(h); z_pur = torch.sigmoid(self.head_pur(h))
        return torch.cat([z_geom, z_pur], dim=-1), z_geom, z_pur


class HybridAE(nn.Module):
    def __init__(self, input_dim, latent_dim_, n_ang, n_noise, paulis_arr, pidx):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim_)
        self.latent2angles = nn.Linear(latent_dim_, n_ang)
        self.latent2noise  = nn.Linear(latent_dim_, n_noise)
        self.circuit = TorchDMCircuit(N_QUBITS, n_layers, n_ang, n_noise, paulis_arr, pidx)
        self.n_ang = n_ang; self.n_noise = n_noise
        nn.init.constant_(self.latent2noise.bias, 0.0)
    def forward(self, x):
        z, z_geom, z_pur = self.encoder(x)
        angles = self.latent2angles(z)
        noise_p = torch.sigmoid(self.latent2noise(z)) * (NOISE_MAX - NOISE_MIN) + NOISE_MIN
        return self.circuit(torch.cat([angles, noise_p], dim=-1)), z, z_geom, z_pur
    @torch.no_grad()
    def get_noise_params(self, x):
        z, _, _ = self.encoder(x)
        return torch.sigmoid(self.latent2noise(z)) * (NOISE_MAX - NOISE_MIN) + NOISE_MIN

print("Model classes defined: TorchDMCircuit, Encoder, HybridAE")

# =============================================================================
# SECTION 4: notebook cell 4 (verbatim)
# =============================================================================
# ============================================================================
# Cell 4: State Generation + Dataset + Evaluation Utilities
# ============================================================================

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
    d = 2 ** n_q; v = np.random.randn(d) + 1j * np.random.randn(d)
    return v / np.linalg.norm(v)

def rho_to_pauli_exps(rho):
    return np.array([np.real(np.trace(rho @ full_paulis_arr[k]))
                     for k in pauli_idx_nonid], dtype=np.float32)

def pauli_exps_to_rho(exps_nonid):
    rho = np.eye(dim, dtype=complex) / dim
    for k, idx in enumerate(pauli_idx_nonid):
        rho += exps_nonid[k] * full_paulis_arr[idx] / dim
    return 0.5 * (rho + rho.conj().T)

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
    def _f(p):
        rho0 = np.zeros((d, d), dtype=complex); rho0[0, 0] = 1.0
        rho = (1 - p) * rho0 + p * np.eye(d, dtype=complex) / d
        return compute_purity(rho) - target_purity
    try: return brentq(_f, 0.0, 1.0)
    except ValueError: return 0.5

# â”€â”€ Channel generators (use global N_QUBITS, dim) â”€â”€

def _amplitude_damped(gamma):
    psi = random_haar_statevector(N_QUBITS); rho = np.outer(psi, psi.conj())
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
    psi = random_haar_statevector(N_QUBITS); rho = np.outer(psi, psi.conj())
    K0 = np.sqrt(1 - gamma) * np.eye(2, dtype=complex)
    K1 = np.sqrt(gamma) * np.array([[1, 0], [0, 0]], dtype=complex)
    K2 = np.sqrt(gamma) * np.array([[0, 0], [0, 1]], dtype=complex)
    kraus = [K0, K1, K2]; nk = len(kraus)
    result = np.zeros((dim, dim), dtype=complex)
    for bits in range(nk ** N_QUBITS):
        K = np.array([[1]], dtype=complex); tmp = bits
        for q in range(N_QUBITS):
            K = np.kron(K, kraus[tmp % nk]); tmp //= nk
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
    def amp_damped(): return _bisect(_amplitude_damped, _sp(), 0.0, 0.99, flipped=False)
    def phase_damped(): return _bisect(_phase_damped, _sp(), 0.0, 0.99, flipped=False)
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
    return [('Depolarized', depolarized), ('Werner', werner), ('Isotropic', isotropic),
            ('AmpDamped', amp_damped), ('PhaseDamped', phase_damped),
            ('Thermal', thermal), ('Separable', separable)]

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
    n_per_type = n_samples // len(generators); remainder = n_samples % len(generators)
    X_list, Rho_list = [], []
    for gi, (name, fn) in enumerate(generators):
        n_this = n_per_type + (1 if gi < remainder else 0); over = n_this * 4
        print(f"  {name}: requesting {n_this} ...", end='', flush=True)
        if HAS_JOBLIB and N_JOBS_DATA > 1:
            results = Parallel(n_jobs=N_JOBS_DATA)(
                delayed(_gen_one)(fn, min_purity, max_purity) for _ in range(over))
        else:
            results = [_gen_one(fn, min_purity, max_purity) for _ in range(over)]
        valid = [r for r in results if r is not None][:n_this]
        if len(valid) < n_this: print(f" WARNING only {len(valid)}/{n_this}!", end='')
        for exps, rho in valid: X_list.append(exps); Rho_list.append(rho)
        print(f" got {len(valid)}")
    X = np.stack(X_list); Rhos = np.stack(Rho_list)
    purs = [compute_purity(r) for r in Rhos]
    print(f"  Total {X.shape[0]} | purity [{min(purs):.3f}, {max(purs):.3f}] mean={np.mean(purs):.3f}")
    return X, Rhos

# â”€â”€ GPU loss & eval functions â”€â”€

def pauli_exps_to_rho_torch(exps_nonid):
    B = exps_nonid.shape[0]
    exps_full = torch.zeros(B, 4**N_QUBITS, dtype=torch.float32, device=exps_nonid.device)
    exps_full[:, pauli_idx_ident] = 1.0; exps_full[:, nonid_idx_torch] = exps_nonid
    exps_c = exps_full.to(torch.complex64).unsqueeze(-1).unsqueeze(-1)
    rho = (exps_c * pauli_torch.unsqueeze(0)).sum(dim=1) * norm_factor
    return 0.5 * (rho + rho.conj().permute(0, 2, 1))

def sqrtm_psd_torch(A, eps=1e-12):
    A = 0.5 * (A + A.conj().transpose(-1, -2))
    A = A + eps * torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    evals, evecs = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=0.0)
    return (evecs * torch.sqrt(evals).unsqueeze(-2)) @ evecs.conj().transpose(-1, -2)

def uhlmann_fidelity_batch(rho_a, rho_b, eps=1e-12):
    sqrt_a = sqrtm_psd_torch(rho_a, eps); M = sqrt_a @ rho_b @ sqrt_a
    eigvals = torch.clamp(torch.linalg.eigvalsh(M).real, min=0.0)
    return torch.clamp(torch.sum(torch.sqrt(eigvals), dim=-1)**2, 0.0, 1.0)

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
    return 1.0 - cov / (torch.sqrt((dl_c**2).mean()+1e-8) * torch.sqrt((db_c**2).mean()+1e-8))

def evaluate_fidelity(model, X_np, Rho_np, bs=256):
    model.eval(); fids = []
    with torch.no_grad():
        for i in range(0, len(X_np), bs):
            xb = torch.tensor(X_np[i:i+bs], dtype=torch.float32, device=device_torch)
            rho_true = torch.tensor(Rho_np[i:i+bs], dtype=torch.complex64, device=device_torch)
            preds_b, *_ = model(xb)
            rho_pred = pauli_exps_to_rho_torch(preds_b)
            fids.extend(uhlmann_fidelity_batch(rho_pred, rho_true).cpu().tolist())
    fids = np.array(fids)
    return float(np.mean(fids)), float(np.median(fids)), fids

def evaluate_shot_noise(model, X_np, Rho_np, shots_count, n_eval=120):
    model.eval()
    idx = np.random.choice(len(X_np), min(n_eval, len(X_np)), replace=False)
    with torch.no_grad():
        xb = torch.tensor(X_np[idx], dtype=torch.float32, device=device_torch)
        preds_np = model(xb)[0].cpu().numpy()
    fids = []
    for pred_analyt, true_idx in zip(preds_np, idx):
        variance = (1.0 - pred_analyt ** 2) / shots_count
        noise_v = np.random.randn(len(pred_analyt)) * np.sqrt(np.maximum(variance, 0.0))
        pred_noisy = np.clip(pred_analyt + noise_v, -1.0, 1.0)
        rho_pred = pauli_exps_to_rho(pred_noisy); rho_true = Rho_np[true_idx]
        fids.append(fidelity_numpy(rho_true, rho_pred))
    return float(np.mean(fids)), float(np.std(fids))

print("All utility functions defined.")

# =============================================================================
# SECTION 5: notebook cell 8 (verbatim)
# =============================================================================
# ============================================================================
# Experiment 1: Seed Variance â€” Generic Training + Evaluation
# ============================================================================

def train_one_seed(seed_val, X_train_np, Rho_train_np, X_val_np, Rho_val_np,
                   latent_dim, batch_size, max_epochs=600, lr=3e-3,
                   lambda_metric=0.20, lambda_purity=2.0, lambda_purity_enc=0.10,
                   eval_every=5, early_stop_patience=15, n_geodesic_pairs=600):
    """Train model with a specific seed. Returns (results_dict, best_state_dict)."""
    print(f"\n{'='*60}")
    print(f"SEED {seed_val} -- {N_QUBITS}-qubit training")
    print(f"{'='*60}")

    random.seed(seed_val); np.random.seed(seed_val); torch.manual_seed(seed_val)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_val)

    model = HybridAE(n_pauli_nonid, latent_dim, n_angles, n_noise_params,
                      full_paulis_arr, pauli_idx_nonid).to(device_torch)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, min_lr=1e-5)

    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device_torch)
    Rho_train = torch.tensor(Rho_train_np, dtype=torch.complex64, device=device_torch)
    train_loader = DataLoader(TensorDataset(X_train, Rho_train),
                              batch_size=batch_size, shuffle=True, drop_last=False)

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
            optimizer.step(); run_loss += loss.item(); iters += 1

        if epoch % eval_every == 0 or epoch == 1:
            val_mean, _, _ = evaluate_fidelity(model, X_val_np, Rho_val_np)
            scheduler.step(1.0 - val_mean)
            if epoch % (eval_every * 10) == 0 or epoch <= eval_every:
                print(f"  Ep {epoch:03d} | val F {val_mean:.4f} | {time.time()-t0:.0f}s")
            if val_mean > best_val_fid:
                best_val_fid = val_mean
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if val_mean > patience_baseline + 1e-3:
                patience_baseline = val_mean; no_improve_evals = 0
            else: no_improve_evals += 1
            if no_improve_evals >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch}"); final_epoch = epoch; break

    if final_epoch == 0: final_epoch = max_epochs
    if best_state is not None: model.load_state_dict(best_state)
    model.to(device_torch)
    train_time = time.time() - t0

    # Full evaluation
    val_mean, val_median, all_fids = evaluate_fidelity(model, X_val_np, Rho_val_np)

    # Geometric analysis
    model.eval(); _enc = model.encoder
    with torch.no_grad():
        X_all = torch.cat([torch.tensor(X_train_np, dtype=torch.float32, device=device_torch),
                           torch.tensor(X_val_np, dtype=torch.float32, device=device_torch)], dim=0)
        Rho_all_np = np.concatenate([Rho_train_np, Rho_val_np], axis=0)
        z_chunks = [_enc(X_all[i:i+batch_size])[0].cpu().numpy() for i in range(0, len(X_all), batch_size)]
        Z = np.vstack(z_chunks)

    np.random.seed(seed_val)
    pair_idx = np.array([np.random.choice(len(Z), 2, replace=False) for _ in range(n_geodesic_pairs)])
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
    pr, _ = pearsonr(geo_v, bures_v); sr, _ = spearmanr(geo_v, bures_v)
    coeffs = np.polyfit(bures_v, geo_v, 1)
    r_sq = 1 - np.sum((geo_v - (coeffs[0]*bures_v + coeffs[1]))**2) / np.sum((geo_v - geo_v.mean())**2)

    # PCA
    pca_full = PCA().fit(Z)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    dim_95 = int(np.argmax(cumvar >= 0.95)) + 1

    # Curvature
    k_curv = min(25, Z.shape[0] // 4)
    nbrs = NearestNeighbors(n_neighbors=k_curv+1).fit(Z); _, idxs = nbrs.kneighbors(Z)
    kappas = []
    for i in range(Z.shape[0]):
        nb = Z[idxs[i, 1:]] - Z[i]
        if nb.shape[0] > 1:
            try:
                _, s, _ = np.linalg.svd(nb, full_matrices=False); s = s[s > 1e-10]
                if len(s) > 1: kappas.append(s[-1] / s[0])
            except Exception: pass
    mean_curv = float(np.mean(kappas)) if kappas else 0.0

    # Shot noise
    shot_results = {}
    for sv in [10_000, 1_000, 100]:
        np.random.seed(seed_val + sv)
        sm, ss = evaluate_shot_noise(model, X_val_np, Rho_val_np, sv, n_eval=120)
        shot_results[sv] = (sm, ss)

    results = {
        'seed': seed_val, 'n_qubits': N_QUBITS, 'final_epoch': final_epoch,
        'train_time_s': train_time, 'mean_fidelity': val_mean, 'median_fidelity': val_median,
        'std_fidelity': float(np.std(all_fids)), 'pearson_r': float(pr),
        'spearman_rho': float(sr), 'r_squared': float(r_sq),
        'pca_dim_95': dim_95, 'mean_curvature': mean_curv,
        'shot_10k_mean': shot_results[10000][0], 'shot_10k_std': shot_results[10000][1],
        'shot_1k_mean': shot_results[1000][0], 'shot_1k_std': shot_results[1000][1],
        'shot_100_mean': shot_results[100][0], 'shot_100_std': shot_results[100][1],
    }
    print(f"  F={val_mean:.4f}, r={pr:.4f}, PCA95={dim_95}, curv={mean_curv:.5f}")

    del model, X_all, Ri, Rj; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return results, best_state

print("train_one_seed() defined.")

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
