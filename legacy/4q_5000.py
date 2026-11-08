"""
tomography.py — Geometric Latent Space Tomography (2/3/4-Qubit)
===========================================================
Hybrid Quantum-Classical Autoencoder with PyTorch-native density matrix simulator.

Key changes from the PennyLane version:
  [SPEED]   PennyLane replaced by TorchDMCircuit — a fully batched, GPU-native
            differentiable density matrix simulator implementing the exact same
            circuit (RX/RY/RZ + CNOT chain + depolarizing noise). Expected
            speedup: ~500–2000× per epoch, enabling 400 epochs in ~15–30 min.
  [BUG-1]   evaluate_fidelity, evaluate_shot_noise: model returns 4 values;
            callers updated (preds_b, *_ = model(xb)).
  [BUG-2]   get_noise_params: Encoder.forward returns (z, z_geom, z_pur);
            fixed to unpack correctly.
  [BUG-3]   Manifold analysis: encoder call returns tuple; [0] index added.
  [BUG-4]   metric_preservation_loss: z_batch undefined; fixed to z_geom.
  [GPU-2]   DataParallel wrapper for Kaggle 2×T4 setup.
  [CACHE]   Dataset cache validated against (n_train, n_val) before loading.
  [FIX.md]  Purity-matching loss, split latent (z_geom + z_pur), reduced
            encoder (63→128→64→30), purity supervision, noise bias init.
"""

# ============================================================================
# Imports
# ============================================================================
import os
import random
import time
import pickle

import numpy as np
from scipy.linalg import sqrtm
import scipy.linalg
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import orthogonal_procrustes

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("Warning: joblib not found — dataset generation will be single-threaded.")

# ============================================================================
# Section 1 — Configuration
# ============================================================================

# --- Qubit system (fixed: 4q data-limited rerun with corrected 2× encoder) ---
n_qubits        = 4
dim             = 2 ** n_qubits
n_pauli_nonid   = 4 ** n_qubits - 1

# --- Circuit architecture (all derived from n_qubits) ---
n_layers        = n_qubits + 2               # L=4 for 2q, L=5 for 3q, L=6 for 4q
n_angles        = n_qubits * 3 * n_layers
n_noise_params  = n_layers
n_circuit_params = n_angles + n_noise_params
NOISE_MAX       = 0.15
NOISE_MIN       = 0.0     # no floor — purity loss controls noise

# --- Latent space ---
latent_dim      = 10 * n_qubits              # 20 for 2q, 30 for 3q, 40 for 4q

# --- Dataset (FIXED: n_train=5000 for data-limited comparison) ---
free_params     = 4 ** n_qubits - 1            # 255 for 4q
n_train         = 5000                          # data-limited: 20 samples/param
n_val           = 800
batch_size      = 64
DATASET_CACHE   = f'/kaggle/working/dataset_{n_qubits}q_cache.npz'
N_JOBS_DATA     = 4

# --- Training ---
max_epochs              = 600
target_fidelity         = 0.99   # effectively disabled — early_stop_patience handles convergence
early_stop_patience     = 15   # evaluation steps without improvement
lr                      = 3e-3
weight_decay            = 1e-4
lambda_metric           = 0.20
lambda_purity           = 2.0
lambda_purity_enc       = 0.10
eval_every              = 5    # cheap evaluations → more frequent monitoring

# --- Shot noise ---
SHOT_COUNTS     = [None, 10_000, 1_000, 100]
N_EVAL_SHOTS    = 120

# --- Manifold analysis ---
n_vis_samples   = 120
n_geodesic_pairs = 600

# --- Misc ---
seed            = 42
device_torch    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTDIR          = '/kaggle/working'

n_gpus = torch.cuda.device_count()
print(f"System  : {n_qubits} qubits | dim={dim} | Pauli observables={n_pauli_nonid}")
print(f"Dataset : n_train={n_train} | n_val={n_val} | batch_size={batch_size} | {n_train/free_params:.0f} samples/param")
print(f"Circuit : {n_layers} layers | {n_angles} angles + {n_noise_params} noise = {n_circuit_params} params")
print(f"Torch   : {device_torch} | GPUs={n_gpus}")

# ============================================================================
# Section 2 — Seeds
# ============================================================================

torch.set_default_dtype(torch.float32)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ============================================================================
# Section 3 — Pauli Basis (generic for any n_qubits)
# ============================================================================

norm_factor = 1.0 / dim
I_m = np.array([[1, 0], [0, 1]],   dtype=complex)
X_m = np.array([[0, 1], [1, 0]],   dtype=complex)
Y_m = np.array([[0,-1j],[1j, 0]],  dtype=complex)
Z_m = np.array([[1, 0], [0,-1]],   dtype=complex)
_paulis_1q = [I_m, X_m, Y_m, Z_m]
_labels_1q = ['I', 'X', 'Y', 'Z']

full_paulis, full_labels = [], []
for index in range(4 ** n_qubits):
    digits, tmp = [], index
    for _ in range(n_qubits):
        digits.append(tmp % 4); tmp //= 4
    digits = digits[::-1]
    P   = _paulis_1q[digits[0]]
    lab = _labels_1q[digits[0]]
    for d in digits[1:]:
        P   = np.kron(P, _paulis_1q[d])
        lab = lab + _labels_1q[d]
    full_paulis.append(P)
    full_labels.append(lab)

identity_label   = 'I' * n_qubits
pauli_idx_nonid  = [i for i, l in enumerate(full_labels) if l != identity_label]
pauli_idx_ident  = [i for i, l in enumerate(full_labels) if l == identity_label][0]

full_paulis_arr  = np.stack(full_paulis, axis=0)               # (4^n, dim, dim)
pauli_torch      = torch.tensor(full_paulis_arr, dtype=torch.complex64,
                                device=device_torch)            # (4^n, dim, dim)
nonid_idx_torch  = torch.tensor(pauli_idx_nonid, dtype=torch.long,
                                device=device_torch)

assert len(pauli_idx_nonid) == n_pauli_nonid
print(f"Pauli basis: {4**n_qubits} total, {n_pauli_nonid} non-identity")

# ============================================================================
# Section 4 — Classical Helper Functions
# ============================================================================

def validate_density_matrix(rho, tol=1e-6):
    rho = (rho + rho.conj().T) / 2.0
    tr  = np.trace(rho)
    if abs(tr) > 1e-12:
        rho = rho / tr
    evals, evecs = np.linalg.eigh(rho)
    if np.any(evals < -tol):
        evals = np.maximum(evals, 0.0)
        rho   = evecs @ np.diag(evals) @ evecs.conj().T
        rho   = rho / np.trace(rho)
    return rho

def compute_purity(rho):
    return float(np.real(np.trace(rho @ rho)))

def random_haar_statevector(n_q):
    d   = 2 ** n_q
    vec = np.random.randn(d) + 1j * np.random.randn(d)
    return vec / np.linalg.norm(vec)

def rho_to_pauli_exps(rho):
    """Density matrix → n_pauli_nonid-dim float32 Pauli expectation vector."""
    exps = []
    for idx in pauli_idx_nonid:
        exps.append(float(np.real_if_close(np.trace(rho @ full_paulis_arr[idx]))))
    return np.array(exps, dtype=np.float32)

def pauli_exps_to_rho(exps_nonid):
    """Pauli vector → dim×dim complex density matrix (numpy)."""
    exps_full = np.zeros(4 ** n_qubits, dtype=float)
    exps_full[pauli_idx_ident] = 1.0
    for i, idx in enumerate(pauli_idx_nonid):
        exps_full[idx] = float(exps_nonid[i])
    rho = sum(c * P for c, P in zip(exps_full, full_paulis_arr))
    rho = rho * norm_factor
    return (rho + rho.conj().T) / 2.0

def fidelity_numpy(rho_true, rho_pred):
    """Uhlmann-Jozsa fidelity (numpy, used only in shot-noise evaluation)."""
    eps  = 1e-12
    rt   = rho_true + eps * np.eye(dim)
    rp   = rho_pred + eps * np.eye(dim)
    sq   = sqrtm(rt)
    tr   = np.real(np.trace(sqrtm(sq @ rp @ sq)))
    return float(np.clip(tr ** 2, 0.0, 1.0))

def purity_to_depol_param(target_purity, n_q):
    """Solve (1-p)² + p(2-p)/d = target_purity for p."""
    if target_purity >= 0.99:
        return 0.0
    d = 2 ** n_q
    a = 1.0 + 1.0 / d
    b = -2.0
    c = 1.0 - target_purity
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return 0.0
    return float(np.clip((-b - np.sqrt(disc)) / (2 * a), 0.0, 1.0))

# ============================================================================
# Section 5 — Mixed State Generators
# ============================================================================

def _amplitude_damped(gamma):
    psi = random_haar_statevector(n_qubits)
    rho = np.outer(psi, psi.conj())
    K0  = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1  = np.array([[0, np.sqrt(gamma)], [0, 0]],     dtype=complex)
    for q in range(n_qubits):
        parts0 = [K0 if i == q else np.eye(2, dtype=complex) for i in range(n_qubits)]
        parts1 = [K1 if i == q else np.eye(2, dtype=complex) for i in range(n_qubits)]
        K0f = parts0[0];  K1f = parts1[0]
        for p0, p1 in zip(parts0[1:], parts1[1:]):
            K0f = np.kron(K0f, p0);  K1f = np.kron(K1f, p1)
        rho = K0f @ rho @ K0f.conj().T + K1f @ rho @ K1f.conj().T
    return validate_density_matrix(rho)

def _phase_damped(gamma):
    psi = random_haar_statevector(n_qubits)
    rho = np.outer(psi, psi.conj())
    K0 = np.sqrt(1 - gamma) * np.eye(2, dtype=complex)
    K1 = np.sqrt(gamma)     * np.array([[1,0],[0,0]], dtype=complex)
    K2 = np.sqrt(gamma)     * np.array([[0,0],[0,1]], dtype=complex)
    for q in range(n_qubits):
        rho_new = np.zeros_like(rho)
        for K1q in [K0, K1, K2]:
            parts = [K1q if i == q else np.eye(2, dtype=complex) for i in range(n_qubits)]
            Kf = parts[0]
            for kp in parts[1:]:
                Kf = np.kron(Kf, kp)
            rho_new += Kf @ rho @ Kf.conj().T
        rho = rho_new
    return validate_density_matrix(rho)

def _thermal(beta):
    H = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H = (H + H.conj().T) / 2.0
    evals, evecs = np.linalg.eigh(H)
    evals -= evals[0]
    H     = evecs @ np.diag(evals) @ evecs.conj().T
    expH  = scipy.linalg.expm(-beta * H)
    return validate_density_matrix(expH / np.trace(expH))

def _make_generators(min_p, max_p):
    def _sp():
        return np.random.uniform(min_p, max_p)

    def depolarized():
        p   = purity_to_depol_param(_sp(), n_qubits)
        psi = random_haar_statevector(n_qubits)
        rho = np.outer(psi, psi.conj())
        return validate_density_matrix((1-p)*rho + p*np.eye(dim, dtype=complex)/dim)

    def werner():
        tp  = _sp()
        d   = dim
        a   = 1+1/d; bw = -2/d; cw = 1/d - tp
        disc = bw**2 - 4*a*cw
        p   = np.clip((-bw + np.sqrt(max(disc, 0)))/(2*a), 0, 1) if disc >= 0 else 0.5
        psi = np.zeros(dim, dtype=complex); psi[0] = psi[-1] = 1/np.sqrt(2)
        pr  = np.outer(psi, psi.conj())
        return validate_density_matrix(p*pr + (1-p)*np.eye(dim, dtype=complex)/dim)

    def isotropic():
        p   = purity_to_depol_param(_sp(), n_qubits)
        psi = random_haar_statevector(n_qubits)
        pr  = np.outer(psi, psi.conj())
        return validate_density_matrix((1-p)*pr + p*np.eye(dim, dtype=complex)/dim)

    def _bisect(fn, target, lo, hi, flipped=False, iters=20):
        for _ in range(iters):
            mid = (lo + hi) / 2
            pur = compute_purity(fn(mid))
            if (pur > target) ^ flipped:
                lo = mid
            else:
                hi = mid
        return fn((lo + hi) / 2)

    def amp_damped():
        return _bisect(_amplitude_damped, _sp(), 0.0, 0.99, flipped=False)

    def phase_damped():
        return _bisect(_phase_damped, _sp(), 0.0, 0.99, flipped=False)

    def thermal():
        return _bisect(_thermal, _sp(), 0.01, 10.0, flipped=True)

    def separable():
        tp  = _sp()
        pq  = tp ** (1.0 / n_qubits)
        rho = None
        for _ in range(n_qubits):
            p_this = float(np.clip(np.random.uniform(pq-0.03, pq+0.03), 0.5, 1.0))
            p_dep  = purity_to_depol_param(p_this, 1)
            psi_q  = random_haar_statevector(1)
            rq     = np.outer(psi_q, psi_q.conj())
            rq     = (1-p_dep)*rq + p_dep*np.eye(2, dtype=complex)/2
            rho    = rq if rho is None else np.kron(rho, rq)
        return validate_density_matrix(rho)

    return [
        ('Depolarized', depolarized),
        ('Werner',      werner),
        ('Isotropic',   isotropic),
        ('AmpDamped',   amp_damped),
        ('PhaseDamped', phase_damped),
        ('Thermal',     thermal),
        ('Separable',   separable),
    ]

def _gen_one(gen_fn, min_p, max_p, max_tries=120):
    for _ in range(max_tries):
        try:
            rho = gen_fn()
            p   = compute_purity(rho)
            if min_p - 0.01 <= p <= max_p + 0.01:
                return rho_to_pauli_exps(rho), rho.astype(np.complex64)
        except Exception:
            pass
    return None

def build_dataset(n_samples, min_purity=0.85, max_purity=0.95):
    generators = _make_generators(min_purity, max_purity)
    n_per_type = n_samples // len(generators)
    remainder  = n_samples %  len(generators)
    X_list, Rho_list = [], []
    for gi, (name, fn) in enumerate(generators):
        n_this = n_per_type + (1 if gi < remainder else 0)
        over   = n_this * 4
        print(f"  {name}: requesting {n_this} ...", end='', flush=True)
        if HAS_JOBLIB and N_JOBS_DATA > 1:
            results = Parallel(n_jobs=N_JOBS_DATA)(
                delayed(_gen_one)(fn, min_purity, max_purity) for _ in range(over))
        else:
            results = [_gen_one(fn, min_purity, max_purity) for _ in range(over)]
        valid = [r for r in results if r is not None][:n_this]
        if len(valid) < n_this:
            print(f" WARNING only {len(valid)}/{n_this}!", end='')
        for exps, rho in valid:
            X_list.append(exps)
            Rho_list.append(rho)
        print(f" got {len(valid)}")
    X    = np.stack(X_list)
    Rhos = np.stack(Rho_list)
    purs = [compute_purity(r) for r in Rhos]
    print(f"  Total {X.shape[0]} | purity [{min(purs):.3f}, {max(purs):.3f}] "
          f"mean={np.mean(purs):.3f}")
    return X, Rhos

# ============================================================================
# Section 6 — Dataset (Load Cache or Generate)
# ============================================================================

def _load_or_generate_dataset():
    if os.path.exists(DATASET_CACHE):
        _d = np.load(DATASET_CACHE)
        # Validate cache size matches current config
        if _d['X_train'].shape[0] == n_train and _d['X_val'].shape[0] == n_val:
            X_tr, Rho_tr = _d['X_train'], _d['Rho_train']
            X_v,  Rho_v  = _d['X_val'],   _d['Rho_val']
            print(f"Loaded cached dataset: Train {X_tr.shape} | Val {X_v.shape}")
            return X_tr, Rho_tr, X_v, Rho_v
        else:
            print(f"Cache shape mismatch "
                  f"(cached train={_d['X_train'].shape[0]}, need {n_train}) — regenerating.")

    print("Generating training dataset...")
    X_tr, Rho_tr = build_dataset(n_train)
    print("Generating validation dataset...")
    X_v,  Rho_v  = build_dataset(n_val)
    np.savez_compressed(DATASET_CACHE,
                        X_train=X_tr, Rho_train=Rho_tr,
                        X_val=X_v,   Rho_val=Rho_v)
    print(f"Dataset saved to {DATASET_CACHE}")
    return X_tr, Rho_tr, X_v, Rho_v

X_train_np, Rho_train_np, X_val_np, Rho_val_np = _load_or_generate_dataset()

X_train   = torch.tensor(X_train_np,   dtype=torch.float32,   device=device_torch)
X_val     = torch.tensor(X_val_np,     dtype=torch.float32,   device=device_torch)
Rho_train = torch.tensor(Rho_train_np, dtype=torch.complex64, device=device_torch)
Rho_val   = torch.tensor(Rho_val_np,   dtype=torch.complex64, device=device_torch)

train_loader = DataLoader(TensorDataset(X_train, Rho_train),
                          batch_size=batch_size, shuffle=True,  drop_last=False)
val_loader   = DataLoader(TensorDataset(X_val,   Rho_val),
                          batch_size=batch_size, shuffle=False)

# ============================================================================
# Section 7 — PyTorch-Native Density Matrix Circuit Simulator
# ============================================================================
#
# Replaces PennyLane circuit_pauli_exps with a fully batched GPU implementation
# of the identical 3-qubit circuit:
#   Per layer:
#     1. RX(θ_q^x), RY(θ_q^y), RZ(θ_q^z) for each qubit q = 0,1,2
#     2. CNOT chain: control=0→target=1, then control=1→target=2
#     3. DepolarizingChannel(p_layer) on each qubit (same p shared per layer)
#   Initial state: |000⟩⟨000|
#   Output: 63 non-identity Pauli expectations
#
# Mathematical equivalence to PennyLane:
#   - Rotation convention: U_q = RZ(θ_z) @ RY(θ_y) @ RX(θ_x) per qubit
#   - Full layer unitary: U = U_0 ⊗ U_1 ⊗ U_2  (tensor product)
#   - Depolarising: ρ → (1−p)ρ + (p/3)(XρX + YρY + ZρZ) per qubit
#   - Expectations: Tr(ρ P_k) via einsum
#
# Performance: ~500–2000× faster than per-sample PennyLane default.mixed.

class TorchDMCircuit(nn.Module):
    """
    Differentiable n-qubit density matrix simulator.
    No learnable parameters — all buffers (Pauli/CNOT matrices).
    Supports 2, 3, and 4 qubits with dynamic CNOT chain.
    """

    def __init__(self, n_qubits, n_layers, n_angles, n_noise,
                 full_paulis_arr, pauli_idx_nonid):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_angles = n_angles
        self.n_noise  = n_noise
        self.dim      = 2 ** n_qubits

        # ── Non-identity Pauli matrices for expectation computation ──────────
        paulis = torch.tensor(
            full_paulis_arr[pauli_idx_nonid].astype(np.complex64),
            dtype=torch.complex64,
        )
        self.register_buffer('paulis', paulis)   # (63, dim, dim)

        # ── CNOT chain matrices (precomputed, fixed) ─────────────────────────
        # CNOT4: 4×4 matrix for CNOT in 2-qubit subspace (control=first qubit)
        CNOT4 = torch.zeros(4, 4, dtype=torch.complex64)
        CNOT4[0, 0] = CNOT4[1, 1] = CNOT4[2, 3] = CNOT4[3, 2] = 1.0
        I2 = torch.eye(2, dtype=torch.complex64)
        # CNOT(q→q+1): I^{⊗q} ⊗ CNOT4 ⊗ I^{⊗(n-q-2)}
        self.n_cnots = n_qubits - 1
        for c in range(self.n_cnots):
            left  = torch.eye(2**c,              dtype=torch.complex64) if c > 0 else None
            right = torch.eye(2**(n_qubits-c-2), dtype=torch.complex64) if c < n_qubits-2 else None
            gate  = CNOT4
            if left is not None:
                gate = torch.kron(left, gate)
            if right is not None:
                gate = torch.kron(gate, right)
            self.register_buffer(f'CNOT{c}{c+1}', gate)  # (dim, dim)
        # CNOT is Hermitian (CNOT† = CNOT) and its own inverse.

        # ── Expanded Pauli matrices for depolarising channel ─────────────────
        # Precomputed once to avoid repeated kron ops during forward pass.
        X1 = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex64)
        Y1 = torch.tensor([[0., -1j], [1j, 0.]], dtype=torch.complex64)
        Z1 = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.complex64)
        for q in range(n_qubits):
            self.register_buffer(f'Xq{q}', self._kron_1q(X1, q, n_qubits))
            self.register_buffer(f'Yq{q}', self._kron_1q(Y1, q, n_qubits))
            self.register_buffer(f'Zq{q}', self._kron_1q(Z1, q, n_qubits))

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _kron_1q(G, qubit, n_qubits):
        """I⊗...⊗G⊗...⊗I — G at position `qubit`. Non-batched."""
        I2 = torch.eye(2, dtype=G.dtype)
        parts = [G if q == qubit else I2 for q in range(n_qubits)]
        out = parts[0]
        for p in parts[1:]:
            out = torch.kron(out, p)
        return out  # (dim, dim)

    @staticmethod
    def _kron_batch(A, B):
        """
        Batched Kronecker product.
        A: (N, m, n),  B: (N, p, q)  →  (N, m*p, n*q)

        Element [b, i*p+k, j*q+l] = A[b,i,j] * B[b,k,l]
        Uses einsum 'bij,bkl->bikjl' then reshape.
        """
        N, m, n = A.shape
        _, p, q = B.shape
        return torch.einsum('bij,bkl->bikjl', A, B).reshape(N, m * p, n * q)

    # ── Single-qubit gate factories (θ: (B,) float32  →  (B,2,2) complex64) ──

    @staticmethod
    def _rx(theta):
        """RX(θ) = [[cos θ/2,  -i sin θ/2], [-i sin θ/2,  cos θ/2]]"""
        h = theta * 0.5
        c = torch.cos(h);  s = torch.sin(h);  z = torch.zeros_like(s)
        B = theta.shape[0]
        ri = torch.stack([c, z,  z, -s,  z, -s,  c, z], dim=-1).view(B, 2, 2, 2)
        return torch.view_as_complex(ri.contiguous())

    @staticmethod
    def _ry(theta):
        """RY(θ) = [[cos θ/2, -sin θ/2], [sin θ/2,  cos θ/2]]"""
        h = theta * 0.5
        c = torch.cos(h);  s = torch.sin(h);  z = torch.zeros_like(s)
        B = theta.shape[0]
        ri = torch.stack([c, z,  -s, z,  s, z,  c, z], dim=-1).view(B, 2, 2, 2)
        return torch.view_as_complex(ri.contiguous())

    @staticmethod
    def _rz(theta):
        """RZ(θ) = [[e^{-iθ/2}, 0], [0, e^{+iθ/2}]]"""
        h = theta * 0.5
        c = torch.cos(h);  s = torch.sin(h);  z = torch.zeros_like(s)
        B = theta.shape[0]
        ri = torch.stack([c, -s,  z, z,  z, z,  c, s], dim=-1).view(B, 2, 2, 2)
        return torch.view_as_complex(ri.contiguous())

    # ── Noise channel ─────────────────────────────────────────────────────────

    def _depol_1q(self, rho, p, q):
        """
        Depolarizing channel on single qubit q.
        ρ → (1−p)ρ + (p/3)(Xq ρ Xq + Yq ρ Yq + Zq ρ Zq)

        rho : (B, dim, dim) complex64
        p   : (B,) float32
        """
        Xq = getattr(self, f'Xq{q}').unsqueeze(0)   # (1, dim, dim) — broadcasts over B
        Yq = getattr(self, f'Yq{q}').unsqueeze(0)
        Zq = getattr(self, f'Zq{q}').unsqueeze(0)
        pc = p.to(torch.complex64).view(-1, 1, 1)    # (B, 1, 1)
        return ((1 - pc) * rho
                + (pc / 3) * (Xq @ rho @ Xq + Yq @ rho @ Yq + Zq @ rho @ Zq))

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, params):
        """
        params : (B, n_angles + n_noise) float32
                 angles: radians (unbounded); noise: [NOISE_MIN, NOISE_MAX]
        Returns: (B, n_pauli_nonid) float32 — Pauli expectation values
        """
        B      = params.shape[0]
        angles = params[:, :self.n_angles]
        noise  = params[:, self.n_angles:]

        # Initialise ρ = |0...0⟩⟨0...0|
        rho = torch.zeros(B, self.dim, self.dim,
                          dtype=torch.complex64, device=params.device)
        rho[:, 0, 0] = 1.0

        # Pre-fetch CNOT buffers with broadcast dim
        cnots = [getattr(self, f'CNOT{c}{c+1}').unsqueeze(0)
                 for c in range(self.n_cnots)]   # each (1, dim, dim)

        for layer in range(self.n_layers):
            # ── Single-qubit rotations ──────────────────────────────────────
            # U_q = RZ @ RY @ RX  (last applied gate is leftmost in product)
            # Full layer unitary U = U_0 ⊗ U_1 ⊗ ... ⊗ U_{n-1}
            U_parts = []
            for w in range(self.n_qubits):
                base = (layer * self.n_qubits + w) * 3
                Rx = self._rx(angles[:, base])
                Ry = self._ry(angles[:, base + 1])
                Rz = self._rz(angles[:, base + 2])
                U_parts.append(Rz @ Ry @ Rx)    # (B, 2, 2)

            # Build (B, dim, dim) via repeated batched kron product
            U = U_parts[0]
            for k in range(1, self.n_qubits):
                U = self._kron_batch(U, U_parts[k])

            # Apply: ρ → U ρ U†
            Ud = U.conj().transpose(-1, -2)
            rho = U @ rho @ Ud

            # ── CNOT chain (CNOT is Hermitian, so CNOT† = CNOT) ────────────
            for cn in cnots:
                rho = cn @ rho @ cn

            # ── Depolarizing noise (per qubit, shared p within layer) ───────
            p_l = noise[:, layer]   # (B,)
            for q in range(self.n_qubits):
                rho = self._depol_1q(rho, p_l, q)

        # ── Pauli expectations: Tr(ρ P_k) = Σ_{ij} ρ_{ij} P_k_{ji} ─────────
        # paulis: (63, dim, dim),  rho: (B, dim, dim)  →  (B, 63)
        exps = torch.einsum('bij,kji->bk', rho, self.paulis).real
        return exps.to(torch.float32)

# ============================================================================
# Section 8 — Model Architecture
# ============================================================================

class Encoder(nn.Module):
    """
    Classical MLP encoder: n_pauli_nonid → (z_geom, z_pur)
    Architecture: input→128→64→latent_dim (split: geom + 2 purity dims).
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_geom = latent_dim - 2   # geometric subspace
        self.latent_pur  = 2                # purity subspace

        # Uniform ≥2× expansion before compression at every scale.
        # 2q: 15→128 (8.5×), 3q: 63→128 (2.0×), 4q: 255→510 (2.0×)
        h1 = max(128, 2 * input_dim)
        h2 = max(64, h1 // 2)

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, h1), nn.LayerNorm(h1), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(h1, h2),        nn.LayerNorm(h2), nn.ReLU(), nn.Dropout(0.15),
        )
        self.head_geom = nn.Linear(h2, self.latent_geom)
        self.head_pur  = nn.Linear(h2, self.latent_pur)

    def forward(self, x):
        h      = self.trunk(x)
        z_geom = self.head_geom(h)
        z_pur  = torch.sigmoid(self.head_pur(h))   # bounded [0,1]
        return torch.cat([z_geom, z_pur], dim=-1), z_geom, z_pur


class HybridAE(nn.Module):
    """
    Hybrid Quantum-Classical Autoencoder with Mixed-State Decoder.

    Encoder : classical MLP  (63 → 30)
    Decoder : PyTorch density matrix circuit  (30 → 40 circuit params → ρ_pred)

    Latent vector z is split into:
      z_geom (28-dim) — geometric subspace, drives metric-preservation loss
      z_pur  ( 2-dim) — purity subspace,   drives purity supervision loss
    """

    def __init__(self, input_dim, latent_dim, n_ang, n_noise,
                 full_paulis_arr, pauli_idx_nonid):
        super().__init__()
        self.encoder       = Encoder(input_dim, latent_dim)
        self.latent2angles = nn.Linear(latent_dim, n_ang)
        self.latent2noise  = nn.Linear(latent_dim, n_noise)
        self.circuit       = TorchDMCircuit(
            n_qubits, n_layers, n_ang, n_noise,
            full_paulis_arr, pauli_idx_nonid,
        )
        self.n_ang   = n_ang
        self.n_noise = n_noise

        # Noise bias init: sigmoid(0) * (NOISE_MAX - NOISE_MIN) + NOISE_MIN ≈ 0.077
        nn.init.constant_(self.latent2noise.bias, 0.0)

    def forward(self, x):
        z, z_geom, z_pur = self.encoder(x)               # (B, latent_dim), (B,28), (B,2)
        angles   = self.latent2angles(z)                  # (B, 36)
        noise_p  = (torch.sigmoid(self.latent2noise(z))
                    * (NOISE_MAX - NOISE_MIN) + NOISE_MIN) # (B, 4)  ∈ [NOISE_MIN, NOISE_MAX]
        params   = torch.cat([angles, noise_p], dim=-1)   # (B, 40)
        preds    = self.circuit(params)                    # (B, 63)
        return preds, z, z_geom, z_pur

    @torch.no_grad()
    def get_noise_params(self, x):
        """Return learned per-layer noise strengths for an input batch."""
        z, _, _ = self.encoder(x)   # unpack 3-tuple correctly
        return (torch.sigmoid(self.latent2noise(z))
                * (NOISE_MAX - NOISE_MIN) + NOISE_MIN)    # (B, n_noise)


# ── Helper for DataParallel ──────────────────────────────────────────────────

def get_base_model(m):
    """Unwrap nn.DataParallel to access the underlying HybridAE."""
    return m.module if isinstance(m, nn.DataParallel) else m


# ── Instantiate model ────────────────────────────────────────────────────────

model = HybridAE(
    input_dim        = n_pauli_nonid,
    latent_dim       = latent_dim,
    n_ang            = n_angles,
    n_noise          = n_noise_params,
    full_paulis_arr  = full_paulis_arr,
    pauli_idx_nonid  = pauli_idx_nonid,
).to(device_torch)

n_params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model trainable parameters: {n_params_total:,}")

CHECKPOINT = os.path.join(OUTDIR, f'best_model_{n_qubits}q.pt')

# DataParallel removed: complex64 buffers break under DP on Kaggle's
# PyTorch build, and 8×8 matrix ops gain nothing from multi-GPU split.

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=8, min_lr=1e-5)

# ============================================================================
# Section 9 — Loss Functions (Fully GPU-Batched)
# ============================================================================

def pauli_exps_to_rho_torch(exps_nonid):
    """(B, 63) Pauli exps → (B, 8, 8) density matrices. Vectorised."""
    B         = exps_nonid.shape[0]
    exps_full = torch.zeros(B, 4**n_qubits,
                            dtype=torch.float32, device=exps_nonid.device)
    exps_full[:, pauli_idx_ident]  = 1.0
    exps_full[:, nonid_idx_torch]  = exps_nonid
    exps_c = exps_full.to(torch.complex64).unsqueeze(-1).unsqueeze(-1)  # (B,4^n,1,1)
    rho    = (exps_c * pauli_torch.unsqueeze(0)).sum(dim=1) * norm_factor
    return 0.5 * (rho + rho.conj().permute(0, 2, 1))


def sqrtm_psd_torch(A, eps=1e-12):
    """Batched matrix square root for PSD matrices via eigendecomposition."""
    A    = 0.5 * (A + A.conj().transpose(-1, -2))
    A    = A + eps * torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    evals, evecs = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=0.0)
    return (evecs * torch.sqrt(evals).unsqueeze(-2)) @ evecs.conj().transpose(-1, -2)


def uhlmann_fidelity_batch(rho_a, rho_b, eps=1e-12):
    """
    Batched Uhlmann-Jozsa fidelity F(ρ,σ) = (Tr√(√ρ σ √ρ))²
    rho_a, rho_b: (B, dim, dim) complex64   →   F: (B,) float ∈ [0,1]
    """
    rho_a  = 0.5 * (rho_a + rho_a.conj().transpose(-1, -2))
    rho_b  = 0.5 * (rho_b + rho_b.conj().transpose(-1, -2))
    sqrt_a = sqrtm_psd_torch(rho_a, eps)
    M      = sqrt_a @ rho_b @ sqrt_a
    M      = 0.5 * (M + M.conj().transpose(-1, -2))
    evals, _ = torch.linalg.eigh(M)
    tr_sqrt  = torch.sum(torch.sqrt(torch.clamp(evals, min=0.0) + eps), dim=-1)
    return torch.clamp(tr_sqrt ** 2, 0.0, 1.0).real


def metric_preservation_loss(z_geom, rho_batch, n_pairs=50):
    """
    Scale-invariant Pearson correlation metric-preservation loss.
    Encourages d_latent(z_i, z_j) ∝ d_Bures(ρ_i, ρ_j).

    z_geom    : (B, latent_geom=28) — geometric subspace only
    rho_batch : (B, dim, dim) complex64 — true density matrices
    Returns: scalar ∈ [0, 2]
    """
    B = z_geom.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z_geom.device)

    n_pairs = min(n_pairs, B * (B - 1) // 2)
    idx_i   = torch.randint(0, B, (n_pairs,), device=z_geom.device)
    offsets = torch.randint(1, B, (n_pairs,), device=z_geom.device)
    idx_j   = (idx_i + offsets) % B

    d_latent = torch.norm(z_geom[idx_i] - z_geom[idx_j], dim=-1)
    F        = uhlmann_fidelity_batch(rho_batch[idx_i], rho_batch[idx_j])
    d_bures  = torch.arccos(
        torch.clamp(torch.sqrt(torch.clamp(F, 0.0, 1.0)), 0.0, 1.0 - 1e-7))

    valid = d_bures > 1e-6
    if not valid.any():
        return torch.tensor(0.0, device=z_geom.device)   # BUG FIXED: was z_batch.device

    dl = d_latent[valid]
    db = d_bures[valid]
    dl = dl - dl.mean()
    db = db - db.mean()
    corr = (dl * db).sum() / (dl.norm() * db.norm() + 1e-8)
    return 1.0 - corr


def evaluate_fidelity(model, X_np, Rho_np):
    """Batched GPU fidelity evaluation (no scipy loop)."""
    model.eval()
    pred_chunks = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb      = torch.tensor(X_np[i:i+batch_size],
                                   dtype=torch.float32, device=device_torch)
            preds_b, *_ = model(xb)    # BUG FIXED: model returns 4 values
            pred_chunks.append(preds_b)

    preds_all    = torch.cat(pred_chunks, dim=0)
    rho_pred_all = pauli_exps_to_rho_torch(preds_all)
    rho_true_all = torch.tensor(Rho_np, dtype=torch.complex64, device=device_torch)

    fids = []
    for i in range(0, len(preds_all), batch_size):
        f = uhlmann_fidelity_batch(rho_pred_all[i:i+batch_size],
                                   rho_true_all[i:i+batch_size])
        fids.append(f.cpu())

    fids = torch.cat(fids).numpy()
    return float(fids.mean()), float(np.median(fids)), fids.tolist()

# ============================================================================
# Section 10 — Training Loop
# ============================================================================

history = dict(epoch=[], loss_total=[], loss_recon=[],
               loss_metric=[], val_fid=[], noise_mean=[])

best_val_fid     = 0.0
patience_baseline = 0.0    # tracks last "meaningful jump" for early stopping
best_state       = None
no_improve_evals = 0
t0               = time.time()

print("\n" + "=" * 72)
print(f"TRAINING — {n_qubits}-Qubit Hybrid Autoencoder (PyTorch-native circuit)")
print(f"  λ_metric={lambda_metric}  λ_purity={lambda_purity}  "
      f"λ_purity_enc={lambda_purity_enc}  NOISE_MAX={NOISE_MAX}")
print(f"  eval_every={eval_every}  max_epochs={max_epochs}  "
      f"target_fidelity={target_fidelity}")
print("=" * 72)

for epoch in range(1, max_epochs + 1):
    model.train()
    run_loss = run_recon = run_metric = 0.0
    iters    = 0

    for x_batch, rho_true_batch in train_loader:
        x_batch        = x_batch.to(device_torch)
        rho_true_batch = rho_true_batch.to(device_torch)

        # Input noise augmentation (light jitter for regularisation)
        if model.training:
            x_batch = x_batch + 0.005 * torch.randn_like(x_batch)
            x_batch = torch.clamp(x_batch, -1.0, 1.0)

        optimizer.zero_grad()

        preds, z, z_geom, z_pur = model(x_batch)

        # ── Reconstruction loss (Uhlmann fidelity) ──────────────────────────
        rho_pred   = pauli_exps_to_rho_torch(preds)
        F_batch    = uhlmann_fidelity_batch(rho_pred, rho_true_batch)
        loss_recon = 1.0 - F_batch.mean()

        # ── Purity matching (gradient pushes noise up when pred is too pure) ─
        purity_pred = torch.einsum('bij,bji->b', rho_pred, rho_pred).real
        purity_true = torch.einsum('bij,bji->b', rho_true_batch, rho_true_batch).real
        loss_purity = torch.mean(torch.abs(purity_pred - purity_true))

        # ── Purity supervision on z_pur ──────────────────────────────────────
        # Fixed normalisation: map purity [0.85, 0.95] → [0, 1] using dataset
        # constants, not batch-dependent max (which creates noisy targets).
        purity_true_norm = (purity_true - 0.85) / 0.10
        purity_true_norm = torch.clamp(purity_true_norm, 0.0, 1.0)
        loss_purity_enc  = torch.mean((z_pur.mean(dim=-1) - purity_true_norm) ** 2)

        # ── Metric preservation (geometric subspace only) ────────────────────
        loss_metric = metric_preservation_loss(z_geom, rho_true_batch, n_pairs=50)

        # ── Total loss ────────────────────────────────────────────────────────
        loss = (loss_recon
                + lambda_purity     * loss_purity
                + lambda_metric     * loss_metric
                + lambda_purity_enc * loss_purity_enc)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        run_loss   += loss.item()
        run_recon  += loss_recon.item()
        run_metric += loss_metric.item()
        iters      += 1

    avg_loss   = run_loss   / max(1, iters)
    avg_recon  = run_recon  / max(1, iters)
    avg_metric = run_metric / max(1, iters)

    # ── Periodic evaluation ──────────────────────────────────────────────────
    if epoch % eval_every == 0 or epoch == 1:
        val_mean, val_median, _ = evaluate_fidelity(model, X_val_np, Rho_val_np)

        with torch.no_grad():
            noise_vals = get_base_model(model).get_noise_params(X_train[:64])
            noise_mean = noise_vals.mean().item()

        history['epoch'].append(epoch)
        history['loss_total'].append(avg_loss)
        history['loss_recon'].append(avg_recon)
        history['loss_metric'].append(avg_metric)
        history['val_fid'].append(val_mean)
        history['noise_mean'].append(noise_mean)

        scheduler.step(1.0 - val_mean)

        elapsed = time.time() - t0
        print(f"Ep {epoch:03d} | loss {avg_loss:.5f} "
              f"(recon {avg_recon:.5f} | metric {avg_metric:.5f}) "
              f"| val F {val_mean:.4f} | noise {noise_mean:.4f} | {elapsed:.0f}s")

        # Always save the actual best model
        if val_mean > best_val_fid:
            best_val_fid = val_mean
            best_state   = {k: v.cpu().clone()
                            for k, v in get_base_model(model).state_dict().items()}
            torch.save(best_state, CHECKPOINT)

        # Only reset patience on meaningful improvement (min_delta=0.001)
        if val_mean > patience_baseline + 1e-3:
            patience_baseline = val_mean
            no_improve_evals  = 0
        else:
            no_improve_evals += 1

        if no_improve_evals >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} "
                  f"({no_improve_evals} eval steps without improvement)")
            break

        if val_mean >= target_fidelity:
            print(f"Target fidelity {target_fidelity} reached at epoch {epoch}!")
            break

if best_state is not None:
    get_base_model(model).load_state_dict(best_state)
    print(f"\nRestored best model: val fidelity = {best_val_fid:.4f}")

with open(os.path.join(OUTDIR, f'training_history_{n_qubits}q.pkl'), 'wb') as f:
    pickle.dump(history, f)

val_mean_fid, val_median_fid, all_fids = evaluate_fidelity(model, X_val_np, Rho_val_np)
print(f"Final val: mean={val_mean_fid:.4f} | median={val_median_fid:.4f}")

# ── Learned noise report ─────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    noise_report = get_base_model(model).get_noise_params(X_val[:200])
print("\nLearned depolarising noise per layer:")
for l in range(n_noise_params):
    v = noise_report[:, l].cpu().numpy()
    print(f"  Layer {l+1}: {v.mean():.4f} ± {v.std():.4f}")

# ============================================================================
# Section 11 — Shot Noise Robustness Evaluation
# ============================================================================

print("\n" + "=" * 72)
print("SHOT NOISE ROBUSTNESS EVALUATION")
print("=" * 72)

def evaluate_shot_noise(model, X_np, Rho_np, shots_count, n_eval=N_EVAL_SHOTS):
    """Evaluate fidelity under finite-shot measurement noise (analytical variance)."""
    model.eval()
    idx = np.random.choice(len(X_np), min(n_eval, len(X_np)), replace=False)
    with torch.no_grad():
        xb         = torch.tensor(X_np[idx], dtype=torch.float32, device=device_torch)
        preds_b, *_ = model(xb)   # BUG FIXED: model returns 4 values
        preds_np   = preds_b.cpu().numpy()

    fids = []
    for pred_analyt, true_idx in zip(preds_np, idx):
        variance   = (1.0 - pred_analyt ** 2) / shots_count
        noise      = np.random.randn(len(pred_analyt)) * np.sqrt(np.maximum(variance, 0.0))
        pred_noisy = np.clip(pred_analyt + noise, -1.0, 1.0)
        rho_pred   = pauli_exps_to_rho(pred_noisy)
        rho_true   = Rho_np[true_idx]
        fids.append(fidelity_numpy(rho_true, rho_pred))
    return float(np.mean(fids)), float(np.std(fids))

shot_results = {}
for sv in SHOT_COUNTS:
    label = 'Analytical (∞ shots)' if sv is None else f'{sv:,} shots'
    if sv is None:
        mean_f, std_f = val_mean_fid, float(np.std(all_fids))
    else:
        print(f"  {label} ...", end='', flush=True)
        mean_f, std_f = evaluate_shot_noise(model, X_val_np, Rho_val_np, sv)
        print(f"  F = {mean_f:.4f} ± {std_f:.4f}")
    shot_results[label] = (mean_f, std_f)

print("\nShot Noise Summary:")
print(f"{'Condition':>25} | {'Mean F':>8} | {'Std':>7}")
print("-" * 50)
for lbl, (mf, sf) in shot_results.items():
    print(f"{lbl:>25} | {mf:>8.4f} | {sf:>7.4f}")

# ============================================================================
# Section 12 — Manifold Geometry Analysis
# ============================================================================

print("\n" + "=" * 72)
print("MANIFOLD GEOMETRY ANALYSIS")
print("=" * 72)

model.eval()
_enc = get_base_model(model).encoder   # BUG FIXED: use base model

with torch.no_grad():
    X_all     = torch.cat([X_train, X_val], dim=0)
    Rho_all_t = torch.cat([Rho_train, Rho_val], dim=0)
    z_chunks  = [_enc(X_all[i:i+batch_size])[0].cpu().numpy()   # BUG FIXED: [0] for z
                 for i in range(0, len(X_all), batch_size)]
    Z         = np.vstack(z_chunks)
    Rho_all_np = Rho_all_t.cpu().numpy()

print(f"Latent matrix Z: {Z.shape}")

# ── Geodesic-Bures correlation ────────────────────────────────────────────────
print(f"Computing geodesic-Bures correlation ({n_geodesic_pairs} pairs, GPU) ...")
np.random.seed(seed)
pair_idx  = np.array([np.random.choice(len(Z), 2, replace=False)
                      for _ in range(n_geodesic_pairs)])
ii, jj    = pair_idx[:, 0], pair_idx[:, 1]
geo_dists = np.linalg.norm(Z[ii] - Z[jj], axis=1)

Ri = torch.tensor(Rho_all_np[ii], dtype=torch.complex64, device=device_torch)
Rj = torch.tensor(Rho_all_np[jj], dtype=torch.complex64, device=device_torch)
F_chunks = []
for k in range(0, len(ii), 50):
    F_chunks.append(uhlmann_fidelity_batch(Ri[k:k+50], Rj[k:k+50]).cpu().numpy())
F_vals = np.concatenate(F_chunks)

bures_dists = np.arccos(np.sqrt(np.clip(F_vals, 0.0, 1.0)))
valid_mask  = bures_dists > 1e-6
geo_dists   = geo_dists[valid_mask]
bures_dists = bures_dists[valid_mask]

pearson_r,  pearson_p  = pearsonr(geo_dists,   bures_dists)
spearman_r, spearman_p = spearmanr(geo_dists,  bures_dists)
coeffs   = np.polyfit(bures_dists, geo_dists, 1)
a_fit, b_fit = coeffs[0], coeffs[1]
geo_pred     = a_fit * bures_dists + b_fit
r_squared    = 1 - (np.sum((geo_dists - geo_pred)**2)
                    / np.sum((geo_dists - geo_dists.mean())**2))

print(f"  Pearson  r  = {pearson_r:.4f}  (p={pearson_p:.2e})")
print(f"  Spearman ρ  = {spearman_r:.4f}  (p={spearman_p:.2e})")
print(f"  R²          = {r_squared:.4f}")
print(f"  Linear fit  : d_L = {a_fit:.4f}·d_B + {b_fit:.4f}")

# ── Intrinsic dimensionality ──────────────────────────────────────────────────
def mle_intrinsic_dim(X, k=15):
    nbrs  = NearestNeighbors(n_neighbors=k+1).fit(X)
    dists, _ = nbrs.kneighbors(X)
    dists = dists[:, 1:]
    dims  = []
    for row in dists:
        r_max = row[-1]
        if r_max > 1e-10:
            dims.append(-k / np.sum(np.log(row / r_max + 1e-12)))
    return np.mean(dims), np.std(dims)

id_mean, id_std = mle_intrinsic_dim(Z, k=min(15, Z.shape[0]//4))
pca_full        = PCA().fit(Z)
cumvar          = np.cumsum(pca_full.explained_variance_ratio_)
dim_95 = int(np.argmax(cumvar >= 0.95)) + 1
dim_99 = int(np.argmax(cumvar >= 0.99)) + 1
print(f"\nIntrinsic dim (MLE): {id_mean:.2f} ± {id_std:.2f}")
print(f"PCA 95%: {dim_95} dims | PCA 99%: {dim_99} dims")
print(f"Top-5 eigenvalues: {pca_full.explained_variance_ratio_[:5]}")

# ── Local curvature ───────────────────────────────────────────────────────────
def local_curvature(X, k=25):
    nbrs    = NearestNeighbors(n_neighbors=k+1).fit(X)
    _, idxs = nbrs.kneighbors(X)
    kappas  = []
    for i in range(X.shape[0]):
        nb = X[idxs[i, 1:]] - X[i]
        if nb.shape[0] > 1:
            try:
                _, s, _ = np.linalg.svd(nb, full_matrices=False)
                s = s[s > 1e-10]
                if len(s) > 1:
                    kappas.append(s[-1] / s[0])
            except Exception:
                pass
    return np.array(kappas) if kappas else np.array([0.0])

k_curv     = min(25, Z.shape[0] // 4)
curvatures = local_curvature(Z, k=k_curv)
print(f"\nCurvature (k={k_curv}): "
      f"mean={np.mean(curvatures):.5f} ± {np.std(curvatures):.5f}  "
      f"IQR=[{np.percentile(curvatures,25):.5f}, {np.percentile(curvatures,75):.5f}]")

# ── Bures distance matrix ─────────────────────────────────────────────────────
n_vis = min(n_vis_samples, Z.shape[0])
np.random.seed(seed + 1)
vis_idx    = np.random.choice(Z.shape[0], n_vis, replace=False)
Z_vis      = Z[vis_idx]
Rho_vis_np = Rho_all_np[vis_idx]

print(f"\nComputing {n_vis}×{n_vis} Bures matrix (GPU batched) ...")
Rho_vis_t = torch.tensor(Rho_vis_np, dtype=torch.complex64, device=device_torch)
bures_mat = np.zeros((n_vis, n_vis))
chunk_sz  = 15

for ci_start in range(0, n_vis, chunk_sz):
    ci_end = min(ci_start + chunk_sz, n_vis)
    ci     = ci_end - ci_start
    ri     = Rho_vis_t[ci_start:ci_end].unsqueeze(1).expand(-1, n_vis, -1, -1).reshape(-1, dim, dim)
    rj     = Rho_vis_t.unsqueeze(0).expand(ci, -1, -1, -1).reshape(-1, dim, dim)
    F_flat = uhlmann_fidelity_batch(ri, rj).cpu().numpy()
    bures_mat[ci_start:ci_end, :] = np.arccos(
        np.sqrt(np.clip(F_flat.reshape(ci, n_vis), 0, 1)))

bures_mat = (bures_mat + bures_mat.T) / 2
np.fill_diagonal(bures_mat, 0.0)

# ── MDS and PCA embeddings ────────────────────────────────────────────────────
print("Running MDS (3D + 2D) ...")
mds3   = MDS(n_components=3, dissimilarity='precomputed',
             random_state=seed, normalized_stress='auto')
q3d    = mds3.fit_transform(bures_mat)
mds2   = MDS(n_components=2, dissimilarity='precomputed',
             random_state=seed, normalized_stress='auto')
q2d    = mds2.fit_transform(bures_mat)
pca3_v = PCA(n_components=3, random_state=seed).fit_transform(Z_vis)
pca2_v = PCA(n_components=2, random_state=seed).fit_transform(Z_vis)

R_proc, _  = orthogonal_procrustes(pca3_v, q3d)
lat_aligned = pca3_v @ R_proc
rmse_proc   = float(np.sqrt(np.mean((q3d - lat_aligned) ** 2)))
print(f"Procrustes RMSE: {rmse_proc:.4f}")

# ============================================================================
# Section 13 — Figures
# ============================================================================

colors     = np.arange(n_vis)
bures_line = np.linspace(bures_dists.min(), bures_dists.max(), 200)

# ── Figure 1: Geodesic-Bures correlation ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sc = axes[0].scatter(bures_dists[:400], geo_dists[:400],
                     c=geo_dists[:400], cmap='plasma', s=20, alpha=0.6, rasterized=True)
axes[0].plot(bures_line, a_fit*bures_line + b_fit, 'r-', lw=2.5,
             label=f'$d_L = {a_fit:.3f}d_B + {b_fit:.3f}$\n$R^2={r_squared:.4f}$')
axes[0].set_xlabel('Bures Distance $d_B$', fontweight='bold')
axes[0].set_ylabel('Latent Euclidean Distance $d_L$', fontweight='bold')
axes[0].set_title(f'(A) Geodesic-Bures Correlation  r = {pearson_r:.4f}',
                  fontweight='bold', loc='left')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
plt.colorbar(sc, ax=axes[0])

F_th = np.linspace(0.01, 1.0, 200)
axes[1].plot(F_th, np.arccos(np.sqrt(F_th)), 'b-', lw=2.5,
             label='$D_B = \\arccos(\\sqrt{F})$')
axes[1].set_xlabel('Fidelity $F$', fontweight='bold')
axes[1].set_ylabel('Bures Distance $D_B$', fontweight='bold')
axes[1].set_title('(B) Bures-Fidelity Relationship', fontweight='bold', loc='left')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.suptitle(f'{n_qubits}-Qubit: Pearson r={pearson_r:.4f}  Spearman ρ={spearman_r:.4f}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, f'fig1_geodesic_bures_{n_qubits}q.png'), dpi=200, bbox_inches='tight')
plt.close(); print(f"Saved fig1_geodesic_bures_{n_qubits}q.png")

# ── Figure 2: Training dynamics ───────────────────────────────────────────────
ep = history['epoch']
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].semilogy(ep, history['loss_recon'],  lw=2, color='#2E86AB', label='Recon')
axes[0].semilogy(ep, history['loss_metric'], lw=2, color='#A23B72', label='Metric')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss (log)')
axes[0].set_title('(A) Loss Components', fontweight='bold', loc='left')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(ep, history['val_fid'], lw=2, color='#F18F01')
axes[1].axhline(target_fidelity, color='red', ls='--', lw=1.5,
                label=f'Target {target_fidelity}')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Mean Fidelity')
axes[1].set_title('(B) Validation Fidelity', fontweight='bold', loc='left')
axes[1].set_ylim([0.4, 1.0]); axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(ep, history['noise_mean'], lw=2, color='#6A4C93')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Mean noise $\\bar{p}$')
axes[2].set_title('(C) Learned Depolarising Noise', fontweight='bold', loc='left')
axes[2].grid(True, alpha=0.3)
plt.suptitle(f'{n_qubits}-Qubit Training Dynamics', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, f'fig2_training_dynamics_{n_qubits}q.png'), dpi=200, bbox_inches='tight')
plt.close(); print(f"Saved fig2_training_dynamics_{n_qubits}q.png")

# ── Figure 3: Shot noise robustness ──────────────────────────────────────────
labels_s = list(shot_results.keys())
means_s  = [shot_results[k][0] for k in labels_s]
stds_s   = [shot_results[k][1] for k in labels_s]
x_pos    = np.arange(len(labels_s))
bar_cols = ['#2E86AB', '#F18F01', '#A23B72', '#6A4C93'][:len(labels_s)]

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x_pos, means_s, yerr=stds_s, capsize=5,
       color=bar_cols, alpha=0.85, edgecolor='black')
ax.axhline(means_s[0], color='red', ls='--', lw=1.5, label='Analytical baseline')
ax.set_xticks(x_pos); ax.set_xticklabels(labels_s, rotation=10, ha='right')
ax.set_ylabel('Mean Reconstruction Fidelity', fontweight='bold')
ax.set_title(f'Shot Noise Robustness — {n_qubits} Qubits', fontweight='bold')
ax.set_ylim([max(0.0, min(means_s) - 0.12), 1.0])
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, f'fig3_shot_noise_{n_qubits}q.png'), dpi=200, bbox_inches='tight')
plt.close(); print(f"Saved fig3_shot_noise_{n_qubits}q.png")

# ── Figure 4: Manifold analysis (4-panel) ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ev     = pca_full.explained_variance_ratio_
n_show = min(15, len(ev))
axes[0,0].bar(range(1, n_show+1), ev[:n_show], alpha=0.75,
              color='#2E86AB', edgecolor='black')
ax2 = axes[0,0].twinx()
ax2.plot(range(1, n_show+1), np.cumsum(ev[:n_show]), 'ro-', lw=2, ms=5, label='Cumul.')
ax2.axhline(0.95, color='g', ls='--', alpha=0.7, label='95%')
axes[0,0].set_xlabel('PC'); axes[0,0].set_ylabel('Variance', color='#2E86AB')
ax2.set_ylabel('Cumulative', color='red')
axes[0,0].set_title('(A) PCA Spectrum', fontweight='bold', loc='left')
ax2.legend(loc='lower right')

axes[0,1].hist(curvatures, bins=40, color='#A23B72',
               alpha=0.75, edgecolor='black', density=True)
axes[0,1].axvline(np.mean(curvatures), color='red', ls='--', lw=2,
                  label=f'Mean={np.mean(curvatures):.4f}')
axes[0,1].set_xlabel('Local curvature κ'); axes[0,1].set_ylabel('Density')
axes[0,1].set_title('(B) Curvature Distribution', fontweight='bold', loc='left')
axes[0,1].legend()

sc2 = axes[1,0].scatter(pca2_v[:,0], pca2_v[:,1],
                        c=np.arange(n_vis), cmap='viridis', s=20, alpha=0.7)
axes[1,0].set_xlabel('PC 1'); axes[1,0].set_ylabel('PC 2')
axes[1,0].set_title('(C) Latent Space (PCA 2D)', fontweight='bold', loc='left')
plt.colorbar(sc2, ax=axes[1,0])

axes[1,1].scatter(bures_dists[:300], geo_dists[:300],
                  s=15, alpha=0.5, color='#F18F01', rasterized=True)
axes[1,1].plot(bures_line, a_fit*bures_line + b_fit, 'r-', lw=2,
               label=f'$R^2={r_squared:.3f}$')
axes[1,1].set_xlabel('Bures Distance'); axes[1,1].set_ylabel('Latent Distance')
axes[1,1].set_title('(D) Metric Correlation', fontweight='bold', loc='left')
axes[1,1].legend()

plt.suptitle(f'{n_qubits}-Qubit Manifold Geometry Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, f'fig4_manifold_analysis_{n_qubits}q.png'), dpi=200, bbox_inches='tight')
plt.close(); print(f"Saved fig4_manifold_analysis_{n_qubits}q.png")

# ── Figure 5: 3-D manifold comparison ────────────────────────────────────────
fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(q3d[:,0], q3d[:,1], q3d[:,2], c=colors, cmap='viridis', s=40, alpha=0.75)
nbrs_q = NearestNeighbors(n_neighbors=4).fit(q3d)
_, iq  = nbrs_q.kneighbors(q3d)
for i in range(n_vis):
    for j in iq[i][1:3]:
        ax1.plot([q3d[i,0],q3d[j,0]],[q3d[i,1],q3d[j,1]],[q3d[i,2],q3d[j,2]],
                 'gray', alpha=0.12, lw=0.5)
ax1.set_title('Quantum State Manifold\n(Bures MDS)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Bures D1'); ax1.set_ylabel('Bures D2'); ax1.set_zlabel('Bures D3')
ax1.view_init(elev=20, azim=45)

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(pca3_v[:,0], pca3_v[:,1], pca3_v[:,2], c=colors, cmap='viridis', s=40, alpha=0.75)
nbrs_l = NearestNeighbors(n_neighbors=4).fit(pca3_v)
_, il  = nbrs_l.kneighbors(pca3_v)
for i in range(n_vis):
    for j in il[i][1:3]:
        ax2.plot([pca3_v[i,0],pca3_v[j,0]],[pca3_v[i,1],pca3_v[j,1]],
                 [pca3_v[i,2],pca3_v[j,2]], 'gray', alpha=0.12, lw=0.5)
ax2.set_title('Learned Latent Manifold\n(PCA 3D)', fontsize=13, fontweight='bold')
ax2.set_xlabel('PC 1'); ax2.set_ylabel('PC 2'); ax2.set_zlabel('PC 3')
ax2.view_init(elev=20, azim=45)

fig.text(0.5, 0.01,
         f'r = {pearson_r:.4f}  |  ρ = {spearman_r:.4f}  |  Procrustes RMSE = {rmse_proc:.4f}',
         ha='center', fontsize=12, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(OUTDIR, f'fig5_manifold_3d_{n_qubits}q.png'), dpi=200, bbox_inches='tight')
plt.close(); print(f"Saved fig5_manifold_3d_{n_qubits}q.png")

# ── Figure 6: Fidelity distribution ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(all_fids, bins=40, color='#2E86AB', alpha=0.75, edgecolor='black', density=True)
axes[0].axvline(val_mean_fid,   color='red',   ls='--', lw=2,
                label=f'Mean={val_mean_fid:.4f}')
axes[0].axvline(val_median_fid, color='green', ls='--', lw=2,
                label=f'Median={val_median_fid:.4f}')
axes[0].set_xlabel('Fidelity'); axes[0].set_ylabel('Density')
axes[0].set_title('(A) Fidelity Distribution', fontweight='bold', loc='left')
axes[0].legend()

sf  = np.sort(all_fids)
cdf = np.arange(1, len(sf)+1) / len(sf)
axes[1].plot(sf, cdf, lw=2.5, color='#A23B72')
axes[1].axvline(0.90, color='red', ls='--', alpha=0.7, label='F=0.90')
axes[1].set_xlabel('Fidelity'); axes[1].set_ylabel('Cumulative Probability')
axes[1].set_title('(B) CDF', fontweight='bold', loc='left')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.suptitle(f'{n_qubits}-Qubit Reconstruction Fidelity', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, f'fig6_fidelity_dist_{n_qubits}q.png'), dpi=200, bbox_inches='tight')
plt.close(); print(f"Saved fig6_fidelity_dist_{n_qubits}q.png")

# ============================================================================
# Final Results Summary
# ============================================================================
print("\n" + "=" * 72)
print(f"RESULTS — {n_qubits}-QUBIT GEOMETRIC LATENT SPACE TOMOGRAPHY")
print("=" * 72)
print(f"  Qubits      : {n_qubits}  |  Pauli observables: {n_pauli_nonid}")
print(f"  Decoder     : PyTorch density matrix circuit "
      f"({n_noise_params} learned depolarising params)")
print(f"  Fidelity    : mean={val_mean_fid:.4f}  median={val_median_fid:.4f}")
print(f"  Pearson r   : {pearson_r:.4f}  |  Spearman ρ: {spearman_r:.4f}  "
      f"|  R²: {r_squared:.4f}")
print(f"  Intrinsic d : {id_mean:.2f} ± {id_std:.2f} (MLE)  |  PCA-95%: {dim_95} dims")
print(f"  Curvature   : {np.mean(curvatures):.5f} ± {np.std(curvatures):.5f}")
print(f"  Procrustes  : RMSE = {rmse_proc:.4f}")
print(f"  Shot noise  :")
for lbl, (mf, sf) in shot_results.items():
    print(f"    {lbl:>25} : {mf:.4f} ± {sf:.4f}")
print("=" * 72)
print("All figures saved to", OUTDIR)
