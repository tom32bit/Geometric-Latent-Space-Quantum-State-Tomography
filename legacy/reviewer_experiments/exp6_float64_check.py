"""
Experiment 6: Float64 Fidelity Precision Check
===============================================
Checks whether float32 vs float64 precision affects fidelity computation.
Loads a trained model, evaluates on validation data in both precisions,
reports the difference.

Also checks: does eps=1e-12 do anything in float32 (machine eps ~1e-7)?

Usage:
    python exp6_float64_check.py --n_qubits 3 --checkpoint /path/to/best_model_3q.pt
"""

import os
import sys
import argparse
import json
import time

import numpy as np
from scipy.linalg import sqrtm
import scipy.linalg

import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description="Float64 precision check")
parser.add_argument("--n_qubits", type=int, required=True, choices=[2, 3, 4])
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--n_eval", type=int, default=500,
                    help="Number of states to evaluate")
parser.add_argument("--outdir", type=str, default="/kaggle/working")
args = parser.parse_args()

N_QUBITS = args.n_qubits
OUTDIR = args.outdir
os.makedirs(OUTDIR, exist_ok=True)

dim             = 2 ** N_QUBITS
n_pauli_nonid   = 4 ** N_QUBITS - 1
n_layers        = N_QUBITS + 2
n_angles        = N_QUBITS * 3 * n_layers
n_noise_params  = n_layers
NOISE_MAX       = 0.15
NOISE_MIN       = 0.0
latent_dim      = 10 * N_QUBITS

device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Pauli Basis
# ============================================================================
I_m = np.array([[1, 0], [0, 1]], dtype=complex)
X_m = np.array([[0, 1], [1, 0]], dtype=complex)
Y_m = np.array([[0,-1j],[1j, 0]], dtype=complex)
Z_m = np.array([[1, 0], [0,-1]], dtype=complex)
_paulis_1q = [I_m, X_m, Y_m, Z_m]
_labels_1q = ['I', 'X', 'Y', 'Z']

full_paulis, full_labels = [], []
for index in range(4 ** N_QUBITS):
    digits, tmp = [], index
    for _ in range(N_QUBITS):
        digits.append(tmp % 4); tmp //= 4
    digits = digits[::-1]
    P = _paulis_1q[digits[0]]; lab = _labels_1q[digits[0]]
    for d in digits[1:]:
        P = np.kron(P, _paulis_1q[d]); lab = lab + _labels_1q[d]
    full_paulis.append(P); full_labels.append(lab)

identity_label  = 'I' * N_QUBITS
pauli_idx_nonid = [i for i, l in enumerate(full_labels) if l != identity_label]
pauli_idx_ident = [i for i, l in enumerate(full_labels) if l == identity_label][0]
full_paulis_arr = np.stack(full_paulis, axis=0)

# ============================================================================
# Model Classes (same as other scripts)
# ============================================================================

class TorchDMCircuit(nn.Module):
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
        else: self.cnot_gates = None
        self.register_buffer('pauli_x', torch.tensor(
            np.array([[0,1],[1,0]], dtype=np.complex64), dtype=torch.complex64))
        self.register_buffer('pauli_y', torch.tensor(
            np.array([[0,-1j],[1j,0]], dtype=np.complex64), dtype=torch.complex64))
        self.register_buffer('pauli_z', torch.tensor(
            np.array([[1,0],[0,-1]], dtype=np.complex64), dtype=torch.complex64))

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

class Encoder(nn.Module):
    def __init__(self, input_dim, ldim):
        super().__init__()
        self.latent_geom = ldim - 2; self.latent_pur = 2
        h1 = max(128, 2*input_dim); h2 = max(64, h1//2)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, h1), nn.LayerNorm(h1), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(h1, h2), nn.LayerNorm(h2), nn.ReLU(), nn.Dropout(0.15))
        self.head_geom = nn.Linear(h2, self.latent_geom)
        self.head_pur = nn.Linear(h2, self.latent_pur)
    def forward(self, x):
        h = self.trunk(x); z_geom = self.head_geom(h); z_pur = torch.sigmoid(self.head_pur(h))
        return torch.cat([z_geom, z_pur], dim=-1), z_geom, z_pur

class HybridAE(nn.Module):
    def __init__(self, input_dim, ldim, n_ang, n_noise, paulis_arr, pidx):
        super().__init__()
        self.encoder = Encoder(input_dim, ldim)
        self.latent2angles = nn.Linear(ldim, n_ang)
        self.latent2noise = nn.Linear(ldim, n_noise)
        self.circuit = TorchDMCircuit(N_QUBITS, n_layers, n_ang, n_noise, paulis_arr, pidx)
        nn.init.constant_(self.latent2noise.bias, 0.0)
    def forward(self, x):
        z, z_geom, z_pur = self.encoder(x)
        angles = self.latent2angles(z)
        noise_p = torch.sigmoid(self.latent2noise(z)) * (NOISE_MAX - NOISE_MIN) + NOISE_MIN
        return self.circuit(torch.cat([angles, noise_p], dim=-1)), z, z_geom, z_pur


# ============================================================================
# Fidelity in float32 and float64
# ============================================================================

def fidelity_numpy_f64(rho_true, rho_pred):
    """Fidelity using scipy sqrtm in float64."""
    rho_true = rho_true.astype(np.complex128)
    rho_pred = rho_pred.astype(np.complex128)
    sq = sqrtm(rho_true)
    if np.any(np.isnan(sq)):
        sq = scipy.linalg.sqrtm(rho_true + 1e-15 * np.eye(dim, dtype=np.complex128))
    M = sq @ rho_pred @ sq
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.maximum(eigvals.real, 0)
    return float(np.sum(np.sqrt(eigvals)) ** 2)

def fidelity_numpy_f32(rho_true, rho_pred):
    """Fidelity using scipy sqrtm in float32 (complex64)."""
    rho_true = rho_true.astype(np.complex64)
    rho_pred = rho_pred.astype(np.complex64)
    sq = sqrtm(rho_true)
    if np.any(np.isnan(sq)):
        sq = scipy.linalg.sqrtm(rho_true + 1e-7 * np.eye(dim, dtype=np.complex64))
    M = sq @ rho_pred @ sq
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.maximum(eigvals.real, 0)
    return float(np.sum(np.sqrt(eigvals)) ** 2)

def sqrtm_psd_torch_f32(A, eps=1e-12):
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag_embed(torch.sqrt(eigvals)) @ eigvecs.conj().transpose(-2,-1)

def sqrtm_psd_torch_f64(A, eps=1e-15):
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag_embed(torch.sqrt(eigvals)) @ eigvecs.conj().transpose(-2,-1)

def uhlmann_fidelity_batch_f32(rho_a, rho_b):
    sqrt_a = sqrtm_psd_torch_f32(rho_a, eps=1e-12)
    M = sqrt_a @ rho_b @ sqrt_a
    eigvals = torch.clamp(torch.linalg.eigvalsh(M).real, min=0.0)
    return torch.clamp(torch.sum(torch.sqrt(eigvals), dim=-1)**2, 0.0, 1.0)

def uhlmann_fidelity_batch_f64(rho_a, rho_b):
    sqrt_a = sqrtm_psd_torch_f64(rho_a.to(torch.complex128), eps=1e-15)
    M = sqrt_a @ rho_b.to(torch.complex128) @ sqrt_a
    eigvals = torch.clamp(torch.linalg.eigvalsh(M).real, min=0.0)
    return torch.clamp(torch.sum(torch.sqrt(eigvals), dim=-1)**2, 0.0, 1.0)

def pauli_exps_to_rho_np(exps_nonid):
    rho = np.eye(dim, dtype=complex) / dim
    for k, idx in enumerate(pauli_idx_nonid):
        rho += exps_nonid[k] * full_paulis_arr[idx] / dim
    return 0.5 * (rho + rho.conj().T)

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import random
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # Generate test states
    print(f"Generating {args.n_eval} test states...")
    from scipy.optimize import brentq

    def compute_purity(rho): return float(np.real(np.trace(rho @ rho)))
    def random_haar(nq):
        d = 2**nq; v = np.random.randn(d) + 1j*np.random.randn(d)
        return v / np.linalg.norm(v)

    X_test, Rho_test = [], []
    while len(X_test) < args.n_eval:
        psi = random_haar(N_QUBITS); rho = np.outer(psi, psi.conj())
        tp = np.random.uniform(0.85, 0.95)
        def _f(p):
            r = (1-p)*rho + p*np.eye(dim, dtype=complex)/dim
            return compute_purity(r) - tp
        try:
            p = brentq(_f, 0.0, 1.0)
        except ValueError:
            continue
        rho_mixed = (1-p)*rho + p*np.eye(dim, dtype=complex)/dim
        rho_mixed = 0.5*(rho_mixed + rho_mixed.conj().T)
        eigv, eigvec = np.linalg.eigh(rho_mixed)
        eigv = np.maximum(eigv, 0)
        rho_mixed = (eigvec * eigv) @ eigvec.conj().T
        rho_mixed /= np.trace(rho_mixed)

        exps = np.array([np.real(np.trace(rho_mixed @ full_paulis_arr[k]))
                         for k in pauli_idx_nonid], dtype=np.float32)
        X_test.append(exps)
        Rho_test.append(rho_mixed.astype(np.complex64))

    X_test = np.stack(X_test)
    Rho_test = np.stack(Rho_test)

    # Load model
    print(f"Loading model: {args.checkpoint}")
    pauli_torch_local = torch.tensor(full_paulis_arr, dtype=torch.complex64, device=device_torch)
    nonid_idx_local = torch.tensor(pauli_idx_nonid, dtype=torch.long, device=device_torch)

    model = HybridAE(n_pauli_nonid, latent_dim, n_angles, n_noise_params,
                      full_paulis_arr, pauli_idx_nonid).to(device_torch)
    state_dict = torch.load(args.checkpoint, map_location=device_torch, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Get predictions
    print("Running model inference...")
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 128):
            xb = torch.tensor(X_test[i:i+128], dtype=torch.float32, device=device_torch)
            preds_b, *_ = model(xb)
            all_preds.append(preds_b.cpu().numpy())
    all_preds = np.concatenate(all_preds)

    # Reconstruct rho_pred
    Rho_pred = np.stack([pauli_exps_to_rho_np(p) for p in all_preds])

    # ── Test 1: eps=1e-12 in float32 ────────────────────────────────
    print("\n--- Test 1: Does eps=1e-12 matter in float32? ---")
    f32_eps12 = []
    f32_eps7 = []
    for i in range(min(100, len(Rho_test))):
        rho_a = torch.tensor(Rho_test[i:i+1], dtype=torch.complex64, device='cpu')
        rho_b = torch.tensor(Rho_pred[i:i+1], dtype=torch.complex64, device='cpu')
        f1 = uhlmann_fidelity_batch_f32(rho_a, rho_b).item()
        # With eps=1e-7 (float32 machine eps)
        sqrt_a = sqrtm_psd_torch_f32(rho_a, eps=1e-7)
        M = sqrt_a @ rho_b @ sqrt_a
        eigvals = torch.clamp(torch.linalg.eigvalsh(M).real, min=0.0)
        f2 = torch.clamp(torch.sum(torch.sqrt(eigvals), dim=-1)**2, 0.0, 1.0).item()
        f32_eps12.append(f1)
        f32_eps7.append(f2)

    diff_eps = np.abs(np.array(f32_eps12) - np.array(f32_eps7))
    print(f"  eps=1e-12 vs eps=1e-7 in float32:")
    print(f"    max |diff|  = {diff_eps.max():.2e}")
    print(f"    mean |diff| = {diff_eps.mean():.2e}")
    if diff_eps.max() < 1e-6:
        print(f"    -> eps=1e-12 has NO EFFECT in float32 (below machine eps)")
    else:
        print(f"    -> eps value MATTERS")

    # ── Test 2: float32 vs float64 fidelity ─────────────────────────
    print("\n--- Test 2: float32 vs float64 fidelity ---")
    fids_f32 = []
    fids_f64 = []
    fids_scipy_f64 = []

    n_compare = min(args.n_eval, 500)
    for i in range(n_compare):
        # Torch float32
        rho_a = torch.tensor(Rho_test[i:i+1], dtype=torch.complex64, device='cpu')
        rho_b = torch.tensor(Rho_pred[i:i+1], dtype=torch.complex64, device='cpu')
        f32 = uhlmann_fidelity_batch_f32(rho_a, rho_b).item()

        # Torch float64
        f64 = uhlmann_fidelity_batch_f64(rho_a, rho_b).item()

        # Scipy float64 (gold standard)
        f_scipy = fidelity_numpy_f64(Rho_test[i], Rho_pred[i])

        fids_f32.append(f32)
        fids_f64.append(f64)
        fids_scipy_f64.append(f_scipy)

    fids_f32 = np.array(fids_f32)
    fids_f64 = np.array(fids_f64)
    fids_scipy = np.array(fids_scipy_f64)

    diff_32_64 = np.abs(fids_f32 - fids_f64)
    diff_32_scipy = np.abs(fids_f32 - fids_scipy)
    diff_64_scipy = np.abs(fids_f64 - fids_scipy)

    print(f"  float32 vs float64 (torch):")
    print(f"    max |diff|  = {diff_32_64.max():.2e}")
    print(f"    mean |diff| = {diff_32_64.mean():.2e}")
    print(f"  float32 vs scipy-f64:")
    print(f"    max |diff|  = {diff_32_scipy.max():.2e}")
    print(f"    mean |diff| = {diff_32_scipy.mean():.2e}")
    print(f"  float64 vs scipy-f64:")
    print(f"    max |diff|  = {diff_64_scipy.max():.2e}")
    print(f"    mean |diff| = {diff_64_scipy.mean():.2e}")

    # ── Test 3: Does precision change mean fidelity? ────────────────
    print(f"\n--- Test 3: Aggregate fidelity impact ---")
    print(f"  Mean F (float32 torch): {np.mean(fids_f32):.6f}")
    print(f"  Mean F (float64 torch): {np.mean(fids_f64):.6f}")
    print(f"  Mean F (scipy float64): {np.mean(fids_scipy):.6f}")
    print(f"  |Δ mean F| (f32 vs f64): {abs(np.mean(fids_f32) - np.mean(fids_f64)):.6f}")

    SIGNIFICANT = abs(np.mean(fids_f32) - np.mean(fids_f64)) > 0.001
    print(f"\n  VERDICT: Precision difference is "
          f"{'SIGNIFICANT (>0.001)' if SIGNIFICANT else 'NEGLIGIBLE (<0.001)'}")
    if SIGNIFICANT:
        print("  -> Consider switching fidelity computation to float64")
    else:
        print("  -> float32 is sufficient for reported results")

    results = {
        'n_qubits': N_QUBITS,
        'n_eval': n_compare,
        'eps_12_vs_7_max_diff': float(diff_eps.max()),
        'f32_vs_f64_max_diff': float(diff_32_64.max()),
        'f32_vs_f64_mean_diff': float(diff_32_64.mean()),
        'f32_vs_scipy_max_diff': float(diff_32_scipy.max()),
        'mean_fid_f32': float(np.mean(fids_f32)),
        'mean_fid_f64': float(np.mean(fids_f64)),
        'mean_fid_scipy': float(np.mean(fids_scipy)),
        'significant': SIGNIFICANT,
    }
    outfile = os.path.join(OUTDIR, f'float64_check_{N_QUBITS}q.json')
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {outfile}")
