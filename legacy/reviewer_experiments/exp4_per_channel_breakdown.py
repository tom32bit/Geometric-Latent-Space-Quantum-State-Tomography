"""
Experiment 4: Per-Channel Fidelity Breakdown
=============================================
Post-hoc analysis on existing trained models. No retraining needed.
Loads saved model checkpoints, generates test states per channel type,
evaluates fidelity per state type.

Reports: mean fidelity per state type (7 channels) for each qubit count.

Usage:
    python exp4_per_channel_breakdown.py --n_qubits 3 --checkpoint /path/to/best_model_3q.pt
    python exp4_per_channel_breakdown.py --n_qubits 2 --checkpoint /path/to/best_model_2q.pt
"""

import os
import sys
import argparse
import json
import time
import random

import numpy as np
from scipy.linalg import sqrtm
import scipy.linalg

import torch
import torch.nn as nn

# ============================================================================
# Arguments
# ============================================================================
parser = argparse.ArgumentParser(description="Per-channel fidelity breakdown")
parser.add_argument("--n_qubits", type=int, required=True, choices=[2, 3, 4])
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to best_model_{n}q.pt checkpoint")
parser.add_argument("--n_eval_per_channel", type=int, default=200,
                    help="Number of test states per channel type")
parser.add_argument("--outdir", type=str, default="/kaggle/working")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

N_QUBITS = args.n_qubits
OUTDIR = args.outdir
SEED = args.seed
N_EVAL = args.n_eval_per_channel
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================================
# Derived Config
# ============================================================================
dim             = 2 ** N_QUBITS
n_pauli_nonid   = 4 ** N_QUBITS - 1
n_layers        = N_QUBITS + 2
n_angles        = N_QUBITS * 3 * n_layers
n_noise_params  = n_layers
NOISE_MAX       = 0.15
NOISE_MIN       = 0.0
latent_dim      = 10 * N_QUBITS

device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Per-channel breakdown: {N_QUBITS}q, {N_EVAL} states/channel")

# ============================================================================
# Pauli Basis
# ============================================================================
norm_factor = 1.0 / dim
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
pauli_torch     = torch.tensor(full_paulis_arr, dtype=torch.complex64, device=device_torch)
nonid_idx_torch = torch.tensor(pauli_idx_nonid, dtype=torch.long, device=device_torch)

# ============================================================================
# State Generation Helpers
# ============================================================================

def validate_density_matrix(rho, tol=1e-6):
    rho = 0.5 * (rho + rho.conj().T)
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0)
    rho = (eigvecs * eigvals) @ eigvecs.conj().T
    rho /= np.trace(rho); return rho

def compute_purity(rho): return float(np.real(np.trace(rho @ rho)))

def random_haar_statevector(n_q):
    d = 2 ** n_q; v = np.random.randn(d) + 1j * np.random.randn(d)
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
    try: return brentq(_f, 0.0, 1.0)
    except ValueError: return 0.5

# ============================================================================
# Individual Channel Generators (each returns a single density matrix)
# ============================================================================

def gen_depolarized():
    tp = np.random.uniform(0.85, 0.95)
    p = purity_to_depol_param(tp, N_QUBITS)
    psi = random_haar_statevector(N_QUBITS); rho = np.outer(psi, psi.conj())
    return validate_density_matrix((1-p)*rho + p*np.eye(dim, dtype=complex)/dim)

def gen_werner():
    tp = np.random.uniform(0.85, 0.95); d = dim
    a = 1+1/d; bw = -2/d; cw = 1/d - tp
    disc = bw**2 - 4*a*cw
    p = np.clip((-bw + np.sqrt(max(disc, 0)))/(2*a), 0, 1) if disc >= 0 else 0.5
    psi = np.zeros(dim, dtype=complex); psi[0] = psi[-1] = 1/np.sqrt(2)
    pr = np.outer(psi, psi.conj())
    return validate_density_matrix(p*pr + (1-p)*np.eye(dim, dtype=complex)/dim)

def gen_isotropic():
    tp = np.random.uniform(0.85, 0.95)
    p = purity_to_depol_param(tp, N_QUBITS)
    psi = random_haar_statevector(N_QUBITS); pr = np.outer(psi, psi.conj())
    return validate_density_matrix((1-p)*pr + p*np.eye(dim, dtype=complex)/dim)

def gen_amplitude_damped():
    def _ad(gamma):
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
    tp = np.random.uniform(0.85, 0.95)
    lo, hi = 0.0, 0.99
    for _ in range(20):
        mid = (lo + hi) / 2
        if compute_purity(_ad(mid)) > tp: lo = mid
        else: hi = mid
    return _ad((lo + hi) / 2)

def gen_phase_damped():
    def _pd(gamma):
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
    tp = np.random.uniform(0.85, 0.95)
    lo, hi = 0.0, 0.99
    for _ in range(20):
        mid = (lo + hi) / 2
        if compute_purity(_pd(mid)) > tp: lo = mid
        else: hi = mid
    return _pd((lo + hi) / 2)

def gen_thermal():
    def _th(beta):
        H = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        H = (H + H.conj().T) / 2
        eigvals, eigvecs = np.linalg.eigh(H)
        rho = eigvecs @ np.diag(np.exp(-beta * eigvals)) @ eigvecs.conj().T
        rho /= np.trace(rho); return validate_density_matrix(rho)
    tp = np.random.uniform(0.85, 0.95)
    lo, hi = 0.01, 10.0
    for _ in range(20):
        mid = (lo + hi) / 2
        if compute_purity(_th(mid)) < tp: lo = mid  # flipped: higher beta = higher purity
        else: hi = mid
    return _th((lo + hi) / 2)

def gen_separable():
    tp = np.random.uniform(0.85, 0.95)
    pq = tp ** (1.0 / N_QUBITS); rho = None
    for _ in range(N_QUBITS):
        p_this = float(np.clip(np.random.uniform(pq-0.03, pq+0.03), 0.5, 1.0))
        p_dep = purity_to_depol_param(p_this, 1)
        psi_q = random_haar_statevector(1); rq = np.outer(psi_q, psi_q.conj())
        rq = (1-p_dep)*rq + p_dep*np.eye(2, dtype=complex)/2
        rho = rq if rho is None else np.kron(rho, rq)
    return validate_density_matrix(rho)

CHANNELS = [
    ('Depolarized',  gen_depolarized),
    ('Werner',       gen_werner),
    ('Isotropic',    gen_isotropic),
    ('AmpDamped',    gen_amplitude_damped),
    ('PhaseDamped',  gen_phase_damped),
    ('Thermal',      gen_thermal),
    ('Separable',    gen_separable),
]

# ============================================================================
# Model Classes (for loading checkpoint)
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
# GPU evaluation helpers
# ============================================================================

def pauli_exps_to_rho_torch(exps_nonid):
    B = exps_nonid.shape[0]
    exps_full = torch.zeros(B, 4**N_QUBITS, dtype=torch.float32, device=exps_nonid.device)
    exps_full[:, pauli_idx_ident] = 1.0; exps_full[:, nonid_idx_torch] = exps_nonid
    exps_c = exps_full.to(torch.complex64).unsqueeze(-1).unsqueeze(-1)
    rho = (exps_c * pauli_torch.unsqueeze(0)).sum(dim=1) * norm_factor
    return 0.5 * (rho + rho.conj().permute(0, 2, 1))

def sqrtm_psd_torch(A, eps=1e-12):
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag_embed(torch.sqrt(eigvals)) @ eigvecs.conj().transpose(-2,-1)

def uhlmann_fidelity_batch(rho_a, rho_b, eps=1e-12):
    sqrt_a = sqrtm_psd_torch(rho_a, eps); M = sqrt_a @ rho_b @ sqrt_a
    eigvals = torch.clamp(torch.linalg.eigvalsh(M).real, min=0.0)
    return torch.clamp(torch.sum(torch.sqrt(eigvals), dim=-1)**2, 0.0, 1.0)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    random.seed(SEED); np.random.seed(SEED)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = HybridAE(n_pauli_nonid, latent_dim, n_angles, n_noise_params,
                      full_paulis_arr, pauli_idx_nonid).to(device_torch)
    state_dict = torch.load(args.checkpoint, map_location=device_torch, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded.")

    # Evaluate per channel
    all_results = {}
    for ch_name, gen_fn in CHANNELS:
        print(f"\n  Evaluating {ch_name} ({N_EVAL} states)...", end='', flush=True)
        X_list, Rho_list = [], []
        attempts = 0
        while len(X_list) < N_EVAL and attempts < N_EVAL * 10:
            attempts += 1
            try:
                rho = gen_fn()
                p = compute_purity(rho)
                if 0.84 <= p <= 0.96:  # slight tolerance
                    X_list.append(rho_to_pauli_exps(rho))
                    Rho_list.append(rho.astype(np.complex64))
            except Exception:
                pass

        if len(X_list) < N_EVAL:
            print(f" WARNING: only got {len(X_list)}/{N_EVAL}")

        X_ch = np.stack(X_list)
        Rho_ch = np.stack(Rho_list)

        # Compute fidelities
        fids = []
        with torch.no_grad():
            for i in range(0, len(X_ch), 128):
                xb = torch.tensor(X_ch[i:i+128], dtype=torch.float32, device=device_torch)
                rt = torch.tensor(Rho_ch[i:i+128], dtype=torch.complex64, device=device_torch)
                preds_b, *_ = model(xb)
                rho_pred = pauli_exps_to_rho_torch(preds_b)
                F_b = uhlmann_fidelity_batch(rho_pred, rt)
                fids.extend(F_b.cpu().tolist())

        fids = np.array(fids)
        purities = np.array([compute_purity(r) for r in Rho_ch])

        result = {
            'channel': ch_name,
            'n_states': len(fids),
            'mean_fidelity': float(np.mean(fids)),
            'median_fidelity': float(np.median(fids)),
            'std_fidelity': float(np.std(fids)),
            'min_fidelity': float(np.min(fids)),
            'max_fidelity': float(np.max(fids)),
            'mean_purity': float(np.mean(purities)),
        }
        all_results[ch_name] = result
        print(f"  F = {result['mean_fidelity']:.4f} +/- {result['std_fidelity']:.4f}")

    # Summary
    print(f"\n{'='*72}")
    print(f"PER-CHANNEL FIDELITY BREAKDOWN — {N_QUBITS}-qubit")
    print(f"{'='*72}")
    print(f"{'Channel':>15} | {'Mean F':>8} | {'Med F':>8} | {'Std':>7} | {'Min':>7} | {'Max':>7} | {'Purity':>7}")
    print("-" * 78)
    for ch_name, r in all_results.items():
        print(f"{ch_name:>15} | {r['mean_fidelity']:.4f}   | {r['median_fidelity']:.4f}   | "
              f"{r['std_fidelity']:.4f}  | {r['min_fidelity']:.4f}  | {r['max_fidelity']:.4f}  | "
              f"{r['mean_purity']:.4f}")

    # Overall
    all_fids = []
    for r in all_results.values():
        all_fids.append(r['mean_fidelity'])
    print(f"\n  Overall mean across channels: {np.mean(all_fids):.4f}")
    print(f"  Hardest channel: {min(all_results, key=lambda k: all_results[k]['mean_fidelity'])}")
    print(f"  Easiest channel: {max(all_results, key=lambda k: all_results[k]['mean_fidelity'])}")

    # Save
    outfile = os.path.join(OUTDIR, f'per_channel_{N_QUBITS}q.json')
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to: {outfile}")
