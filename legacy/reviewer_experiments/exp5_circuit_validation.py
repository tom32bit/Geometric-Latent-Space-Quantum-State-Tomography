"""
Experiment 5: TorchDMCircuit Validation vs Numpy Reference
===========================================================
Generates random circuit parameters, computes density matrices and Pauli
expectations with both TorchDMCircuit (float32) and an independent numpy
reference implementation (float64). Reports max absolute error.

Usage:
    python exp5_circuit_validation.py
    python exp5_circuit_validation.py --n_qubits 3 --n_circuits 200
"""

import os
import sys
import argparse
import json
import time

import numpy as np

import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description="TorchDMCircuit validation vs Qiskit")
parser.add_argument("--n_qubits", type=int, default=3, choices=[2, 3, 4])
parser.add_argument("--n_circuits", type=int, default=100,
                    help="Number of random circuits to test")
parser.add_argument("--outdir", type=str, default="/kaggle/working")
args = parser.parse_args()

N_QUBITS = args.n_qubits
N_CIRCUITS = args.n_circuits
OUTDIR = args.outdir
os.makedirs(OUTDIR, exist_ok=True)

dim             = 2 ** N_QUBITS
n_pauli_nonid   = 4 ** N_QUBITS - 1
n_layers        = N_QUBITS + 2
n_angles        = N_QUBITS * 3 * n_layers
n_noise_params  = n_layers
NOISE_MAX       = 0.15

device_torch = torch.device("cpu")  # CPU for exact comparison

print(f"Validating TorchDMCircuit: {N_QUBITS}q, {N_CIRCUITS} random circuits")

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
full_paulis_arr = np.stack(full_paulis, axis=0)

# ============================================================================
# TorchDMCircuit (same as tomography.py)
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

    def forward(self, params):
        rho = self.forward_rho(params)
        return torch.einsum('bij,pji->bp', rho, self.paulis).real


# ============================================================================
# Numpy Reference Implementation
# ============================================================================

def reference_circuit_rho(angles_np, noise_np, nq, nl):
    """
    Independent numpy implementation of the same circuit for validation.
    Uses float64 for maximum precision.
    angles_np: (n_angles,) rotation angles
    noise_np:  (n_layers,) depolarizing noise strengths
    """
    d = 2 ** nq
    rho = np.zeros((d, d), dtype=complex)
    rho[0, 0] = 1.0  # |0...0><0...0|

    # Build rotation matrices
    def rx(theta):
        c, s = np.cos(theta/2), np.sin(theta/2)
        return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
    def ry(theta):
        c, s = np.cos(theta/2), np.sin(theta/2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    def rz(theta):
        return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=complex)

    # CNOT via projectors
    def build_cnot(ctrl, nq_):
        I2 = np.eye(2, dtype=complex)
        P0 = np.array([[1,0],[0,0]], dtype=complex)
        P1 = np.array([[0,0],[0,1]], dtype=complex)
        Xg = np.array([[0,1],[1,0]], dtype=complex)
        t0 = np.array([[1]], dtype=complex); t1 = np.array([[1]], dtype=complex)
        for q in range(nq_):
            if q == ctrl: t0 = np.kron(t0, P0); t1 = np.kron(t1, P1)
            elif q == ctrl+1: t0 = np.kron(t0, I2); t1 = np.kron(t1, Xg)
            else: t0 = np.kron(t0, I2); t1 = np.kron(t1, I2)
        return t0 + t1

    cnot_gates = [build_cnot(q, nq) for q in range(nq - 1)]

    # Depolarizing channel per qubit
    X_p = np.array([[0,1],[1,0]], dtype=complex)
    Y_p = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z_p = np.array([[1,0],[0,-1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def depol_1q(rho_, p_, qubit_):
        def full_pauli(P_1q):
            parts = [P_1q if j == qubit_ else I2 for j in range(nq)]
            r = parts[0]
            for j in range(1, nq): r = np.kron(r, parts[j])
            return r
        Px = full_pauli(X_p); Py = full_pauli(Y_p); Pz = full_pauli(Z_p)
        return (1-p_)*rho_ + (p_/3)*(Px@rho_@Px.conj().T + Py@rho_@Py.conj().T + Pz@rho_@Pz.conj().T)

    idx = 0
    for layer in range(nl):
        # Single-qubit rotations: per qubit, compose RZ @ RY @ RX
        U_full = np.eye(1, dtype=complex)
        # Build tensor product of per-qubit unitaries
        for q in range(nq):
            U_q = rz(angles_np[idx+2]) @ ry(angles_np[idx+1]) @ rx(angles_np[idx])
            idx += 3
            U_full = np.kron(U_full, U_q)

        # Apply unitary
        rho = U_full @ rho @ U_full.conj().T

        # CNOT chain
        for C in cnot_gates:
            rho = C @ rho @ C.conj().T

        # Depolarizing noise per qubit
        p_layer = noise_np[layer]
        for q in range(nq):
            rho = depol_1q(rho, p_layer, q)

    return rho


# ============================================================================
# Main Validation
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    circuit = TorchDMCircuit(N_QUBITS, n_layers, n_angles, n_noise_params,
                              full_paulis_arr, pauli_idx_nonid)
    circuit.eval()

    max_rho_errors = []
    max_exp_errors = []
    trace_errors = []
    psd_violations = []

    t0 = time.time()
    for ci in range(N_CIRCUITS):
        # Random parameters
        angles = np.random.uniform(-np.pi, np.pi, n_angles).astype(np.float32)
        noise = np.random.uniform(0.0, NOISE_MAX, n_noise_params).astype(np.float32)
        params_np = np.concatenate([angles, noise])

        # TorchDMCircuit
        params_t = torch.tensor(params_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            rho_torch = circuit.forward_rho(params_t)[0].numpy()
            exps_torch = circuit(params_t)[0].numpy()

        # Qiskit/numpy reference
        rho_ref = reference_circuit_rho(angles.astype(np.float64),
                                      noise.astype(np.float64),
                                      N_QUBITS, n_layers)

        # Reference expectations
        exps_ref = np.array([np.real(np.trace(rho_ref @ full_paulis_arr[k]))
                             for k in pauli_idx_nonid], dtype=np.float64)

        # Errors
        rho_err = np.max(np.abs(rho_torch - rho_ref))
        exp_err = np.max(np.abs(exps_torch - exps_ref))
        tr_err = abs(np.trace(rho_torch) - 1.0)
        eigvals = np.linalg.eigvalsh(rho_torch)
        psd_viol = abs(min(eigvals.real.min(), 0.0))

        max_rho_errors.append(rho_err)
        max_exp_errors.append(exp_err)
        trace_errors.append(tr_err)
        psd_violations.append(psd_viol)

        if (ci + 1) % 20 == 0:
            print(f"  [{ci+1}/{N_CIRCUITS}] rho_err={rho_err:.2e}, exp_err={exp_err:.2e}")

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*60}")
    print(f"TORCHDMCIRCUIT VALIDATION — {N_QUBITS}q, {N_CIRCUITS} circuits")
    print(f"{'='*60}")
    print(f"Density matrix max error:  {np.max(max_rho_errors):.2e} "
          f"(mean {np.mean(max_rho_errors):.2e})")
    print(f"Pauli exp max error:       {np.max(max_exp_errors):.2e} "
          f"(mean {np.mean(max_exp_errors):.2e})")
    print(f"Trace error:               {np.max(trace_errors):.2e} "
          f"(mean {np.mean(trace_errors):.2e})")
    print(f"PSD violation:             {np.max(psd_violations):.2e} "
          f"(mean {np.mean(psd_violations):.2e})")
    print(f"Time: {elapsed:.1f}s")

    PASS = (np.max(max_exp_errors) < 1e-4)
    print(f"\nVERDICT: {'PASS' if PASS else 'FAIL'} "
          f"(threshold: max Pauli exp error < 1e-4)")

    results = {
        'n_qubits': N_QUBITS,
        'n_circuits': N_CIRCUITS,
        'max_rho_error': float(np.max(max_rho_errors)),
        'mean_rho_error': float(np.mean(max_rho_errors)),
        'max_exp_error': float(np.max(max_exp_errors)),
        'mean_exp_error': float(np.mean(max_exp_errors)),
        'max_trace_error': float(np.max(trace_errors)),
        'max_psd_violation': float(np.max(psd_violations)),
        'pass': bool(PASS),
    }

    outfile = os.path.join(OUTDIR, f'circuit_validation_{N_QUBITS}q.json')
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to: {outfile}")
