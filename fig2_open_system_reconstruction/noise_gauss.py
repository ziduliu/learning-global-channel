"""Paper (arXiv:2603.07037) Fig.2(c,d) noise model:
perturb each local Pauli expansion coefficient by additive Gaussian noise,
project back to a physical state, and record the local estimation error
    eta_i = 1/2 || J_hat_i - J_i ||_1 .
"""
import numpy as np


def add_gauss_pauli_noise(rho_choi, sigma, pauli_ops, rng=None):
    """c~_alpha = c_alpha + N(0, sigma^2) on all Pauli expectation values,
    then hermitize, clip negative eigenvalues, renormalize to unit trace.

    Returns (rho_noisy, eta) with eta = 1/2 ||rho_noisy - rho_choi||_1.
    """
    if rng is None:
        rng = np.random.default_rng()
    D = rho_choi.shape[0]
    # exact coefficients s_alpha = Tr(rho P_alpha)  (P_alpha: 4096 six-qubit Paulis)
    s = np.array([np.trace(rho_choi @ P).real for P in pauli_ops])
    s_noisy = s + rng.normal(0.0, sigma, size=s.shape)
    rho = sum(sn * P for sn, P in zip(s_noisy, pauli_ops)) / D
    # physical projection: hermitize, PSD, unit trace
    rho = (rho + rho.conj().T) / 2
    w, U = np.linalg.eigh(rho)
    w = np.clip(w, 0, None)
    rho = (U * w) @ U.conj().T
    tr = np.trace(rho).real
    if tr > 1e-14:
        rho = rho / tr
    # local estimation error (trace distance)
    dw = np.linalg.eigvalsh((rho - rho_choi + (rho - rho_choi).conj().T) / 2)
    eta = 0.5 * float(np.sum(np.abs(dw)))
    return rho, eta
