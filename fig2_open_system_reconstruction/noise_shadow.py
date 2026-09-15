import numpy as np
import itertools

# -------------------------
# Pauli-like basis on dim=64 ( (C^4)^{\otimes 3} ), total 4096 operators
# -------------------------

def generate_local_pauli_d4():
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    p1 = [I, X, Y, Z]
    return [np.kron(A, B) for A in p1 for B in p1]  # 16 ops on C^4

def generate_pauli_basis_choi_3qubit():
    lb = generate_local_pauli_d4()
    return [np.kron(np.kron(A, B), C) for A in lb for B in lb for C in lb]  # 4096 ops on 64-dim

# -------------------------
# Classical shadow (local Pauli) with shared shots -> includes covariance automatically
# -------------------------

def _kron_all(mats):
    out = mats[0]
    for M in mats[1:]:
        out = np.kron(out, M)
    return out

_SHADOW_CACHE = None

def _init_shadow_cache_6q():
    # 6 qubits because dim=64
    n = 6
    D = 64

    # all measurement settings (3^6 = 729), each entry in {1:X, 2:Y, 3:Z}
    settings = np.array(list(itertools.product([1, 2, 3], repeat=n)), dtype=np.int8)  # (729,6)
    n_settings = settings.shape[0]

    # subset masks (which qubits are non-identity in the Pauli string)
    masks = np.arange(1 << n, dtype=np.int64)  # 0..63
    bitpos = np.arange(n - 1, -1, -1, dtype=np.int64)  # MSB-first bits for outcome index
    mask_bits = ((masks[:, None] >> bitpos) & 1).astype(np.int8)  # (64,6)
    w = mask_bits.sum(axis=1).astype(int)
    fac = (3.0 ** w).astype(float)  # 3^weight, shape (64,)

    # sign(outcome, mask) = (-1)^{popcount(outcome & mask)}
    #popcount = np.array([i.bit_count() for i in range(1 << n)], dtype=np.int8)  # (64,)
    popcount = np.array([bin(i).count("1") for i in range(1 << n)], dtype=np.int8)

    parity = popcount[(np.arange(1 << n)[:, None] & masks[None, :])] & 1        # (64,64)
    sign_table_T = (1 - 2 * parity).astype(np.int8).T  # transpose to use matmul: (64,64)

    # index mapping: digits base-4 on 6 qubits (0=I,1=X,2=Y,3=Z), q0 is MSB
    pow4 = (4 ** np.arange(n - 1, -1, -1)).astype(np.int64)  # [4^5,...,1]

    # for each setting, the 64 compatible Pauli strings are digits = mask_bits * setting
    digits = settings[:, None, :] * mask_bits[None, :, :]           # (729,64,6)
    idxs_setting = (digits * pow4[None, None, :]).sum(axis=2)       # (729,64)

    # single-qubit rotations to measure X/Y/Z by rotating then measuring Z
    I2 = np.eye(2, dtype=complex)
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)  # S^\dagger
    U1 = {1: H, 2: H @ Sdg, 3: I2}

    # precompute the 64x64 unitary for each setting (saves time if you call many times)
    U_list = []
    for meas in settings:
        U_list.append(_kron_all([U1[int(m)] for m in meas]))
    # also cache conjugate-transposes to avoid repeated .conj().T
    Udag_list = [U.conj().T for U in U_list]

    return {
        "settings": settings,
        "idxs_setting": idxs_setting,
        "fac": fac,
        "sign_table_T": sign_table_T,
        "U_list": U_list,
        "Udag_list": Udag_list,
        "D": D,
        "n_settings": n_settings
    }

def add_shadow_noise_choi_3qubit_cov(rho_choi, N_snapshots, pauli_ops=None, seed=None):
    """
    Local-Pauli classical shadows on the 6-qubit (dim=64) state rho_choi.
    All 4096 coefficients share the same shots

    Returns: rho_noisy reconstructed by linear inversion + PSD projection.
    """
    global _SHADOW_CACHE
    rng = np.random.default_rng(seed)

    D = rho_choi.shape[0]
    assert rho_choi.shape == (D, D) and D == 64, "Expect rho_choi shape (64,64)."

    if pauli_ops is None:
        pauli_ops = generate_pauli_basis_choi_3qubit()  # may be memory-heavy

    if _SHADOW_CACHE is None:
        _SHADOW_CACHE = _init_shadow_cache_6q()

    settings = _SHADOW_CACHE["settings"]
    idxs_setting = _SHADOW_CACHE["idxs_setting"]
    fac = _SHADOW_CACHE["fac"]
    sign_T = _SHADOW_CACHE["sign_table_T"]
    sign_T = sign_T.astype(np.int64)   # upgrade to int64

    U_list = _SHADOW_CACHE["U_list"]
    Udag_list = _SHADOW_CACHE["Udag_list"]
    n_settings = _SHADOW_CACHE["n_settings"]

    # 1) Precompute p(outcome | setting) for all settings
    probs = np.empty((n_settings, D), dtype=float)
    for si in range(n_settings):
        U = U_list[si]
        Udag = Udag_list[si]
        rho_rot = U @ rho_choi @ Udag
        p = np.real(np.diag(rho_rot))
        p = np.maximum(p, 0.0)
        s = p.sum()
        probs[si] = (p / s) if s > 0 else (np.ones(D) / D)

    # 2) Sample how many shots go to each setting (equivalent to i.i.d. uniform settings)
    m_setting = rng.multinomial(int(N_snapshots), np.ones(n_settings) / n_settings)

    # 3) Accumulate all 4096 Pauli coefficients with shared shots
    accum = np.zeros(4096, dtype=float)
    for si, m in enumerate(m_setting):
        if m == 0:
            continue
        cnt = rng.multinomial(m, probs[si]).astype(np.int64)   # outcome counts, shape (64,)
        s_mask = sign_T @ cnt                                   # (64,), sums of +/- over outcomes for each mask
        accum[idxs_setting[si]] += fac * s_mask                 # update the 64 compatible Paulis for this setting

    s_hat = accum / float(N_snapshots)  # estimated Pauli expectations (shared-shot -> correlated)

    # 4) Reconstruct rho_noisy = (1/64) * sum_k s_hat[k] P_k
    rho_noisy = np.zeros_like(rho_choi, dtype=complex)
    for s, P in zip(s_hat, pauli_ops):
        rho_noisy += s * P
    rho_noisy /= 64.0

    # 5) Project to PSD + trace 1 (same style as your original code)
    rho_noisy = 0.5 * (rho_noisy + rho_noisy.conj().T)
    evals, evecs = np.linalg.eigh(rho_noisy)
    evals = np.clip(evals, 0.0, None)
    rho_noisy = evecs @ np.diag(evals) @ evecs.conj().T
    tr = np.trace(rho_noisy).real
    return (np.eye(64, dtype=complex) / 64.0) if tr <= 0 else (rho_noisy / tr)
