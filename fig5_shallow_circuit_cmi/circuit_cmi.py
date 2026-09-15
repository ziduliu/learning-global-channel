"""Choi-state CMI of the noisy shallow circuit of Sec. IV.B, with COHERENT error.

Model (paper Sec. IV.B):
    Lambda = (prod_j D_gamma^(j)) . U_eps . U
    U        = prod_{i even} iSWAP_{i,i+1},      iSWAP = exp[i pi/4 (XX+YY)]
    U_eps    = prod_{i odd} R_{i,i+1}(eps) . prod_{i even} R_{i,i+1}(eps)
    R(eps)   = exp[i eps (a1 XX + a2 YY + a3 ZZ)],  a_k ~ U[0,1) per bond
    D_gamma  = (1-gamma) rho + gamma Z rho Z   (single-qubit dephasing)

The Choi state lives on a doubled chain: site i = (ref_i, sys_i), local dim 4.
All gates act on the sys qubits only.  The density matrix is propagated as a
vectorised MPS (physical index = 16 = row x col of the local 4x4 operator),
which keeps the bond dimension at 4^depth or below for a shallow circuit.

Entropies of contiguous windows are obtained by exact diagonalisation of the
reduced Choi state, giving

    I_J(A:C|B) = S(AB) + S(BC) - S(B) - S(ABC),   |A| = |C| = 1.

A dense-simulation self-test (small n) validates the MPS pipeline.
"""
import argparse
import json

import numpy as np

# ---------------------------------------------------------------- operators
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def build_two_site_gate(V):
    """16x16 matrix acting on (ref_i, sys_i, ref_{i+1}, sys_{i+1}) that applies
    the 4x4 gate V to (sys_i, sys_{i+1}) and identity to the refs."""
    G = np.zeros((16, 16), dtype=complex)
    Vt = V.reshape(2, 2, 2, 2)  # [s_i_out, s_j_out, s_i_in, s_j_in]
    for ri in range(2):
        for rj in range(2):
            for so in range(2):
                for sjo in range(2):
                    for si in range(2):
                        for sj in range(2):
                            out = ((ri * 2 + so) * 4) + (rj * 2 + sjo)
                            inn = ((ri * 2 + si) * 4) + (rj * 2 + sj)
                            G[out, inn] += Vt[so, sjo, si, sj]
    return G


def expm_herm(H):
    w, v = np.linalg.eigh(H)
    return (v * np.exp(1j * w)) @ v.conj().T


def iswap_gate():
    XX, YY = np.kron(X, X), np.kron(Y, Y)
    return expm_herm(np.pi / 4 * (XX + YY))


def coherent_gate(eps, a):
    XX, YY, ZZ = np.kron(X, X), np.kron(Y, Y), np.kron(Z, Z)
    return expm_herm(eps * (a[0] * XX + a[1] * YY + a[2] * ZZ))


# ------------------------------------------------------- vectorised MPS core
# site tensor: (Dl, 16, Dr); the physical index is (row, col) of the local 4x4
# operator, i.e. p = row * 4 + col.

def bell_site():
    """|Phi+><Phi+| on (ref, sys) as a 16-vector."""
    v = np.zeros(4, dtype=complex)
    v[0] = v[3] = 1 / np.sqrt(2)          # (|00> + |11>)/sqrt(2)
    rho = np.outer(v, v.conj())
    return rho.reshape(16)


def init_state(n):
    return [bell_site().reshape(1, 16, 1) for _ in range(n)]


def superop_two_site(G):
    """Superoperator rho -> G rho G^dagger on two sites, as a 256x256 matrix in
    the MPS index convention (row_i, col_i, row_j, col_j)."""
    S = np.kron(G, G.conj())              # acts on (row_i row_j, col_i col_j)
    S = S.reshape(4, 4, 4, 4, 4, 4, 4, 4)  # [ri,rj,ci,cj ; ri',rj',ci',cj']
    S = S.transpose(0, 2, 1, 3, 4, 6, 5, 7)  # -> [ri,ci,rj,cj ; ...]
    return S.reshape(256, 256)


def superop_one_site_kraus(kraus):
    S = np.zeros((16, 16), dtype=complex)
    for K in kraus:
        S += np.kron(K, K.conj())         # (row, col) ordering already correct
    return S


def apply_two_site(mps, i, S, max_bond=256, cutoff=1e-14):
    A, B = mps[i], mps[i + 1]
    Dl, _, Dm = A.shape
    _, _, Dr = B.shape
    T = np.tensordot(A, B, axes=([2], [0]))          # (Dl, 16, 16, Dr)
    T = T.reshape(Dl, 256, Dr)
    T = np.einsum("ab,lbr->lar", S, T)
    T = T.reshape(Dl * 16, 16 * Dr)
    U, s, Vh = np.linalg.svd(T, full_matrices=False)
    keep = max(1, min(max_bond, int(np.sum(s > cutoff * s[0]))))
    U, s, Vh = U[:, :keep], s[:keep], Vh[:keep]
    mps[i] = U.reshape(Dl, 16, keep)
    mps[i + 1] = (np.diag(s) @ Vh).reshape(keep, 16, Dr)


def apply_one_site(mps, i, S):
    mps[i] = np.einsum("ab,lbr->lar", S, mps[i])


def window_rdm(mps, keep):
    """Reduced Choi state on the contiguous window `keep` (list of site indices)."""
    trace_vec = np.eye(4, dtype=complex).reshape(16)     # vec(I_4)
    keep = set(keep)
    res = np.ones((1,), dtype=complex)                   # shape (*phys, bond)
    for i, A in enumerate(mps):
        if i in keep:
            res = np.tensordot(res, A, axes=([-1], [0]))          # (*phys, 16, Dr)
        else:
            A2 = np.tensordot(A, trace_vec, axes=([1], [0]))      # (Dl, Dr)
            res = np.tensordot(res, A2, axes=([-1], [0]))         # (*phys, Dr)
    res = res.reshape(res.shape[:-1])                    # drop trailing bond (=1)
    k = len(keep)
    res = res.reshape([4, 4] * k)
    rows = list(range(0, 2 * k, 2))
    cols = list(range(1, 2 * k, 2))
    return res.transpose(rows + cols).reshape(4 ** k, 4 ** k)


def vn_entropy(rho, tol=1e-13):
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    w = w[w > tol]
    w = w / w.sum()
    return float(-np.sum(w * np.log2(w)))


def cmi_window(mps, s, a, b, c):
    A = list(range(s, s + a))
    B = list(range(s + a, s + a + b))
    C = list(range(s + a + b, s + a + b + c))
    S_AB = vn_entropy(window_rdm(mps, A + B))
    S_BC = vn_entropy(window_rdm(mps, B + C))
    S_B = vn_entropy(window_rdm(mps, B))
    S_ABC = vn_entropy(window_rdm(mps, A + B + C))
    return S_AB + S_BC - S_B - S_ABC


# ------------------------------------------------------------------ circuit
def build_choi(n, eps, gamma, seed=0, max_bond=256):
    rng = np.random.default_rng(seed)
    mps = init_state(n)
    iswap = build_two_site_gate(iswap_gate())
    S_iswap = superop_two_site(iswap)
    # layer 1: iSWAP on even bonds
    for i in range(0, n - 1, 2):
        apply_two_site(mps, i, S_iswap, max_bond)
    # layer 2 + 3: coherent perturbation on even then odd bonds
    if eps > 0:
        for parity in (0, 1):
            for i in range(parity, n - 1, 2):
                G = build_two_site_gate(coherent_gate(eps, rng.random(3)))
                apply_two_site(mps, i, superop_two_site(G), max_bond)
    # dephasing on every sys qubit
    if gamma > 0:
        Zs = np.kron(I2, Z)
        S_dep = superop_one_site_kraus(
            [np.sqrt(1 - gamma) * np.eye(4, dtype=complex), np.sqrt(gamma) * Zs])
        for i in range(n):
            apply_one_site(mps, i, S_dep)
    return mps


# ------------------------------------------------------------- dense checker
def build_choi_dense(n, eps, gamma, seed=0):
    rng = np.random.default_rng(seed)
    dim = 4 ** n
    rho = np.array([1.0], dtype=complex).reshape(1, 1)
    site = bell_site().reshape(4, 4)
    rho = site
    for _ in range(n - 1):
        rho = np.kron(rho, site)

    def lift(V, i):
        """4x4 gate on (sys_i, sys_{i+1}) lifted to the full doubled chain."""
        G = build_two_site_gate(V)                      # 16x16 on sites i,i+1
        left = np.eye(4 ** i, dtype=complex)
        right = np.eye(4 ** (n - i - 2), dtype=complex)
        return np.kron(np.kron(left, G), right)

    for i in range(0, n - 1, 2):
        Uf = lift(iswap_gate(), i)
        rho = Uf @ rho @ Uf.conj().T
    if eps > 0:
        for parity in (0, 1):
            for i in range(parity, n - 1, 2):
                Uf = lift(coherent_gate(eps, rng.random(3)), i)
                rho = Uf @ rho @ Uf.conj().T
    if gamma > 0:
        for i in range(n):
            Zi = np.kron(np.kron(np.eye(4 ** i, dtype=complex),
                                 np.kron(I2, Z)),
                         np.eye(4 ** (n - i - 1), dtype=complex))
            rho = (1 - gamma) * rho + gamma * (Zi @ rho @ Zi)
    return rho


def dense_window_rdm(rho, n, keep):
    """Dense partial trace: keep the given sites, trace out the rest."""
    t = rho.reshape([4] * (2 * n))
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    row = list(letters[:n])
    col = []
    nxt = n
    for i in range(n):
        if i in keep:
            col.append(letters[nxt]); nxt += 1
        else:
            col.append(row[i])                       # tie row=col -> traced
    out = "".join(row[i] for i in keep) + "".join(col[i] for i in keep)
    t = np.einsum("".join(row) + "".join(col) + "->" + out, t)
    k = len(keep)
    return t.reshape(4 ** k, 4 ** k)


def selftest(n=5, eps=0.05, gamma=0.02, seed=1):
    mps = build_choi(n, eps, gamma, seed)
    rho = build_choi_dense(n, eps, gamma, seed)
    print(f"[selftest] n={n} eps={eps} gamma={gamma}")
    print(f"  trace: mps={np.trace(window_rdm(mps, list(range(n)))).real:.12f} "
          f"dense={np.trace(rho).real:.12f}")
    ok = True
    for keep in ([0], [1], [1, 2], [0, 1, 2], [1, 2, 3]):
        r1 = window_rdm(mps, keep)
        r2 = dense_window_rdm(rho, n, keep)
        err = np.max(np.abs(r1 - r2))
        s1, s2 = vn_entropy(r1), vn_entropy(r2)
        flag = "OK " if err < 1e-9 and abs(s1 - s2) < 1e-9 else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"  {flag} window={keep}: max|drho|={err:.2e}  S_mps={s1:.10f} S_dense={s2:.10f}")
    return ok


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--bmax", type=int, default=4)
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[0.0, 0.01, 0.02, 0.04, 0.08])
    ap.add_argument("--gamma", type=float, default=0.01)
    ap.add_argument("--reals", type=int, default=10)
    ap.add_argument("--out", default="circuit_cmi.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    import time
    acc = {}   # (eps, b) -> list of values
    for r in range(args.reals):
        for eps in args.eps:
            t0 = time.time()
            # seed depends only on r: identical circuit disorder across eps
            mps = build_choi(args.n, eps, args.gamma, seed=1234 + r)
            for b in range(1, args.bmax + 1):
                w = 1 + b + 1
                s0 = (args.n - w) // 2
                positions = [s0 - 1, s0, s0 + 1]
                val = max(cmi_window(mps, s, 1, b, 1) for s in positions)
                acc.setdefault((eps, b), []).append(val)
            print(f"real={r} eps={eps:<5} wall={time.time()-t0:.1f}s  "
                  + " ".join(f"b{b}={np.mean(acc[(eps,b)]):.3e}"
                             for b in range(1, args.bmax + 1)), flush=True)
        # write incrementally after each realization sweep
        results = {f"g{args.gamma}_e{e}_b{b}":
                   dict(gamma=args.gamma, eps=e, b=b,
                        mean=float(np.mean(v)),
                        sem=float(np.std(v) / np.sqrt(len(v))),
                        nreals=len(v))
                   for (e, b), v in acc.items()}
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    print("done ->", args.out)


if __name__ == "__main__":
    main()

