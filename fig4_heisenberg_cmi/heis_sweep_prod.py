#!/usr/bin/env python3
"""
Choi-CMI of the Heisenberg + Z-dephasing Lindbladian via the TEBD/MPO pipeline
of main.py, so we can push to large N (e.g. N=50) and read off bulk reduced
density matrices on small windows -- killing the finite-size / boundary effect
that contaminated the dense brute-force (n<=6) calculation.

Physics (build_bell_pair_chain, identical to main.py):
    each site = one Bell pair (system qubit + reference qubit), dim 4
    H = sum_n (eps/2) Z^sys_n + sum_n ( Jx XX + Jy YY + Jz ZZ )^sys_{n,n+1}
    dissipator: local depolarizing, jump ops X/Y/Z with rates gamma_x/y/z
The evolved many-body density matrix rho(t) IS the Choi state J(t): site n
carries the Choi pair (ref_n, sys_n).  Region R of |R| physical sites -> a
4^|R| reduced density matrix obtained by MPO partial trace.

CMI:  I_J(A:C|B) = S(J_AB) + S(J_BC) - S(J_B) - S(J_ABC),
windows A=[s,s+a), B=[s+a,s+a+b), C=[..,+c).  Because N is large we sweep s only
over BULK positions (away from both ends) and report the max + central value.

Reuses main.py's get_augmented_mps_manual / augmented_mps_to_mpo_tensors /
mpo_partial_trace, copied here to avoid importing main.py (whose top level
executes a heavy PtTebd run).
"""

# ── Config -------------------------------------------------------------------
N_PAIRS   = 50
EPS       = 0.0        # uniform Z field; 0 = production (paper model has no field)
JX = JY = JZ = 0.5       # Heisenberg coupling
GAMMA_X   = 0.5          # depolarization rates (single-site X,Y,Z jumps)
GAMMA_Y   = 0.5
GAMMA_Z   = 1.0
DT        = 1.0e-2      # production value
EPSREL    = 1.0e-12       # TEBD SVD truncation (production value)
ORDER     = 1
PHYS_DIM  = 4            # local site dim (= 1 Bell pair = 2 qubits)
A_C       = 1            # |A| = |C|
B_VALUES  = [1, 2, 3, 4, 5] # buffer sizes (ABC window = a+b+c, 4^win reduced dm)
STEP_TIMES = [0.2, 0.3]  # times to record CMI
BULK_MARGIN = 6          # exclude this many sites from each end when sweeping s
SWEEP_HALF = 99           # only sweep s in [center-3, center+3] (bulk is uniform)
EIG_TOL   = 1e-12
CSV_PATH  = "cmi_mpo_results.csv"
PLOT_PATH = "cmi_mpo_results.png"
# ----------------------------------------------------------------------------

import argparse
import csv

import numpy as np
import oqupy
from oqupy.config import NpDtype
from oqupy.operators import sigma
import quimb.tensor as qtn

# ── single-qubit ops, acting on the SYSTEM qubit of each Bell pair -----------
sx = sigma("x"); sy = sigma("y"); sz = sigma("z")
id2 = np.eye(2, dtype=complex)
sx_sys = np.kron(sx, id2)
sy_sys = np.kron(sy, id2)
sz_sys = np.kron(sz, id2)


# ══════════════════════════════════════════════════════════════════════════
#  System + initial Choi (Bell) state   (copied from main.py)
# ══════════════════════════════════════════════════════════════════════════
def bell_dm_phi_plus():
    v00 = np.array([1, 0, 0, 0], dtype=complex)
    v11 = np.array([0, 0, 0, 1], dtype=complex)
    bell = (v00 + v11) / np.sqrt(2.0)
    return np.outer(bell, bell.conj())


def build_bell_pair_chain(N, eps, jx, jy, jz, gamma_x, gamma_y, gamma_z):
    chain = oqupy.SystemChain(hilbert_space_dimensions=[4] * N)
    for n in range(N):
        chain.add_site_hamiltonian(site=n, hamiltonian=eps * 0.5 * sz_sys)
    for n in range(N - 1):
        if jx:
            chain.add_nn_hamiltonian(n, jx * sx_sys, sx_sys)
        if jy:
            chain.add_nn_hamiltonian(n, jy * sy_sys, sy_sys)
        if jz:
            chain.add_nn_hamiltonian(n, jz * sz_sys, sz_sys)
    for n in range(N):        # single-site DEPOLARIZATION (X,Y,Z jumps)
        if gamma_x:
            chain.add_site_dissipation(n, sx_sys, gamma=gamma_x)
        if gamma_y:
            chain.add_site_dissipation(n, sy_sys, gamma=gamma_y)
        if gamma_z:
            chain.add_site_dissipation(n, sz_sys, gamma=gamma_z)
    return chain


def make_initial_augmented_mps_bell(N):
    rho = bell_dm_phi_plus()
    return oqupy.AugmentedMPS([rho for _ in range(N)])


# ── MPS/MPO extraction (copied from main.py, AugmentedMPS-bug-free reader) ───
class SimpleAugMPS:
    def __init__(self, gammas, lambdas):
        self.gammas = gammas
        self.lambdas = lambdas
        self.n = len(gammas)


def get_augmented_mps_manual(pt_tebd):
    t_mps = pt_tebd._t_mps
    gammas = []
    for i in range(t_mps.n):
        g = np.array(t_mps.get_gamma(i), dtype=NpDtype)
        sh = g.shape
        r = len(sh)
        if r == 4:
            tg = g
        elif r == 3:
            tg = g.reshape(sh[0], sh[1], 1, sh[2])
        elif r == 2:
            tg = g.reshape(1, sh[0] * sh[1], 1, 1)
        elif r == 1:
            tg = g.reshape(1, sh[0], 1, 1)
        else:
            raise ValueError(f"gamma rank {r}")
        gammas.append(tg)

    bond_dims = []
    for g1, g2 in zip(gammas[:-1], gammas[1:]):
        assert g1.shape[3] == g2.shape[0], (g1.shape, g2.shape)
        bond_dims.append(g1.shape[3])

    lambdas = []
    for chi in bond_dims:
        lam = t_mps.get_lambda(len(lambdas))
        if lam is None:
            lam_vec = np.ones(chi, dtype=NpDtype)
        else:
            lam = np.array(lam, dtype=NpDtype)
            lam_vec = np.diag(lam) if lam.ndim == 2 else lam
            assert lam_vec.shape[0] == chi
        lambdas.append(lam_vec)
    return SimpleAugMPS(gammas, lambdas)


def augmented_mps_to_mpo_tensors(aug, vec_order="C", swap_ud=False):
    N = aug.n
    gammas_hat = []
    for n in range(N):
        G = np.array(aug.gammas[n], copy=True)
        L, P, A, R = G.shape
        if A != 1:
            raise NotImplementedError("only A=1 (process_tensors=None)")
        if n > 0:
            lam = np.array(aug.lambdas[n - 1])
            if lam.ndim == 2:
                lam = np.diag(lam)
            G = lam[:, None, None, None] * G
        gammas_hat.append(G)

    mpo_tensors = []
    d = None
    for G in gammas_hat:
        L, P, A, R = G.shape
        if d is None:
            d = int(round(P ** 0.5))
            assert d * d == P
        G2 = G[:, :, 0, :]
        T = np.reshape(G2, (L, d, d, R), order=vec_order)
        if swap_ud:
            T = np.transpose(T, (0, 2, 1, 3))
        mpo_tensors.append(T)
    return mpo_tensors


def build_choi_mpo(pt_tebd):
    aug = get_augmented_mps_manual(pt_tebd)
    mpo_tensors = augmented_mps_to_mpo_tensors(aug, vec_order="C", swap_ud=False)
    tlist = []
    for i, t in enumerate(mpo_tensors):
        if i == 0:
            l, u, d, r = t.shape
            t = t.reshape(u, d, r)
        if i == len(mpo_tensors) - 1:
            l, u, d, r = t.shape
            t = t.reshape(l, u, d)
        tlist.append(t)
    return qtn.MatrixProductOperator(tlist, shape="ludr")


def mpo_partial_trace(mpo, keep):
    tn = mpo.copy()
    for i in range(tn.L):
        if i not in keep:
            bra, ket = tn[i].inds[-2:]
            tn[i].reindex_({bra: ket})
    data = (tn ^ all).data
    phys_d = data.shape[0]
    sites = len(data.shape) // 2
    x1 = np.arange(0, 2 * sites, 2)
    x2 = np.arange(1, 2 * sites, 2)
    data = data.transpose(np.append(x1, x2))
    return data.reshape([phys_d ** sites, phys_d ** sites])


# ══════════════════════════════════════════════════════════════════════════
#  Entropy / CMI on bulk windows
# ══════════════════════════════════════════════════════════════════════════
def vn_entropy(rho, tol=EIG_TOL):
    rho = (rho + rho.conj().T) / 2
    w = np.linalg.eigvalsh(rho)
    w = w[w > tol]
    return float(-np.sum(w * np.log2(w)))


def cmi_window(mpo, s, a, b, c):
    A = list(range(s, s + a))
    B = list(range(s + a, s + a + b))
    C = list(range(s + a + b, s + a + b + c))
    S_AB = vn_entropy(mpo_partial_trace(mpo, set(A + B)))
    S_BC = vn_entropy(mpo_partial_trace(mpo, set(B + C)))
    S_B = vn_entropy(mpo_partial_trace(mpo, set(B)))
    S_ABC = vn_entropy(mpo_partial_trace(mpo, set(A + B + C)))
    return S_AB + S_BC - S_B - S_ABC


def cmi_bulk(mpo, a, b, c, margin):
    """Sweep window start s over bulk positions; return (max, central, s_max)."""
    N = mpo.L
    w = a + b + c
    center = (N - w) // 2
    s_lo = max(margin, center - SWEEP_HALF)
    s_hi = min(N - w - margin, center + SWEEP_HALF)
    if s_hi < s_lo:
        s_lo, s_hi = 0, N - w           # fall back to full sweep
    vals = []
    for s in range(s_lo, s_hi + 1):
        vals.append((cmi_window(mpo, s, a, b, c), s))
    vmax, smax = max(vals)
    s_center = s_lo + (s_hi - s_lo) // 2
    v_center = cmi_window(mpo, s_center, a, b, c)
    return vmax, v_center, smax


# ══════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=N_PAIRS)
    p.add_argument("--eps", type=float, default=EPS)
    p.add_argument("--jx", type=float, default=JX)
    p.add_argument("--jy", type=float, default=JY)
    p.add_argument("--jz", type=float, default=JZ)
    p.add_argument("--gx", type=float, default=GAMMA_X)
    p.add_argument("--gy", type=float, default=GAMMA_Y)
    p.add_argument("--gz", type=float, default=GAMMA_Z)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--epsrel", type=float, default=EPSREL)
    p.add_argument("--ac", type=int, default=A_C)
    p.add_argument("--ac-list", type=int, nargs="*", default=None,
                   help="additional a=c values computed at b=1 (window 2a+1 <= 7)")
    p.add_argument("--bvals", type=int, nargs="+", default=B_VALUES)
    p.add_argument("--times", type=float, nargs="+", default=STEP_TIMES)
    p.add_argument("--margin", type=int, default=BULK_MARGIN)
    p.add_argument("--csv", type=str, default=CSV_PATH)
    p.add_argument("--plot", type=str, default=PLOT_PATH)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    N = args.n
    a = c = args.ac
    bvals = [b for b in args.bvals if a + b + c <= N - 2 * args.margin]
    # target step index for each requested time
    target_steps = sorted(set(int(round(t / args.dt)) for t in args.times))
    target_steps = [s for s in target_steps if s >= 1]
    max_step = max(target_steps)

    print(f"N={N} sites (Choi pairs), dt={args.dt}, epsrel={args.epsrel}; "
          f"stepping to {max_step} steps (t={max_step*args.dt:.3f})")
    chain = build_bell_pair_chain(N, args.eps, args.jx, args.jy, args.jz,
                                  args.gx, args.gy, args.gz)
    init = make_initial_augmented_mps_bell(N)
    params = oqupy.PtTebdParameters(dt=args.dt, order=args.order if hasattr(args, "order") else ORDER,
                                    epsrel=args.epsrel)
    pt = oqupy.PtTebd(
        initial_augmented_mps=init,
        system_chain=chain,
        process_tensors=[None] * N,
        parameters=params,
        dynamics_sites=list(range(N)),
    )
    pt.initialize()

    import time
    rows = []
    for step in range(1, max_step + 1):
        ts = time.time()
        pt.compute_step()
        try:
            tm = pt._t_mps
            chi = max(np.array(tm.get_gamma(i)).shape[-1] for i in range(tm.n - 1))
        except Exception:
            chi = -1
        print(f"  step {step:3d}/{max_step}  dt_step={time.time()-ts:5.1f}s  chi~{chi}",
              flush=True)
        if step not in target_steps:
            continue
        t = step * args.dt
        mpo = build_choi_mpo(pt)
        maxchi = max(t.shape for t in mpo.arrays)  # rough bond info
        combos = [(a, b, c) for b in bvals]
        if args.ac_list:
            combos += [(ac, 1, ac) for ac in args.ac_list
                       if ac != a and 2 * ac + 1 <= 7]
        for (aa, b, cc) in combos:
            vmax, vcen, smax = cmi_bulk(mpo, aa, b, cc, args.margin)
            rows.append((t, aa, b, cc, b, vmax, vcen, smax))
            print(f"  t={t:5.2f} step={step:3d}  a={aa} |B|={b}  "
                  f"I_J^max(bulk)={vmax:.6e}  I_J(center)={vcen:.6e}", flush=True)

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "a", "b", "c", "buffer_B", "cmi_max_bulk",
                    "cmi_center", "s_max"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {args.csv}")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib unavailable; skipping plot.")
            return
        times = sorted(set(r[0] for r in rows))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # left: CMI vs |B| for each time
        for t in times:
            xs = [r[4] for r in rows if r[0] == t and r[1] == a]
            ys = [r[5] for r in rows if r[0] == t and r[1] == a]
            ax1.semilogy(xs, np.maximum(ys, 1e-16), "o-", label=f"t={t:g}")
        ax1.set_xlabel("buffer |B|")
        ax1.set_ylabel(r"$I_J^{\max}$ (bulk) [bits]")
        ax1.set_title(f"CMI vs |B|  (N={N}, a=c={a})")
        ax1.legend(fontsize=8); ax1.grid(True, which="both", alpha=0.3)
        # right: CMI vs t for each |B|
        for b in bvals:
            xs = [r[0] for r in rows if r[4] == b and r[1] == a]
            ys = [r[5] for r in rows if r[4] == b and r[1] == a]
            ax2.semilogy(xs, np.maximum(ys, 1e-16), "o-", label=f"|B|={b}")
        ax2.set_xlabel("evolution time t")
        ax2.set_ylabel(r"$I_J^{\max}$ (bulk) [bits]")
        ax2.set_title(f"CMI vs t  (N={N}, a=c={a})")
        ax2.legend(fontsize=8); ax2.grid(True, which="both", alpha=0.3)
        fig.suptitle("Choi-CMI via TEBD/MPO: Heisenberg + local depolarizing "
                     f"(eps={args.eps}, J={args.jx}, "
                     f"gamma=({args.gx},{args.gy},{args.gz}))")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Wrote plot -> {args.plot}")


if __name__ == "__main__":
    main()
