#!/usr/bin/env python3
"""Choi-CMI of random brickwork Clifford circuits via stabilizer formalism.

Circuit: depth-t brickwork of uniform random 2-qubit Cliffords on n qubits
(even bonds on even layers, odd bonds on odd layers).
Choi state on 2n qubits (site i -> ref_i, sys_i) is a pure stabilizer state
with generators  X_i^ref (x) U X_i U^dag  and  Z_i^ref (x) U Z_i U^dag.

Entropy of a qubit subset A:  S_A = |A| - dim{ g in S : g|_{A^c} = 1 }
computed by GF(2) rank.  For stabilizer states ALL Renyi entropies are equal
(flat spectrum), so I_2 = I_vN exactly.

CMI window on sites: A=[s,s+a), B, C contiguous; Choi qubits = both (ref,sys)
per site.  We record the center-window CMI and the max over bulk positions,
averaged over R realizations per depth.
"""
import numpy as np
import stim


def random_brickwork_tableau(n, depth, rng):
    sim = stim.TableauSimulator()
    sim.set_num_qubits(n)
    for layer in range(depth):
        start = 0 if layer % 2 == 0 else 1
        for i in range(start, n - 1, 2):
            t2 = stim.Tableau.random(2)
            sim.do_tableau(t2, [i, i + 1])
    return sim.current_inverse_tableau() ** -1


def choi_generator_matrix(tab, n):
    """Binary symplectic matrix (2n generators x 4n columns [x|z]) of the Choi
    stabilizers on 2n qubits ordered (ref_0, sys_0, ref_1, sys_1, ...)."""
    N = 2 * n
    G = np.zeros((2 * n, 2 * N), dtype=np.uint8)

    def put(row, qubit, xbit, zbit):
        G[row, qubit] ^= xbit
        G[row, N + qubit] ^= zbit

    for i in range(n):
        # generator X_i^ref (x) U X_i U^dag   and   Z_i^ref (x) U Z_i U^dag
        for kind, out in ((0, tab.x_output(i)), (1, tab.z_output(i))):
            row = 2 * i + kind
            if kind == 0:
                put(row, 2 * i, 1, 0)          # X on ref_i
            else:
                put(row, 2 * i, 0, 1)          # Z on ref_i
            for j in range(n):
                p = out[j]                      # 0=I,1=X,2=Y,3=Z
                if p == 1:   put(row, 2 * j + 1, 1, 0)
                elif p == 2: put(row, 2 * j + 1, 1, 1)
                elif p == 3: put(row, 2 * j + 1, 0, 1)
    return G


def gf2_rank(M):
    M = M.copy()
    rank = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = None
        for r in range(rank, rows):
            if M[r, c]:
                piv = r; break
        if piv is None:
            continue
        M[[rank, piv]] = M[[piv, rank]]
        mask = M[:, c].astype(bool).copy()
        mask[rank] = False
        M[mask] ^= M[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def entropy(G, qubits_keep, N):
    """S(rho_A) in bits for stabilizer state with generator matrix G on N qubits."""
    comp = np.array([q for q in range(N) if q not in qubits_keep], dtype=int)
    cols = np.concatenate([comp, N + comp])
    d_triv = G.shape[0] - gf2_rank(G[:, cols])
    return len(qubits_keep) - d_triv


def site_qubits(sites):
    out = []
    for s in sites:
        out += [2 * s, 2 * s + 1]
    return out


def cmi(G, N, s, a, b, c):
    A = list(range(s, s + a)); B = list(range(s + a, s + a + b))
    C = list(range(s + a + b, s + a + b + c))
    S_AB  = entropy(G, site_qubits(A + B), N)
    S_BC  = entropy(G, site_qubits(B + C), N)
    S_B   = entropy(G, site_qubits(B), N)
    S_ABC = entropy(G, site_qubits(A + B + C), N)
    return S_AB + S_BC - S_B - S_ABC


def dense_selftest(reps=20):
    """Validate GF(2) entropies against dense vN entropies of the actual
    Choi statevector (small n).  Stabilizer entropy theorem => must be equal."""
    import itertools
    n = 5
    for r in range(reps):
        depth = 1 + r % 4
        sim = stim.TableauSimulator()
        sim.set_num_qubits(n)
        layers = []
        for layer in range(depth):
            start = 0 if layer % 2 == 0 else 1
            for i in range(start, n - 1, 2):
                t2 = stim.Tableau.random(2)
                sim.do_tableau(t2, [i, i + 1])
                layers.append((t2, i))
        tab = sim.current_inverse_tableau() ** -1
        G = choi_generator_matrix(tab, n)
        # dense Choi state on 2n qubits: Bell pairs + U on sys
        sv = stim.TableauSimulator()
        sv.set_num_qubits(2 * n)
        for i in range(n):
            sv.h(2 * i)
            sv.cnot(2 * i, 2 * i + 1)      # ref_i -> sys_i
        for t2, i in layers:
            sv.do_tableau(t2, [2 * i + 1, 2 * (i + 1) + 1])   # act on sys qubits
        psi = np.array(sv.state_vector(endian="little"), dtype=complex)
        N = 2 * n
        for (s, a, b, c) in [(0,1,1,1),(1,1,2,1),(0,1,3,1),(1,2,1,1)]:
            if s + a + b + c > n: continue
            A=list(range(s,s+a)); B=list(range(s+a,s+a+b)); C=list(range(s+a+b,s+a+b+c))
            for sites in (A+B, B+C, B, A+B+C):
                q = site_qubits(sites)
                S_gf2 = entropy(G, q, N)
                # dense reduced entropy
                keep = sorted(q)
                t = psi.reshape([2] * N, order="F")   # little endian: qubit0 fastest
                perm = keep + [x for x in range(N) if x not in keep]
                t = np.transpose(t, perm).reshape(2 ** len(keep), -1)
                w = np.linalg.svd(t, compute_uv=False) ** 2
                w = w[w > 1e-12]
                S_dense = float(-(w * np.log2(w)).sum())
                assert abs(S_gf2 - S_dense) < 1e-5, (S_gf2, S_dense, s, a, b, c)
    print(f"SELFTEST PASSED: GF(2) stabilizer entropy == dense vN entropy "
          f"({reps} random circuits)", flush=True)


def main():
    import argparse, csv, time
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--depths", type=int, nargs="+", default=[1,2,3,4,5,6,7,8,9,10])
    p.add_argument("--bvals", type=int, nargs="+", default=list(range(1, 35)))
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--margin", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--csv", type=str, default="clifford_cmi.csv")
    args = p.parse_args()

    if args.selftest:
        dense_selftest()
    n = args.n; a = c = 1
    rng = np.random.default_rng(args.seed)
    rows = []
    for depth in args.depths:
        t0 = time.time()
        acc_cen = {b: [] for b in args.bvals}
        acc_max = {b: [] for b in args.bvals}
        for r in range(args.reps):
            tab = random_brickwork_tableau(n, depth, rng)
            G = choi_generator_matrix(tab, n)
            N = 2 * n
            for b in args.bvals:
                w = a + b + c
                s_cen = (n - w) // 2
                acc_cen[b].append(cmi(G, N, s_cen, a, b, c))
                vals = [cmi(G, N, s, a, b, c)
                        for s in range(args.margin, n - w - args.margin + 1)]
                acc_max[b].append(max(vals))
        for b in args.bvals:
            mc, sc = np.mean(acc_cen[b]), np.std(acc_cen[b]) / np.sqrt(args.reps)
            mm = np.mean(acc_max[b])
            frac = np.mean(np.array(acc_cen[b]) > 0)
            rows.append((depth, b, mc, sc, mm, frac))
            print(f"depth={depth}  b={b}:  <I>_center={mc:.4f}±{sc:.4f}  "
                  f"<I_max>={mm:.4f}  P(I>0)={frac:.2f}", flush=True)
        print(f"  [depth {depth}: {time.time()-t0:.1f}s]", flush=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth", "b", "cmi_center_mean", "cmi_center_sem",
                    "cmi_max_mean", "frac_nonzero"])
        w.writerows(rows)
    print("wrote", args.csv)


if __name__ == "__main__":
    main()
