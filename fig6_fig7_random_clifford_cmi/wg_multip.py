#!/usr/bin/env python3
"""window_growth variant evaluating SEVERAL p values in one enumeration pass.

The spacetime-support counts K(g) depend only on (circuit, window); the
noise enters only through c_g = (1-4p/3)^K.  So we build K once per
component and take the WHT / purity sum once per p -- ~4 p values for the
price of one.  Same window geometry as window_growth.py: centered A|B|C,
a=c in 1..5, 2a+b <= WMAX, loud SKIP above dmax.
"""
import argparse, csv, time
import numpy as np

from noisy_clifford_cmi import (random_brickwork_layers, full_tableau,
                                choi_generator_matrix, pullback_generators,
                                gf2_nullspace, site_qubits)

WMAX = 14


def fwht(c):
    """Fully vectorized Walsh--Hadamard transform (identical result to the
    stage-wise butterfly loop in noisy_clifford_cmi, but no python-level
    inner loop -- essential for 2^26+ components)."""
    q = c.copy()
    n = len(q)
    h = 1
    while h < n:
        q = q.reshape(-1, 2, h)
        a_ = q[:, 0, :] + q[:, 1, :]
        b_ = q[:, 0, :] - q[:, 1, :]
        q = np.concatenate([a_[:, None, :], b_[:, None, :]], axis=1)
        h *= 2
    return q.reshape(n)


def window_entropies_multip(G, gen_xz, n, qubits_keep, ps, dmax=27):
    """Return arrays (SvN[ip], S2[ip]) for all p in ps."""
    N = 2 * n
    comp = np.array([q for q in range(N) if q not in qubits_keep], dtype=int)
    cols = np.concatenate([comp, N + comp])
    basis = gf2_nullspace(G[:, cols])
    d = basis.shape[0]
    np_ = len(ps)
    if d == 0:
        w = float(len(qubits_keep))
        return np.full(np_, w), np.full(np_, w)
    lams = np.array([1.0 - 4.0 * p / 3.0 for p in ps])
    t = gen_xz.shape[1]
    xz = np.zeros((d, t, 2, n), dtype=np.uint8)
    for i in range(d):
        sel = basis[i].astype(bool)
        if sel.any():
            xz[i] = np.bitwise_xor.reduce(gen_xz[sel], axis=0)
    def _supp_size(v):
        return int((v[:, 0, :] | v[:, 1, :]).sum())
    improved = True
    guard = 0
    while improved and guard < 40:
        improved = False; guard += 1
        sizes = [_supp_size(xz[i]) for i in range(d)]
        for i in range(d):
            for j in range(d):
                if i == j: continue
                cand = xz[i] ^ xz[j]
                cs = int((cand[:, 0, :] | cand[:, 1, :]).sum())
                if cs < sizes[i]:
                    xz[i] = cand; sizes[i] = cs; improved = True
    supp = (xz[:, :, 0, :] | xz[:, :, 1, :]).reshape(d, -1).astype(bool)
    parent = list(range(d))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(d):
        for j in range(i + 1, d):
            if (supp[i] & supp[j]).any():
                ri, rj = find(i), find(j)
                if ri != rj: parent[ri] = rj
    comps = {}
    for i in range(d):
        comps.setdefault(find(i), []).append(i)
    H_tot = np.zeros(np_)
    S2_tot = np.zeros(np_)
    for members in comps.values():
        dc = len(members)
        if dc > dmax:
            raise RuntimeError(f"component dim {dc} > dmax={dmax}")
        d_in = min(dc, 20)
        d_out = dc - d_in
        K = np.zeros(2 ** dc, dtype=np.int32)
        for l in range(t):
            X = np.zeros((1, n), dtype=np.uint8)
            Z = np.zeros((1, n), dtype=np.uint8)
            for i in members[:d_in]:
                X = np.concatenate([X, X ^ xz[i, l, 0][None, :]])
                Z = np.concatenate([Z, Z ^ xz[i, l, 1][None, :]])
            for hi in range(2 ** d_out):
                Xo = np.zeros(n, dtype=np.uint8); Zo = np.zeros(n, dtype=np.uint8)
                for k in range(d_out):
                    if (hi >> k) & 1:
                        Xo ^= xz[members[d_in + k], l, 0]
                        Zo ^= xz[members[d_in + k], l, 1]
                blk = slice(hi * 2 ** d_in, (hi + 1) * 2 ** d_in)
                K[blk] += ((X ^ Xo) | (Z ^ Zo)).sum(axis=1, dtype=np.int32)
        Kf = K.astype(float)
        for ip, lam in enumerate(lams):
            c = lam ** Kf
            S2_tot[ip] += np.log2((c * c).sum())
            q = fwht(c)
            q /= 2 ** dc
            q = np.clip(q, 1e-300, None)
            H_tot[ip] += float(-(q * np.log2(q)).sum())
    w = len(qubits_keep)
    return w - d + H_tot, w - S2_tot


def cmi_center_multip(G, gen_xz, n, a, b, ps, memo, dmax):
    w = 2 * a + b
    s = (n - w) // 2
    A = list(range(s, s + a))
    B = list(range(s + a, s + a + b))
    C = list(range(s + a + b, s + w))
    def W(x):
        key = tuple(x)
        if key not in memo:
            memo[key] = window_entropies_multip(G, gen_xz, n, site_qubits(x),
                                                ps, dmax=dmax)
        return memo[key]
    v = [W(x) for x in (A + B, B + C, B, A + B + C)]
    return (v[0][0] + v[1][0] - v[2][0] - v[3][0],
            v[0][1] + v[1][1] - v[2][1] - v[3][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--ps", type=float, nargs="+",
                    default=[0.01, 0.02, 0.04, 0.08])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dmax", type=int, default=27)
    ap.add_argument("--wmax", type=int, default=WMAX)
    ap.add_argument("--wmin", type=int, default=0,
                    help="skip cells with 2a+b < wmin (already computed)")
    ap.add_argument("--csv", type=str, required=True)
    args = ap.parse_args()

    np.random.seed(args.seed)
    n = args.n
    grid = [(a, b) for a in range(1, 6) for b in range(1, 11)
            if args.wmin <= 2 * a + b <= args.wmax]
    acc = {(a, b, ip): [] for (a, b) in grid for ip in range(len(args.ps))}
    nskip = 0
    for r in range(args.reps):
        t0 = time.time()
        layers = random_brickwork_layers(n, args.depth)
        tab = full_tableau(layers, n)
        G = choi_generator_matrix(tab, n)
        gen_xz = pullback_generators(G, layers, n)
        memo = {}
        for (a, b) in grid:
            try:
                ivn, ir2 = cmi_center_multip(G, gen_xz, n, a, b, args.ps,
                                             memo, args.dmax)
            except RuntimeError as e:
                nskip += 1
                print(f"SKIP rep={r} a={a} b={b}: {e}", flush=True)
                continue
            for ip in range(len(args.ps)):
                acc[(a, b, ip)].append((ivn[ip], ir2[ip]))
        print(f"rep {r}: {time.time()-t0:.1f}s", flush=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth", "p", "a", "b", "ivn_mean", "ivn_sem",
                    "ir2_mean", "ir2_sem", "nreps"])
        for (a, b) in grid:
            for ip, p in enumerate(args.ps):
                vals = acc[(a, b, ip)]
                if not vals:
                    continue
                v = np.array(vals)
                m = len(v)
                w.writerow([args.depth, p, a, b,
                            v[:, 0].mean(), v[:, 0].std() / np.sqrt(m),
                            v[:, 1].mean(), v[:, 1].std() / np.sqrt(m), m])
    print(f"wrote {args.csv}  (total SKIPs: {nskip})", flush=True)


if __name__ == "__main__":
    main()
