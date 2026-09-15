#!/usr/bin/env python3
"""EXACT vN and Renyi-2 Choi-CMI for random brickwork Clifford circuits with
single-site depolarizing noise (prob p per site after every layer).

Structure theorem: the noisy Choi state is diagonal in the eigenbasis of the
noiseless stabilizer group G.  For any subgroup element g in G_X (supported in
window X), its coefficient is
    c_g = (1 - 4p/3)^{K(g)},
K(g) = total spacetime support of g pulled back through the circuit, summed
over noise layers.  Then, with d = dim G_X and q = WHT(c)/2^d:
    S_vN(X)  = |X| - d + H(q)          (exact eigenvalues!)
    S_2(X)   = |X| - log2( sum_g c_g^2 )
"""
import numpy as np
import stim


LAM = None  # set from p


# --------------------------------------------------------------- circuits
def random_brickwork_layers(n, depth, random_phase=True):
    """Per-layer stim.Tableau; brick phase (even/odd first) randomized per circuit."""
    off = np.random.randint(2) if random_phase else 0
    layers = []
    for layer in range(depth):
        sim = stim.TableauSimulator()
        sim.set_num_qubits(n)
        start = 0 if (layer + off) % 2 == 0 else 1
        for i in range(start, n - 1, 2):
            sim.do_tableau(stim.Tableau.random(2), [i, i + 1])
        layers.append(sim.current_inverse_tableau() ** -1)
    return layers


def full_tableau(layers, n):
    sim = stim.TableauSimulator()
    sim.set_num_qubits(n)
    for T in layers:
        sim.do_tableau(T, list(range(n)))
    return sim.current_inverse_tableau() ** -1


# --------------------------------------------------- stabilizer bookkeeping
def choi_generator_matrix(tab, n):
    """(2n x 4n) binary [x|z] for Choi stabilizers; qubit order ref0,sys0,..."""
    N = 2 * n
    G = np.zeros((2 * n, 2 * N), dtype=np.uint8)
    for i in range(n):
        for kind, out in ((0, tab.x_output(i)), (1, tab.z_output(i))):
            row = 2 * i + kind
            if kind == 0:
                G[row, 2 * i] ^= 1                    # X on ref_i
            else:
                G[row, N + 2 * i] ^= 1                # Z on ref_i
            for j in range(n):
                pj = out[j]
                if pj == 1:   G[row, 2 * j + 1] ^= 1
                elif pj == 2: G[row, 2 * j + 1] ^= 1; G[row, N + 2 * j + 1] ^= 1
                elif pj == 3: G[row, N + 2 * j + 1] ^= 1
    return G


def gf2_nullspace(M):
    """Basis of {v: v M = 0} over GF(2).  M: (rows x cols)."""
    M = M.copy() % 2
    rows, cols = M.shape
    aug = np.concatenate([M, np.eye(rows, dtype=np.uint8)], axis=1)
    rank = 0
    for c in range(cols):
        piv = None
        for r in range(rank, rows):
            if aug[r, c]:
                piv = r; break
        if piv is None:
            continue
        aug[[rank, piv]] = aug[[piv, rank]]
        mask = aug[:, c].astype(bool).copy(); mask[rank] = False
        aug[mask] ^= aug[rank]
        rank += 1
        if rank == rows:
            break
    return aug[rank:, cols:]        # combos with zero projection


def sys_pauli_of_combo(G_rows, combo, n):
    """stim.PauliString of the SYSTEM part of the group element given by combo."""
    N = 2 * n
    v = np.bitwise_xor.reduce(G_rows[combo.astype(bool)], axis=0) if combo.any() \
        else np.zeros(2 * N, dtype=np.uint8)
    ps = ["_"] * n
    for j in range(n):
        x, z = v[2 * j + 1], v[N + 2 * j + 1]
        ps[j] = "_XZY"[x + 2 * z] if (x or z) else "_"
    # map: x=1,z=0->X ; x=0,z=1->Z ; x=1,z=1->Y
    s = "".join({(0, 0): "_", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}[
        (int(v[2*j+1]), int(v[N+2*j+1]))] for j in range(n))
    return stim.PauliString(s)




def pullback_generators(G, layers, n):
    """Pull the SYSTEM part of all 2n Choi generators back through the circuit
    ONCE.  Returns gen_xz[g, layer, 2, n] support bit masks (x,z)."""
    t = len(layers)
    inv = [T.inverse() for T in layers]
    gen_xz = np.zeros((2 * n, t, 2, n), dtype=np.uint8)
    unit = np.zeros(2 * n, dtype=np.uint8)
    for g in range(2 * n):
        unit[:] = 0; unit[g] = 1
        P = sys_pauli_of_combo(G, unit, n)
        for l in range(t - 1, -1, -1):
            xs, zs = P.to_numpy()
            gen_xz[g, l, 0] = xs.astype(np.uint8)
            gen_xz[g, l, 1] = zs.astype(np.uint8)
            P = inv[l](P)
    return gen_xz

# ------------------------------------------------------- exact noisy entropy
def window_entropies(G, gen_xz, n, qubits_keep, p, dmax=26):
    """Return (S_vN, S_2) of the noisy Choi reduced state on qubits_keep."""
    N = 2 * n
    comp = np.array([q for q in range(N) if q not in qubits_keep], dtype=int)
    cols = np.concatenate([comp, N + comp])
    basis = gf2_nullspace(G[:, cols])            # combos localized in window
    d = basis.shape[0]
    if d == 0:
        return float(len(qubits_keep)), float(len(qubits_keep))
    lam = 1.0 - 4.0 * p / 3.0
    t = gen_xz.shape[1]
    # basis-element support masks = XOR of generator pullback masks (linear!)
    xz = np.zeros((d, t, 2, n), dtype=np.uint8)
    for i in range(d):
        sel = basis[i].astype(bool)
        if sel.any():
            xz[i] = np.bitwise_xor.reduce(gen_xz[sel], axis=0)
    # EXACT component decomposition: basis elements whose spacetime supports
    # are disjoint have additive K => the sign distribution factorizes exactly.
    # support-minimizing basis sweep: greedily replace b_i by b_i^b_j when it
    # shrinks spacetime support -- splits block-decomposable groups (exact:
    # any basis of the same subgroup gives identical entropies).
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
    supp = (xz[:, :, 0, :] | xz[:, :, 1, :]).reshape(d, -1).astype(bool)  # (d, t*n)
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
    H_tot, S2_tot = 0.0, 0.0
    for members in comps.values():
        dc = len(members)
        if dc > dmax:
            raise RuntimeError(f"component dim {dc} > dmax={dmax}")
        # chunked exact enumeration: inner d_in bits vectorized, outer looped
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
        c = lam ** K.astype(float)
        S2_tot += np.log2((c * c).sum())
        q = c.copy()
        h = 1
        while h < len(q):
            for start in range(0, len(q), 2 * h):
                a_ = q[start:start + h].copy()
                b_ = q[start + h:start + 2 * h].copy()
                q[start:start + h] = a_ + b_
                q[start + h:start + 2 * h] = a_ - b_
            h *= 2
        q /= 2 ** dc
        q = np.clip(q, 1e-300, None)
        H_tot += float(-(q * np.log2(q)).sum())
    S2 = len(qubits_keep) - S2_tot
    SvN = len(qubits_keep) - d + H_tot
    return SvN, S2


def site_qubits(sites):
    out = []
    for s in sites:
        out += [2 * s, 2 * s + 1]
    return out


def cmi_pair(G, gen_xz, n, s, a, b, c, p, memo=None):
    A = list(range(s, s + a)); B = list(range(s + a, s + a + b))
    C = list(range(s + a + b, s + a + b + c))
    def W(x):
        key = tuple(x)
        if memo is not None and key in memo:
            return memo[key]
        v = window_entropies(G, gen_xz, n, site_qubits(x), p)
        if memo is not None:
            memo[key] = v
        return v
    vals = [W(x) for x in (A + B, B + C, B, A + B + C)]
    I_vn = vals[0][0] + vals[1][0] - vals[2][0] - vals[3][0]
    I_r2 = vals[0][1] + vals[1][1] - vals[2][1] - vals[3][1]
    return I_vn, I_r2


# ----------------------------------------------------------- dense self-test
def dense_selftest(p=0.1, reps=6):
    n = 5
    P1 = {"X": np.array([[0, 1], [1, 0]], dtype=complex),
          "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
          "Z": np.diag([1.0, -1.0]).astype(complex)}

    def op_on(op, q, nq):
        out = np.eye(1, dtype=complex)
        for i in range(nq):
            out = np.kron(out, op if i == q else np.eye(2))
        return out
    for r in range(reps):
        depth = 1 + r % 3
        layers = random_brickwork_layers(n, depth)
        tab = full_tableau(layers, n)
        G = choi_generator_matrix(tab, n)
        # dense noisy Choi on 2n qubits (ref_i=2i, sys_i=2i+1), little endian kron order
        N = 2 * n
        psi = np.zeros(2 ** N, dtype=complex); psi[0] = 1.0
        # Bell pairs: H on ref, CNOT ref->sys  (build via stim for simplicity)
        sv = stim.TableauSimulator(); sv.set_num_qubits(N)
        for i in range(n):
            sv.h(2 * i); sv.cnot(2 * i, 2 * i + 1)
        psi = np.array(sv.state_vector(endian="big"), dtype=complex)
        rho = np.outer(psi, psi.conj())
        for l, T in enumerate(layers):
            U2n = np.eye(2 ** N, dtype=complex)
            # embed layer unitary on sys qubits: build via stim on 2n qubits
            svl = stim.TableauSimulator(); svl.set_num_qubits(N)
            start = 0 if l % 2 == 0 else 1
            # reconstruct layer by conjugating basis: easier to re-sample?? ->
            # instead: get unitary of T on n qubits, embed on sys positions
            Un = T.to_unitary_matrix(endian="big")   # 2^n x 2^n
            # embed: qubit j of T -> qubit 2j+1 of the 2n register (big endian)
            # build permutation between (ref,sys) interleaved and (ref-block, sys-block)
            # simpler: apply via tensor reshape
            rho_t = rho.reshape([2] * (2 * N))
            # apply Un on sys ket indices then dagger on bra indices
            sys_axes = [2 * j + 1 for j in range(n)]
            Unt = Un.reshape([2] * (2 * n))
            # ket side
            NEW = list(range(2 * N, 2 * N + n))
            rho_t = np.einsum(Unt, NEW + sys_axes,
                              rho_t, list(range(2 * N)),
                              [x if x not in sys_axes else NEW[sys_axes.index(x)]
                               for x in range(2 * N)])
            # bra side
            bra_axes = [N + 2 * j + 1 for j in range(n)]
            rho_t = np.einsum(np.conj(Unt), NEW + bra_axes,
                              rho_t, list(range(2 * N)),
                              [x if x not in bra_axes else NEW[bra_axes.index(x)]
                               for x in range(2 * N)])
            rho = rho_t.reshape(2 ** N, 2 ** N)
            for j in range(n):                       # depolarize sys qubit j
                q = 2 * j + 1
                acc = (1 - p) * rho
                for Pm in P1.values():
                    O = op_on(Pm, q, N)
                    acc += (p / 3) * (O @ rho @ O.conj().T)
                rho = acc
        for (s, a, b, c) in [(0, 1, 1, 1), (1, 1, 2, 1), (0, 1, 3, 1)]:
            if s + a + b + c > n:
                continue
            gen_xz = pullback_generators(G, layers, n)
            I_vn, I_r2 = cmi_pair(G, gen_xz, n, s, a, b, c, p)
            # dense
            def dS(keep, alpha):
                idx = site_qubits(keep)
                rest = [x for x in range(N) if x not in idx]
                rr = np.einsum(rho.reshape([2] * (2 * N)),
                               list(range(2 * N)),
                               idx + [x + N for x in idx] if False else
                               idx + [(x + N) for x in idx])
                # simpler: partial trace via reshape
                perm = idx + rest
                rt = rho.reshape([2] * (2 * N))
                rt = np.transpose(rt, perm + [x + N for x in perm])
                dk = 2 ** len(idx); dr = 2 ** len(rest)
                rt = rt.reshape(dk, dr, dk, dr)
                rw = np.einsum("arbr->ab", rt)
                w = np.linalg.eigvalsh(rw); w = w[w > 1e-12]
                if alpha == 1:
                    return float(-(w * np.log2(w)).sum())
                return float(-np.log2((w ** 2).sum()))
            def dcmi(alpha):
                A = list(range(s, s + a)); B = list(range(s + a, s + a + b))
                C = list(range(s + a + b, s + a + b + c))
                return (dS(A + B, alpha) + dS(B + C, alpha)
                        - dS(B, alpha) - dS(A + B + C, alpha))
            assert abs(I_vn - dcmi(1)) < 1e-6, ("vN", I_vn, dcmi(1), s, a, b, c)
            assert abs(I_r2 - dcmi(2)) < 1e-6, ("R2", I_r2, dcmi(2), s, a, b, c)
    print(f"SELFTEST PASSED: exact formula == dense channel simulation "
          f"(vN and Renyi-2, {reps} circuits, p={p})", flush=True)


# ------------------------------------------------------------------- main
def main():
    import argparse, csv, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--depths", type=int, nargs="+", default=[1,2,3,4,5,6,7,8,9,10])
    ap.add_argument("--bvals", type=int, nargs="+", default=list(range(1, 15)))
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--margin", type=int, default=2)
    ap.add_argument("--dmax", type=int, default=24)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--csv", type=str, default="noisy_clifford_cmi.csv")
    args = ap.parse_args()
    if args.selftest:
        dense_selftest(p=args.p if args.p > 0 else 0.1)
    n = args.n; a = c = 1
    rows = []
    for depth in args.depths:
        t0 = time.time()
        acc_vn = {b: [] for b in args.bvals}
        acc_r2 = {b: [] for b in args.bvals}
        for r in range(args.reps):
            layers = random_brickwork_layers(n, depth)   # ONE circuit for all b
            tab = full_tableau(layers, n)
            G = choi_generator_matrix(tab, n)
            gen_xz = pullback_generators(G, layers, n)
            memo = {}
            for b in args.bvals:
                w = a + b + c
                best_vn, best_r2 = -np.inf, -np.inf
                for s in range(args.margin, n - w - args.margin + 1):
                    try:
                        I_vn, I_r2 = cmi_pair(G, gen_xz, n, s, a, b, c, args.p, memo=memo)
                    except RuntimeError as e:
                        print(f"SKIP depth={depth} b={b} s={s}: {e}", flush=True)
                        continue
                    best_vn = max(best_vn, I_vn); best_r2 = max(best_r2, I_r2)
                if np.isfinite(best_vn):
                    acc_vn[b].append(best_vn); acc_r2[b].append(best_r2)
        for b in args.bvals:
            if not acc_vn[b]: continue
            va, ra = np.array(acc_vn[b]), np.array(acc_r2[b])
            rows.append((depth, b, args.p, va.mean(), va.std()/np.sqrt(len(va)),
                         ra.mean(), ra.std()/np.sqrt(len(ra))))
            print(f"depth={depth} b={b}: <maxI_vN>={va.mean():.5f} "
                  f"<maxI_R2>={ra.mean():.5f}  ({len(va)} reps)", flush=True)
        print(f"  [depth {depth}: {time.time()-t0:.1f}s]", flush=True)
        with open(args.csv, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["depth","b","p","ivn_mean","ivn_sem","ir2_mean","ir2_sem"])
            wcsv.writerows(rows)
        np.savez_compressed(args.csv.replace(".csv", "_raw.npz"),
                            bvals=np.array(args.bvals),
                            ivn=np.array([acc_vn[b] for b in args.bvals], dtype=object),
                            ir2=np.array([acc_r2[b] for b in args.bvals], dtype=object),
                            allow_pickle=True)
    print("wrote", args.csv)


if __name__ == "__main__":
    main()
