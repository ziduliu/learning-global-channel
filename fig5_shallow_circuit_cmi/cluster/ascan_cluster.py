"""a-dependence scan, cluster array version: one task = one (real, eps) pair.

Usage: python ascan_cluster.py <task_id>   with task_id = real * len(EPS) + eps_idx
Writes ascan_results/ascan_r{real}_e{eps}.json
Merge locally with the plotting script.
"""
import json, os, sys, time
import numpy as np
import scipy.linalg as sla
import circuit_cmi as cc

def window_rdm_split(mps, keep):
    """Reduced Choi state on contiguous window; split contraction, low memory."""
    keep = sorted(keep)
    i0, i1 = keep[0], keep[-1]
    tr = np.eye(4, dtype=complex).reshape(16)
    L = np.ones((1,), dtype=complex)
    for i in range(i0):
        L = L @ np.tensordot(mps[i], tr, axes=([1], [0]))
    R = np.ones((1,), dtype=complex)
    for i in range(len(mps) - 1, i1, -1):
        R = np.tensordot(mps[i], tr, axes=([1], [0])) @ R
    k = len(keep)
    k1 = k // 2 + k % 2
    A = L
    for j, i in enumerate(keep[:k1]):
        A = np.tensordot(A, mps[i], axes=([-1], [0]))
        A = A.reshape(-1, A.shape[-1])
    B = R
    for i in reversed(keep[k1:]):
        B = np.tensordot(mps[i], B, axes=([2], [0]))
        B = B.reshape(B.shape[0], -1)
    res = (A @ B).reshape([16] * k)
    res = res.reshape([4, 4] * k)
    rows = list(range(0, 2 * k, 2)); cols = list(range(1, 2 * k, 2))
    return res.transpose(rows + cols).reshape(4 ** k, 4 ** k)

def vn_lean(rho, tol=1e-13):
    w = sla.eigvalsh(rho, overwrite_a=True, check_finite=False)
    w = w[w > tol]; w = w / w.sum()
    return float(-(w * np.log2(w)).sum())

def cmi_split(mps, s0, a, b, c):
    val = 0.0
    for keep, sign in ((list(range(s0, s0+a+b)), 1),
                       (list(range(s0+a, s0+a+b+c)), 1),
                       (list(range(s0+a, s0+a+b)), -1),
                       (list(range(s0, s0+a+b+c)), -1)):
        r = window_rdm_split(mps, keep)
        val += sign * vn_lean(r)
        del r
    return val

# --- runtime self-check: split contraction vs validated pipeline (small window)
_m = cc.build_choi(8, 0.05, 0.01, seed=7)
for _keep in ([1, 2, 3], [2, 3, 4, 5]):
    _r1 = cc.window_rdm(_m, _keep); _r2 = window_rdm_split(_m, _keep)
    assert np.max(np.abs(_r1 - _r2)) < 1e-12, "split contraction mismatch!"
_c1 = cc.cmi_window(_m, 1, 1, 2, 1); _c2 = cmi_split(_m, 1, 1, 2, 1)
assert abs(_c1 - _c2) < 1e-10, (_c1, _c2)
print("[selfcheck] split contraction matches validated pipeline", flush=True)

N, GAMMA, REALS = 20, 0.01, 10
EPS = [round(0.01 * k, 2) for k in range(1, 11)]
COMBOS = [(1, 1), (2, 1), (3, 1), (1, 3), (2, 3)]   # (a, b), c = a

task = int(sys.argv[1])
r, ei = divmod(task, len(EPS))
eps = EPS[ei]
assert r < REALS

t0 = time.time()
mps = cc.build_choi(N, eps, GAMMA, seed=1234 + r)
out = {}
for (a, b) in COMBOS:
    w = a + b + a
    s0 = (N - w) // 2
    # max over central window positions, matching the production b-scan (I_J^max)
    val = max(cmi_split(mps, s, a, b, a) for s in (s0 - 1, s0, s0 + 1))
    out[f"a{a}_b{b}"] = dict(eps=eps, real=r, a=a, b=b, cmi=val)
    print(f"real={r} eps={eps} a={a} b={b} cmi={out[f'a{a}_b{b}']['cmi']:.4e} "
          f"({time.time()-t0:.0f}s)", flush=True)

os.makedirs("ascan_results", exist_ok=True)
json.dump(out, open(f"ascan_results/ascan_r{r}_e{ei}.json", "w"), indent=1)
print(f"done task {task} wall={time.time()-t0:.1f}s")
