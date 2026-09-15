"""Recompute the b = 4 entries of the dense eps-scan with the same
three-position maximum used for b <= 3 (fixes the convention inconsistency
where b = 4 was evaluated at the central window only).

Updates circuit_cmi_dense_eps.json in place (backup written first).
"""
import json
import shutil
import time

import numpy as np

import circuit_cmi as cc

N, GAMMA, REALS, B = 20, 0.01, 10, 4
EPS = [round(0.01 * k, 2) for k in range(1, 11)]

shutil.copy("circuit_cmi_dense_eps.json", "circuit_cmi_dense_eps.json.bak")
res = json.load(open("circuit_cmi_dense_eps.json"))

acc = {e: [] for e in EPS}
for r in range(REALS):
    t0 = time.time()
    for eps in EPS:
        mps = cc.build_choi(N, eps, GAMMA, seed=1234 + r)
        w = 1 + B + 1
        s0 = (N - w) // 2
        val = max(cc.cmi_window(mps, s, 1, B, 1) for s in (s0 - 1, s0, s0 + 1))
        acc[eps].append(val)
    print(f"real={r} done  wall={time.time()-t0:.1f}s", flush=True)

for eps in EPS:
    v = acc[eps]
    key = f"g{GAMMA}_e{eps}_b{B}"
    old = res[key]["mean"]
    res[key] = dict(gamma=GAMMA, eps=eps, b=B, mean=float(np.mean(v)),
                    sem=float(np.std(v) / np.sqrt(len(v))), nreals=len(v))
    print(f"eps={eps}: {old:.3e} -> {res[key]['mean']:.3e}")

json.dump(res, open("circuit_cmi_dense_eps.json", "w"), indent=1)
print("updated circuit_cmi_dense_eps.json (backup: .bak)")
