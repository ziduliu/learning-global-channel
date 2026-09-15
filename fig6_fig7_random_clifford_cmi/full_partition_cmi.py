"""Full-partition CMI for random brickwork Clifford + depolarizing noise.

A|B|C covers the entire chain (a + b + c = n): for each b, sweep the position
of B over all splits (a >= 1, c = n - a - b >= 1) and report the maximum.
By SSA monotonicity this upper-bounds I(A:C|B) for all smaller a, c at the
same B, so exponential decay in b here covers the sup over subsystems in
Definition 1.

Usage: python full_partition_cmi.py <t> <p> <real0> <nreals> [n]
Writes fullpart_n{n}_t{t}_p{p}_r{real0}.csv
"""
import csv
import sys
import time

import numpy as np

import noisy_clifford_cmi as ncc

# NOTE on reproducibility: the two-qubit gates are drawn via
# stim.Tableau.random(), whose RNG cannot be seeded from here; runs are
# therefore statistically i.i.d. but not bit-reproducible. The ir2_max column
# records the Renyi-2 CMI at the split maximizing the von Neumann CMI (it is
# not independently maximized) and is not used in the figures.

t_depth = int(sys.argv[1])
p = float(sys.argv[2])
real0 = int(sys.argv[3])
nreals = int(sys.argv[4])
n = int(sys.argv[5]) if len(sys.argv) > 5 else 12

rows = []
for r in range(real0, real0 + nreals):
    t0 = time.time()
    layers = ncc.random_brickwork_layers(n, t_depth)
    tab = ncc.full_tableau(layers, n)
    G = ncc.choi_generator_matrix(tab, n)
    gen_xz = ncc.pullback_generators(G, layers, n)
    memo = {}
    for b in range(1, n - 1):
        vmax_vn, vmax_r2, a_at_max = -1.0, -1.0, -1
        for a in range(1, n - b):
            c = n - a - b
            ivn, ir2 = ncc.cmi_pair(G, gen_xz, n, 0, a, b, c, p, memo=memo)
            if ivn > vmax_vn:
                vmax_vn, vmax_r2, a_at_max = ivn, ir2, a
        rows.append((t_depth, p, n, r, b, vmax_vn, vmax_r2, a_at_max))
        print(f"real={r} t={t_depth} p={p} b={b:2d}  I_vn^max={vmax_vn:.6e} "
              f"(a={a_at_max})  wall={time.time()-t0:.1f}s", flush=True)

out = f"fullpart_n{n}_t{t_depth}_p{p}_r{real0}.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["depth", "p", "n", "real", "b", "ivn_max", "ir2_max", "a_at_max"])
    w.writerows(rows)
print(f"wrote {len(rows)} rows -> {out}")
