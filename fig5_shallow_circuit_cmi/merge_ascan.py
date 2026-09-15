"""Merge the per-task JSONs of ascan_cluster.py into circuit_cmi_ascan.json.

Usage: python merge_ascan.py [results_dir]   (default: ascan_results)
Aggregates mean and SEM over realizations per (eps, a, b).
"""
import glob
import json
import sys

import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "ascan_results"
raw = {}
for f in glob.glob(f"{d}/ascan_r*_e*.json"):
    for v in json.load(open(f)).values():
        raw.setdefault((v["eps"], v["a"], v["b"]), []).append(v["cmi"])
out = {}
for (eps, a, b), vals in sorted(raw.items()):
    out[f"e{eps}_a{a}_b{b}"] = dict(
        eps=eps, a=a, b=b, mean=float(np.mean(vals)),
        sem=float(np.std(vals) / np.sqrt(len(vals))), nreals=len(vals))
json.dump(out, open("circuit_cmi_ascan.json", "w"), indent=1)
print(f"merged {len(out)} combos from {len(glob.glob(f'{d}/*.json'))} task files")
