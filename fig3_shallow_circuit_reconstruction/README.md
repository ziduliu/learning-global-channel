# Fig. 3 — reconstruction of a noisy shallow circuit (Sec. IV.B)

Channel Λ = (⊗_j D_γ) ∘ U_ε ∘ U on n = 20 qubits: a layer U of iSWAP gates on the bonds
(0,1), (2,3), …, a coherent perturbation U_ε = ⊗_odd e^{−iεH_i} ∘ ⊗_even e^{−iεH_i} with
H_i = α₁XX + α₂YY + α₃ZZ and α_k uniform in [0,1) on every bond, and single-qubit
Z dephasing of strength γ. The Choi state is built as an MPO from OQuPy propagators and
reconstructed from its three-site window marginals (w = 1) with CVXPY/SCS.

## Files

| File | Role |
|---|---|
| `main.py` | `python main.py k` runs γ = `np.linspace(0, 0.01, 10)[k]` together with the γ = 0 reference channel. The ratio r = ε/γ is set by `eta_x, eta_y, eta_z = [r, r, r]`; the figure uses r = 1, 2, 4, 8 |
| `_functions_.py`, `dynamics.py`, `data_processing.py` | helper routines (MPO composition and marginals, `mpo_grow`, CPTP constraint, overlaps) |
| `job.sh` | SLURM array job (array range 0–9) |

## Output

`main.py` stores the exact Choi MPO (`mpo_ref_{k}_*.npy`) and the reconstruction
(`mpo_out_{k}_*.npy`) and prints 1 − F with F = Tr[J(Λ_{U,ε,γ}) J(Λ_U)]. The purity of
panel (c) is Tr[J J†] evaluated on these MPOs. The MPO files are large and are kept on the
cluster; they are available on request.
