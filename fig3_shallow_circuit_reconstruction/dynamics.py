import numpy as np
from typing import List, Tuple

# ---- OQuPy ----
from oqupy.system import SystemChain
from oqupy.operators import sigma
from oqupy.mps_mpo import compute_tebd_propagator

# =========================
# =========================

def identity_site_kernel(d2: int):
    I = np.eye(d2, dtype=complex)
    return I.reshape(1, d2, d2, 1)

def identity_mpo(L: int, d2: int) -> List[np.ndarray]:
    return [identity_site_kernel(d2) for _ in range(L)]

def layer_to_chain_mpo(layer, L: int, d2: int) -> List[np.ndarray]:
    mpo = identity_mpo(L, d2)
    for gate in layer.gates:
        sites = gate.sites
        tensors = gate.tensors

        if len(sites) == 1:
            t = tensors[0].reshape(1, d2, d2, 1)
            mpo[sites[0]] = t

        elif len(sites) == 2:
            TL, TR = tensors
            TL4 = TL.reshape(1, d2, d2, TL.shape[-1])
            TR4 = TR.reshape(TR.shape[0], d2, d2, 1)
            i = sites[0]
            assert sites[1] == i + 1
            mpo[i]     = TL4
            mpo[i + 1] = TR4
        else:
            raise NotImplementedError("only nearest-neighbour gates are supported")
    return mpo

def compose_mpo(B_after: List[np.ndarray],
                A_before: List[np.ndarray],
                max_bond: int = None,
                cutoff: float = 0.0) -> List[np.ndarray]:
    L = len(A_before)
    C = []
    for s in range(L):
        A = A_before[s]  # (La, Ia, Oa, Ra)
        B = B_after[s]   # (Lb, Ib, Ob, Rb)
        La, Ia, Oa, Ra = A.shape
        Lb, Ib, Ob, Rb = B.shape
        assert Ia == Ib and Oa == Ib == Ia, "Liouville dimension must be the same on every site"
        T = np.tensordot(A, B, axes=([2], [1]))   # (La, Ia, Ra, Lb, Ob, Rb)
        T = np.transpose(T, (0, 3, 1, 4, 2, 5))   # (La,Lb, Ia, Ob, Ra, Rb)
        T = T.reshape(La*Lb, Ia, Ob, Ra*Rb)       # (L, I, O, R)
        C.append(T)

    #if (max_bond is not None) or (cutoff > 0.0):
    #    C = compress_mpo(C, max_bond=max_bond, cutoff=cutoff)
    return C

def compress_mpo(mpo: List[np.ndarray],
                 max_bond: int = 256,
                 cutoff: float = 1e-12) -> List[np.ndarray]:
    L = len(mpo)
    for s in range(L - 1):
        A = mpo[s]
        La, Ia, Oa, Ra = A.shape
        A_mat = A.reshape(La * Ia * Oa, Ra)
        U, S, Vh = np.linalg.svd(A_mat, full_matrices=False)

        if cutoff is not None and cutoff > 0.0:
            rel_thr = cutoff * (S[0] if S.size else 1.0)
            keep = (S > rel_thr)
        else:
            keep = np.ones_like(S, dtype=bool)

        chi = int(min(np.sum(keep), max_bond if max_bond else S.size))
        U = U[:, :chi]
        S = S[:chi]
        Vh = Vh[:chi, :]

        mpo[s] = U.reshape(La, Ia, Oa, chi)

        right = mpo[s + 1]                  # (Lr, Ir, Or, Rr)
        Lr, Ir, Or, Rr = right.shape
        assert Lr == Ra
        SV = (S[:, None] * Vh).reshape(chi, Ra)    # (chi, Ra)
        # (chi, Ra) ⋅ (Ra, Ir, Or, Rr) = (chi, Ir, Or, Rr)
        new_right = np.tensordot(SV, right, axes=(1, 0))
        mpo[s + 1] = new_right
    return mpo

def full_tebd_step_mpo(tebd, L: int, dloc: int,
                       max_bond: int = 512,
                       cutoff: float = 1e-12) -> List[np.ndarray]:
    d2 = dloc * dloc
    full = identity_mpo(L, d2)
    for layer in tebd.gate_layers:
        layer_mpo = layer_to_chain_mpo(layer, L, d2)
        full = compose_mpo(layer_mpo, full, max_bond=max_bond, cutoff=cutoff)
    return full

# =========================
# =========================

def build_chain(L=6, dloc=2, Jx=1.0, Jy=1.0, Jz=1.0, gamma=0.2):
    chain = SystemChain([dloc]*L)
    for i in range(L-1):
        if Jx: chain.add_nn_hamiltonian(i, Jx*sigma('x'), sigma('x'))
        if Jy: chain.add_nn_hamiltonian(i, Jy*sigma('y'), sigma('y'))
        if Jz: chain.add_nn_hamiltonian(i, Jz*sigma('z'), sigma('z'))
    for i in range(L-1):
        if gamma: chain.add_site_dissipation(i, gamma*sigma('x'))
        if gamma: chain.add_site_dissipation(i, gamma*sigma('y'))
        if gamma: chain.add_site_dissipation(i, gamma*sigma('z'))
    return chain

def to_dense_superop_small(mpo: List[np.ndarray]) -> np.ndarray:
    T = mpo[0]  # (1, I, O, R)
    for s in range(1, len(mpo)):
        T = np.tensordot(T, mpo[s], axes=(3, 0))
    T = np.transpose(T, (1, 2, 0, 3))  # (I^L, O^L, 1, 1)
    T = T.reshape(T.shape[1], T.shape[0]).T
    return T
def super_to_choi_site(A, d, phys_axes=None):
    A = np.asarray(A)
    if A.ndim != 4:
        raise ValueError("Expect rank-4 site tensor, got shape {}".format(A.shape))
    ds = d * d
    shp = list(A.shape)

    if phys_axes is None:
        phys = [ax for ax, s in enumerate(shp) if s == ds]
        if len(phys) != 2:
            raise ValueError(
                f"Can't infer physical axes: found dims=={ds} at {phys}, "
                "please pass phys_axes=(u_axis, d_axis)."
            )
        u_ax, d_ax = phys
    else:
        u_ax, d_ax = phys_axes

    bond_axes = [ax for ax in range(4) if ax not in (u_ax, d_ax)]
    A_lrud = np.moveaxis(A, bond_axes + [u_ax, d_ax], [0, 1, 2, 3])  # (Dl, Dr, ds, ds)
    Dl, Dr, _, _ = A_lrud.shape

    A6 = A_lrud.reshape(Dl, Dr, d, d, d, d)          # (l, r,  i, j,  k, l)
    B6 = A6.transpose(0, 1, 2, 4, 3, 5)              # (l, r,  i, k,  j, l)
    B_lrud = B6.reshape(Dl, Dr, ds, ds)              # (l, r, ds, ds)

    B = np.moveaxis(B_lrud, [0, 1, 2, 3], bond_axes + [u_ax, d_ax])
    return B


def super_to_choi_mpo(arrays, d, phys_axes=None):
    return [super_to_choi_site(A, d, phys_axes=phys_axes) for A in arrays]
def get_W_list(mpo):
    return mpo._W

def tenpy_super_to_choi_mpo(mpo_super, d, phys_axes=None, inplace=False):
    new_mpo = mpo_super
    W_list = new_mpo
    for i in range(len(W_list)):
        A_np = W_list[i]
        A_new = super_to_choi_site(A_np, d, phys_axes=phys_axes)
        W_list[i] = A_new
    return new_mpo
