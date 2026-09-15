import numpy as np
import quimb as qu
import quimb.tensor as qtn
import numpy as np
import quimb.tensor as qtn
import cvxpy as cp
def mps_projector_to_mpo(mps):
    arrays = []
    for i in range(mps.L):
        Ti = mps[i]
        #print(Ti.shape)

        if i == 0:
            Dr, d = Ti.shape
        elif i == mps.L-1:
            Dl, d = Ti.shape
        else:
            Dl, Dr, d = Ti.shape

        A = Ti.data                  # (Dl, Dr, d)
        B = A.conj()                 # (Dl, Dr, d)

        if i == 0:
            W = np.einsum('rd,RD->rRdD', A, B).reshape(Dr*Dr, d, d)
        elif i == mps.L-1:
            W = np.einsum('ld,LD->lLdD', A, B).reshape(Dl*Dl, d, d)
        else:
            W = np.einsum('lrd,LRD->lLrRdD', A, B).reshape(Dl*Dl, Dr*Dr, d, d)

        arrays.append(W)

    return qtn.MatrixProductOperator(
        arrays,
        shape='lrud',
        upper_ind_id='k{}',
        lower_ind_id='b{}',
        site_tag_id=mps.site_tag_id,
    )

def generate_mps(N, g):
    """
    Build an MPS suitable for quimb:
      bulk tensor axes: (left, phys, right)
      left boundary: (phys, right), right boundary: (left, phys)
    """
    A1 = np.array([[0, 0],
                   [1, 1]], dtype=complex)
    A2 = np.array([[1, g],
                   [0, 0]], dtype=complex)

    vL = 1/2*np.array([1, 1], dtype=complex)
    vR = 1/2*np.array([1, 1], dtype=complex)

    # start as (phys, left, right) then permute to (left, phys, right)
    T_bulk = np.stack([A1, A2], axis=0)            # (phys, left, right)
    T_bulk = np.transpose(T_bulk, (1, 0, 2))       # -> (left, phys, right)

    # boundaries
    T_left  = np.tensordot(vL, T_bulk, axes=([0], [0]))  # (phys, right)
    T_right = np.tensordot(T_bulk, vR, axes=([2], [0]))  # (left, phys)

    tensors = [T_left] + [T_bulk] * (N - 2) + [T_right]
    mps = qtn.MatrixProductState(tensors)
    return mps


def mpo_dense_on_sites(mps, keep_sites):
    mpo = mps.partial_trace_to_mpo(keep=keep_sites)
    return mpo.to_dense(), mpo

def mpo_partial_trace(mpo, keep):
    
    
    tn = mpo.copy()
    for i in range(tn.L):
        #print(i)
        if i not in keep:
            bra, ket = tn[i].inds[-2:]
            tn[i].reindex_({bra:ket})
    mpo_data  = (tn^all).data
    phys_d = mpo_data.shape[0]

    sites = len(mpo_data.shape)//phys_d

    x1 = np.arange(0,2*sites,2)
    x2 = np.arange(1,2*sites,2)
    #print(np.append(x1,x2))
    mpo_data = mpo_data.transpose(np.append(x1,x2))
    
    return mpo_data.reshape([phys_d**sites, phys_d**sites])



def dims_from_abc(a, b, c, dloc=2):
    dA = dloc ** a
    dB = dloc ** b
    dBout = dloc ** b
    dC = dloc ** c
    dOut = dBout * dC
    return dA, dB, dBout, dC, dOut

def ptr_out_trace_preserving_constraint(J, d_out, d_in):
    blocks_sum = 0
    for o in range(d_out):
        r = slice(o * d_in, (o + 1) * d_in)
        c = slice(o * d_in, (o + 1) * d_in)
        blocks_sum = blocks_sum + J[r, c]
    return blocks_sum

def blockify_rho_AB(rho_AB, dA, dB):
    M = rho_AB.reshape(dA, dB, dA, dB)      # (a, i, a', i')
    return [[M[:, i, :, ip] for ip in range(dB)] for i in range(dB)]

def J_block(J, d_out, d_in, i, ip):
    return J[i::d_in, ip::d_in]
def _pow_psd(M, z: complex, tol=1e-12):
    M = (M + M.conj().T) / 2
    w, U = np.linalg.eigh(M)
    keep = w > tol
    if not np.any(keep):
        return np.zeros_like(M, dtype=complex)
    return (U[:, keep] * (w[keep] ** z)) @ U[:, keep].conj().T

def _hermitize_trace1(R):
    R = (R + R.conj().T) / 2
    tr = np.trace(R).real
    return R / tr if tr != 0 else R


def rotated_petz_B_to_BC(rho_AB, rho_B, rho_BC, t: float, tol=1e-12):
    dB = rho_B.shape[0]
    dAB = rho_AB.shape[0]; dA = dAB // dB
    dBC = rho_BC.shape[0]; dC = dBC // dB
    IA, IC = np.eye(dA, dtype=complex), np.eye(dC, dtype=complex)

    aL = 0.5 + 0.5j*t  # (1+it)/2
    aR = 0.5 - 0.5j*t  # (1-it)/2
    bL = -0.5 - 0.5j*t # -(1+it)/2
    bR = -0.5 + 0.5j*t # -(1-it)/2

    SBC_L = np.kron(IA, _pow_psd(rho_BC, aL, tol))
    SBC_R = np.kron(IA, _pow_psd(rho_BC, aR, tol))
    B_L   = np.kron(np.kron(IA, _pow_psd(rho_B, bL, tol)), IC)
    B_R   = np.kron(np.kron(IA, _pow_psd(rho_B, bR, tol)), IC)

    X = np.kron(rho_AB.astype(complex), IC)
    rho_rec = SBC_L @ B_L @ X @ B_R @ SBC_R
    return _hermitize_trace1(rho_rec)

import numpy as np


def _beta0(t):
    # β0(t) = π / (2*(cosh(π t) + 1))
    return 0.5*np.pi / (np.cosh(np.pi*t) + 1.0)
def universal_rotated_petz(rho_AB, rho_B, rho_BC, T=10.0, n=601, tol=1e-12):
    ts = np.linspace(-T, T, int(n))
    beta = _beta0(ts)

    trap = np.ones_like(beta, dtype=float)
    trap[0] = trap[-1] = 0.5
    w = beta * trap
    w = w / w.sum()

    dAB = rho_AB.shape[0]; dB = rho_B.shape[0]; dBC = rho_BC.shape[0]
    assert dAB % dB == 0 and dBC % dB == 0
    dA = dAB // dB; dC = dBC // dB

    rho_sum = np.zeros((dA*dB*dC, dA*dB*dC), dtype=complex)
    for t, wt in zip(ts, w):
        rho_sum += wt * rotated_petz_at_t(rho_AB, rho_B, rho_BC, t, tol=tol)

    return _hermitize_trace1(rho_sum)
def _beta0(t):
    # β0(t) = π / (2*(cosh(π t) + 1))
    return 0.5*np.pi / (np.cosh(np.pi*t) + 1.0)

def universal_rotated_petz(rho_AB, rho_B, rho_BC, T=10.0, n=601, tol=1e-12):
    ts = np.linspace(-T, T, n)
    w = _beta0(ts)
    Z = np.trapz(w, ts); w = w / Z

    dAB = rho_AB.shape[0]; dB = rho_B.shape[0]; dBC = rho_BC.shape[0]
    dA = dAB // dB; dC = dBC // dB
    rho_sum = np.zeros((dA*dB*dC, dA*dB*dC), dtype=complex)
    for t, wt in zip(ts, w):
        rho_sum += wt * rotated_petz_B_to_BC(rho_AB, rho_B, rho_BC, t, tol)
    return _hermitize_trace1(rho_sum)

def fro_error(rho, sigma):
    return np.linalg.norm(rho - sigma, 'fro')

def fidelity(rho, sigma, tol=1e-12):
    # Uhlmann F(ρ,σ) = (Tr √(√ρ σ √ρ))^2
    def _sqrt_psd(M):
        w, U = np.linalg.eigh((M+M.conj().T)/2)
        w = np.clip(w, 0, None)
        return (U * np.sqrt(w)) @ U.conj().T
    S = _sqrt_psd(rho)
    R = _sqrt_psd(S @ sigma @ S)
    return float(np.real(np.trace(R))**2)


def S(rho):
    rho = (rho + rho.conj().T) / 2
    rho = rho / np.trace(rho).real
    w = np.linalg.eigvalsh(rho)
    w = w[w > 0]
    return float(-np.sum(w * np.log2(w)))




def mps_projector_to_mpo(mps):
    arrays = []
    for i in range(mps.L):
        Ti = mps[i]
        print(Ti.shape)

        if i == 0:
            Dr, d = Ti.shape
        elif i == mps.L-1:
            Dl, d = Ti.shape
        else:
            Dl, Dr, d = Ti.shape

        A = Ti.data                  # (Dl, Dr, d)
        B = A.conj()                 # (Dl, Dr, d)

        if i == 0:
            W = np.einsum('rd,RD->rRdD', A, B).reshape(Dr*Dr, d, d)
        elif i == mps.L-1:
            W = np.einsum('ld,LD->lLdD', A, B).reshape(Dl*Dl, d, d)
        else:
            W = np.einsum('lrd,LRD->lLrRdD', A, B).reshape(Dl*Dl, Dr*Dr, d, d)

        arrays.append(W)

    return qtn.MatrixProductOperator(
        arrays,
        shape='lrud',
        upper_ind_id='k{}',
        lower_ind_id='b{}',
        site_tag_id=mps.site_tag_id,
    )
"""
def purity_norm(rho, sigma):
    Z1 = np.real((rho&rho.H)^all)
    Z2 = np.real((sigma&sigma.H)^all)

    return np.real(((rho&sigma.H)^all)/np.sqrt(Z1*Z2))
"""
def purity_norm(rho, sigma,purity=True):
    Z1 = np.real((rho&rho.H)^all)
    Z2 = np.real((sigma&sigma.H)^all)
    if purity == False:

        return np.real(((rho&sigma.H)^all))
    else:
        return np.real(((rho&sigma.H)^all))/np.sqrt(Z1*Z2)


import numpy as np
from numpy.polynomial.legendre import leggauss

np.set_printoptions(precision=5, suppress=True)

# --------- helpers ---------
def _herm(X): return (X + X.conj().T)/2

def _mp_psd_real(H, a, eps=1e-12):
    H = _herm(H); vals, vecs = np.linalg.eigh(H)
    vals = np.clip(vals.real, eps, None)
    return vecs @ np.diag(vals**a) @ vecs.conj().T

def _mp_psd_complex(H, a, eps=1e-12):
    H = _herm(H); vals, vecs = np.linalg.eigh(H)
    vals = np.clip(vals.real, eps, None)
    return vecs @ np.diag(np.exp(a*np.log(vals))) @ vecs.conj().T
def partial_trace(rho, dims, keep=('A','B','C')):
    dA,dB,dC = dims
    R = rho.reshape(dA,dB,dC,dA,dB,dC)  # (A,B,C,A',B',C')
    sys = ['A','B','C']
    keep_set = set(keep)
    axes_map = {'A':0,'B':1,'C':2}
    idx = ['a','b','c','A','B','C']
    for s in sys:
        if s not in keep_set:
            i = axes_map[s]
            idx[i] = idx[i+3]  # same label -> trace
    ein_in  = ''.join(idx)
    ein_out = ''.join([idx[i] for i in range(3) if sys[i] in keep_set] +
                      [idx[i] for i in range(3,6) if sys[i-3] in keep_set])
    out = np.einsum(f"{ein_in}->{ein_out}", R)
    kept_dims = [dims[i] for i,s in enumerate(sys) if s in keep_set]
    D = int(np.prod(kept_dims)) if kept_dims else 1
    return out.reshape(D, D)

def random_density_matrix(dim, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(dim,dim)) + 1j*rng.normal(size=(dim,dim))
    rho = X @ X.conj().T
    rho /= np.trace(rho).real
    return rho

def beta_weight(t): return (np.pi/2.0) / (np.cosh(np.pi*t) + 1.0)

# --------- universal petz: state-form ---------
def universal_petz_state_AB_to_ABC(rho_AB, rho_BC, rho_B, dims, n_points=41, t_max=6.0):
    dA,dB,dC = dims
    I_A = np.eye(dA, dtype=complex)
    I_C = np.eye(dC, dtype=complex)
    xs, ws = leggauss(n_points)
    ts = t_max * xs; wts = t_max * ws
    rho_AB_up = np.kron(rho_AB, I_C)
    acc = np.zeros((dA*dB*dC, dA*dB*dC), dtype=complex)
    for t, wt in zip(ts, wts):
        a  = (1.0 + 1j*t)/2.0
        ah = (1.0 - 1j*t)/2.0
        sigBC_a   = _mp_psd_complex(rho_BC, a)
        sigBC_ah  = _mp_psd_complex(rho_BC, ah)
        sigB_mina = _mp_psd_complex(rho_B, -a)
        sigB_minh = _mp_psd_complex(rho_B, -ah)
        L  = np.kron(I_A, sigBC_a)
        Rm = np.kron(I_A, sigBC_ah)
        ML = np.kron(I_A, np.kron(sigB_mina, I_C))
        MR = np.kron(I_A, np.kron(sigB_minh, I_C))
        acc += wt * beta_weight(t) * (L @ (ML @ rho_AB_up @ MR) @ Rm)
    rho_rec = _herm(acc)
    tr = rho_rec.trace().real
    if abs(tr) > 1e-14: rho_rec /= tr
    return rho_rec

# --------- universal petz: channel (Choi) ---------
def rotated_petz_apply_on_B_noH(rho_BC, rho_B, Y_B, dB, dC, n_points=41, t_max=6.0):
    I_C = np.eye(dC, dtype=complex)
    xs, ws = leggauss(n_points)
    ts = t_max * xs; wts = t_max * ws
    acc = np.zeros((dB*dC, dB*dC), dtype=complex)
    for t, wt in zip(ts, wts):
        a  = (1.0 + 1j*t)/2.0
        ah = (1.0 - 1j*t)/2.0
        sigBC_a   = _mp_psd_complex(rho_BC, a)
        sigBC_ah  = _mp_psd_complex(rho_BC, ah)
        sigB_mina = _mp_psd_complex(rho_B, -a)
        sigB_minh = _mp_psd_complex(rho_B, -ah)
        mid_B  = sigB_mina @ Y_B @ sigB_minh
        mid_BC = np.kron(mid_B, I_C)
        acc   += wt * beta_weight(t) * (sigBC_a @ mid_BC @ sigBC_ah)
    return acc

def choi_rotated_petz_from_rhos(rho_BC, rho_B, n_points=41, t_max=6.0):
    dB = rho_B.shape[0]
    dBC = rho_BC.shape[0]
    if dBC % dB != 0:
        raise ValueError("rho_BC dimension not divisible by rho_B dimension")
    dC = dBC // dB
    d_in  = dB
    d_out = dB*dC
    J = np.zeros((d_out*d_in, d_out*d_in), dtype=complex)
    for i in range(dB):
        for j in range(dB):
            E_ij = np.zeros((dB,dB), dtype=complex); E_ij[i,j]=1.0
            Y_ij = rotated_petz_apply_on_B_noH(rho_BC, rho_B, E_ij, dB, dC, n_points, t_max)
            J += np.kron(Y_ij, E_ij)
    return J  # shape: ((dB*dC)*dB, (dB*dC)*dB)

def apply_idA_tensor_channelB_to_AB_with_J(rho_AB, J_R, dims):
    dA,dB,dC = dims
    d_out = dB*dC
    Rb = rho_AB.reshape(dA, dB, dA, dB)     # (a,b,a',b')
    Jb = J_R.reshape(d_out, dB, d_out, dB)  # (bc,b,bc',b')
    rho_rec = np.zeros((dA*d_out, dA*d_out), dtype=complex)
    for i in range(dB):
        for j in range(dB):
            X_ij = Rb[:, i, :, j].reshape(dA, dA)
            Y_ij = Jb[:, i, :, j]
            rho_rec += np.kron(X_ij, Y_ij)
    rho_rec = _herm(rho_rec)
    tr = rho_rec.trace().real
    if abs(tr)>1e-14: rho_rec /= tr
    return rho_rec

# --------- verification runner ---------
def run_verification(dA,dB,dC, seed, n_points=41, t_max=6.0):
    dims=(dA,dB,dC)
    d=dA*dB*dC
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(d,d)) + 1j*rng.normal(size=(d,d))
    rho_ABC = X @ X.conj().T; rho_ABC /= np.trace(rho_ABC).real

    rho_AB = partial_trace(rho_ABC, dims, keep=('A','B'))
    rho_BC = partial_trace(rho_ABC, dims, keep=('B','C'))
    rho_B  = partial_trace(rho_ABC, dims, keep=('B',))

    rho_state = universal_petz_state_AB_to_ABC(rho_AB, rho_BC, rho_B, dims, n_points=n_points, t_max=t_max)
    J = choi_rotated_petz_from_rhos(rho_BC, rho_B, n_points=n_points, t_max=t_max)
    rho_chan = apply_idA_tensor_channelB_to_AB_with_J(rho_AB, J, dims)

    return {
        "dims": f"{dA}x{dB}x{dC}",
        "seed": seed,
        "J_shape": J.shape,
        "F(state,channel)": fidelity(rho_state, rho_chan),
        "max|Δ| (state vs channel)": float(np.max(np.abs(rho_state - rho_chan))),
        "F(true, recovered)": fidelity(rho_ABC, rho_chan)
    }
def trace_distance(X, Y):
    H = (X - Y + (X - Y).conj().T)/2
    w = np.linalg.eigvalsh(H)
    return 0.5 * np.sum(np.abs(w))
import quimb.tensor as qtn

def mpo_transpose_phys(mpo, up=None, lo=None, inplace=False):
    up = up or getattr(mpo, 'upper_ind_id', 'k{}')
    lo = lo or getattr(mpo, 'lower_ind_id', 'b{}')
    X = mpo if inplace else mpo.copy()
    X.reindex_({up.format(i): f'__tmp{i}__' for i in range(mpo.L)})
    X.reindex_({lo.format(i): up.format(i) for i in range(mpo.L)})
    X.reindex_({f'__tmp{i}__': lo.format(i) for i in range(mpo.L)})
    return X

def mpo_adjoint(mpo, up=None, lo=None, inplace=False):
    X = mpo.conj() if inplace else mpo.conj()
    return mpo_transpose_phys(X, up=up, lo=lo, inplace=True)
def mpo_phys_to_last(mpo, up=None, lo=None, inplace=False):
    X = mpo if inplace else mpo.copy()
    up = up or getattr(mpo, 'upper_ind_id', 'k{}')
    lo = lo or getattr(mpo, 'lower_ind_id', 'b{}')

    for i in range(X.L):
        T = X[i]
        ki = up.format(i)
        bi = lo.format(i)
        bonds = [ix for ix in T.inds if ix not in (ki, bi)]
        T.transpose_(*(bonds + [ki, bi]))
    return X
def is_hermitian(X):
    return np.allclose(X, X.conj().T)


def contract_rho_segment(rho_list, start, length):
    #print(f"####start is {start}, and length is {length}")
    contracted = rho_list[start]
    phys_out_inds = [rho_list[start].inds[-2]]
    phys_in_inds  = [rho_list[start].inds[-1]]
    
    #print(rho_list)
    for i in range(start + 1, start + length):
        #print(f"#################i is {i}")
        #print(f"contracted tensor_{i} is {contracted}, rho_list_{i} is {rho_list[i]}")
        contracted = contracted @ rho_list[i]
        
        phys_out_inds.append(rho_list[i].inds[-2])
        phys_in_inds.append(rho_list[i].inds[-1])
        #print(f"step {i} phys_out_inds:{phys_out_inds}, phys_in_inds:{phys_in_inds}, start is {start}, length is {length}")
    
    
    return contracted, phys_out_inds, phys_in_inds




def build_S_matrix(J_value, d_phys, b, c, phys_out_inds, phys_in_inds):
    #print(f"d_phys is {d_phys}, a:{a}, b:{b}, c:{c}")
    d_out = d_phys ** (b + c)
    db = d_phys ** b
    dc = d_phys ** c
    dims_b = [d_phys] * b

    shape_out = [db, dc] + dims_b + [db, dc] + dims_b

    S = np.asarray(J_value).reshape([d_out, db, d_out, db]).reshape(shape_out)

    inds = (['d_out', 'dc']
            + phys_out_inds
            + ['d_out_', 'dc_']
            + phys_in_inds)

    return qtn.Tensor(S, inds=inds)



def decompose_to_mpo(T_orig, b, c, max_bond=64, cutoff=None):
    m = b + c
    R = T_orig.copy()
    mpo_tensors_list = []
    current_left_bond = 'b_left'

    for i in range(m - 1):
        po_ind = f'p_out_{i}'
        pi_ind = f'p_in_{i}'
        left_inds = [current_left_bond, po_ind, pi_ind]

        missing = [x for x in left_inds if x not in R.inds]
        assert not missing, f"R is missing required indices: {missing}"

        W_i, R = R.split(
            left_inds=left_inds,
            max_bond=max_bond,
            cutoff=cutoff,
            absorb='both',
        )

        shared = [ind for ind in W_i.inds if ind in R.inds and ind not in left_inds]
        assert len(shared) == 1, f"expected exactly one shared bond, got {shared}"
        new_bond_ind = shared[0]

        next_left_bond = f'b_{i+1}'
        W_i.reindex_({new_bond_ind: next_left_bond})
        R.reindex_({new_bond_ind: next_left_bond})

        W_i.transpose_(current_left_bond, next_left_bond, po_ind, pi_ind)
        mpo_tensors_list.append(W_i)
        current_left_bond = next_left_bond

    last_po = f'p_out_{m-1}'
    last_pi = f'p_in_{m-1}'
    assert last_po in R.inds and last_pi in R.inds and current_left_bond in R.inds

    W_last = qtn.Tensor(
        data=R.data[:, None, :, :],
        inds=(current_left_bond, 'b_right', last_po, last_pi),
        tags=R.tags,
    )
    W_last.transpose_(current_left_bond, 'b_right', last_po, last_pi)
    W_last.squeeze_("b_right")
    mpo_tensors_list.append(W_last)
    
    return mpo_tensors_list


def build_T_orig(S_matrix, contracted_tensor, d_phys, b, c):
    m = b + c
    #print(f"S_matrix is shape:{S_matrix.shape}, contracted_tensors is {contracted_tensor.shape}")
    S_out = (S_matrix @ contracted_tensor)

    S_out = S_out.transpose(S_out.inds[-1], *S_out.inds[:-1])
    S_out = S_out.data
    shape_out = [S_out.shape[0]] + [d_phys] * m + [d_phys] * m
    assert np.prod(shape_out) == S_out.size, "element count changed by reshape"
    S_out = S_out.reshape(shape_out)

    phys_out_inds = tuple(f'p_out_{i}' for i in range(m))
    phys_in_inds  = tuple(f'p_in_{i}'  for i in range(m))
    T_inds = ('b_left',) + phys_out_inds + phys_in_inds

    return qtn.Tensor(S_out, inds=T_inds, tags='ORIGINAL')


def mpo_grow(step, rho_AB_mpo, J, a, b, c, d_phys=2):
    contracted, phys_out_inds, phys_in_inds = contract_rho_segment(
        rho_AB_mpo, start=step*c + a, length=b
    )
    
    # -----------------------------
    # -----------------------------
    if type(J) == np.ndarray:
        S_matrix = build_S_matrix(
        J_value=J, d_phys=d_phys, b=b, c=c,
        phys_out_inds=phys_out_inds, phys_in_inds=phys_in_inds)
    else:   
        S_matrix = build_S_matrix(
            J_value=J.value, d_phys=d_phys, b=b, c=c,
            phys_out_inds=phys_out_inds, phys_in_inds=phys_in_inds
        )
    
    # -----------------------------
    # -----------------------------
    T_orig = build_T_orig(
        S_matrix=S_matrix,
        contracted_tensor=contracted,
        d_phys=d_phys, b=b, c=c, 
    )
    
    # -----------------------------
    # -----------------------------
    mpo_tensors_list = decompose_to_mpo(
        T_orig, b=b, c=c,
        max_bond=64,
        cutoff=None
    )
    array = [rho_AB_mpo[:step*c+a][i].data for i in range(len(rho_AB_mpo[:step*c+a].tensors))]+[mpo_tensors_list[i].data for i in range(len(mpo_tensors_list))]
    right_env_bond  = rho_AB_mpo[step*c + a].inds[1]          # right bond
    left_env_bond = mpo_tensors_list[0].inds[0]
    mpo_tensors_list[0].reindex_({left_env_bond:right_env_bond})
    inds = [rho_AB_mpo[:step*c+a][i].inds for i in range(len(rho_AB_mpo[:step*c+a].tensors))]+[mpo_tensors_list[i].inds for i in range(len(mpo_tensors_list))]
    mpo = qtn.MatrixProductOperator(
        array,
        shape='lrud',
        upper_ind_id='k{}',
        lower_ind_id='b{}',
        site_tag_id='I{}',
    )
    mpo_norm = mpo.multiply_(1/mpo.trace())
    return mpo_norm
