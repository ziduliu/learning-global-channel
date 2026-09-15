import os
import numpy as np
from copy import deepcopy
from oqupy.config import NpDtype
from oqupy.base_api import BaseAPIClass
import oqupy
import numpy as np
import numpy as np
import oqupy
from oqupy.operators import sigma
import sys
from noise_shadow import *
from noise_gauss import add_gauss_pauli_noise
sx = sigma("x")
sy = sigma("y")
sz = sigma("z")
id2 = np.eye(2, dtype=complex)

sx_sys = np.kron(sx, id2)
sy_sys = np.kron(sy, id2)
sz_sys = np.kron(sz, id2)

def bell_dm_phi_plus():
    v00 = np.array([1, 0, 0, 0], dtype=complex)
    v11 = np.array([0, 0, 0, 1], dtype=complex)
    bell = (v00 + v11) / np.sqrt(2.0)
    rho = np.outer(bell, bell.conj())
    return rho
def build_bell_pair_chain(
    N_pairs: int,
    epsilon: float = 0.0,
    Jx: float = 0.5,
    Jy: float = 0.5,
    Jz: float = 0.5,
    gamma_x: float = 0.5,
    gamma_y: float = 0.5,
    gamma_z: float = 1,
):
    hs_dims = [4] * N_pairs
    system_chain = oqupy.SystemChain(hilbert_space_dimensions=hs_dims)

    # on-site Hamiltonian: epsilon * s^z_sys
    for n in range(N_pairs):
        H_site = epsilon * 0.5 * sz_sys
        system_chain.add_site_hamiltonian(
            site=n,
            hamiltonian=H_site
        )

    #   (σ^γ_sys ⊗ I_ref)_n  (σ^γ_sys ⊗ I_ref)_{n+1}
    for n in range(N_pairs - 1):
        if Jx:
            system_chain.add_nn_hamiltonian(
                site=n,
                hamiltonian_l=Jx * sx_sys,
                hamiltonian_r=      sx_sys)
        if Jy:
            system_chain.add_nn_hamiltonian(
                site=n,
                hamiltonian_l=Jy * sy_sys,
                hamiltonian_r=      sy_sys)
        if Jz:
            system_chain.add_nn_hamiltonian(
                site=n,
                hamiltonian_l=Jz * sz_sys,
                hamiltonian_r=      sz_sys)

    # DEPOLARIZATION: single-site X, Y, Z jumps (gamma_x=gamma_y=0.5, gamma_z=1)
    for n in range(N_pairs):
        if gamma_x:
            system_chain.add_site_dissipation(site=n, lindblad_operator=sx_sys, gamma=gamma_x)
        if gamma_y:
            system_chain.add_site_dissipation(site=n, lindblad_operator=sy_sys, gamma=gamma_y)
        if gamma_z:
            system_chain.add_site_dissipation(site=n, lindblad_operator=sz_sys, gamma=gamma_z)

    return system_chain
def make_initial_augmented_mps_bell(N_pairs: int):
    bell_rho = bell_dm_phi_plus()   # 4x4
    gammas = [bell_rho for _ in range(N_pairs)]
    initial_augmented_mps = oqupy.AugmentedMPS(gammas)
    return initial_augmented_mps


import numpy as np
from copy import deepcopy
from oqupy.config import NpDtype

def get_augmented_mps_manual(pt_tebd):
    t_mps = pt_tebd._t_mps

    gammas = []
    for i in range(t_mps.n):
        g = np.array(t_mps.get_gamma(i), dtype=NpDtype)
        shape = g.shape
        rank = len(shape)

        if rank == 4:
            tmp_gamma = g
        elif rank == 3:
            # (L, P, R) → (L, P, A=1, R)
            tmp_gamma = g.reshape(shape[0], shape[1], 1, shape[2])
        elif rank == 2:
            tmp_gamma = g.reshape(1, shape[0]*shape[1], 1, 1)
        elif rank == 1:
            tmp_gamma = g.reshape(1, shape[0], 1, 1)
        else:
            raise ValueError(f"Unexpected gamma rank {rank}")
        gammas.append(tmp_gamma)

    bond_dims = []
    for g1, g2 in zip(gammas[:-1], gammas[1:]):
        assert g1.shape[3] == g2.shape[0], (g1.shape, g2.shape)
        bond_dims.append(g1.shape[3])

    lambdas = []
    for i, chi in enumerate(bond_dims):
        lam = t_mps.get_lambda(i)
        if lam is None:
            lam_vec = np.ones(chi, dtype=NpDtype)
        else:
            lam = np.array(lam, dtype=NpDtype)
            if lam.ndim == 2:
                lam_vec = np.diag(lam)
            elif lam.ndim == 1:
                lam_vec = lam
            else:
                raise ValueError(f"Unexpected lambda rank {lam.ndim}")
        lambdas.append(lam_vec)

    class SimpleAugMPS:
        def __init__(self, gammas, lambdas):
            self.gammas = gammas
            self.lambdas = lambdas
            self.n = len(gammas)

    return SimpleAugMPS(gammas, lambdas)



N_pairs = 3
system_chain = build_bell_pair_chain(N_pairs)
initial_aug_mps = make_initial_augmented_mps_bell(N_pairs)

pt_tebd_params = oqupy.PtTebdParameters(
    dt=10**-3,
    order=1,
    epsrel=1.0e-6,
)



pt_tebd = oqupy.PtTebd(
    initial_augmented_mps=initial_aug_mps,
    system_chain=system_chain,
    process_tensors=[None]*N_pairs,    
    parameters=pt_tebd_params,
    dynamics_sites=list(range(N_pairs)),  
)

class SimpleAugMPS:
    def __init__(self, gammas, lambdas):
        self.gammas = gammas
        self.lambdas = lambdas
        self.n = len(gammas)

def get_augmented_mps_manual(pt_tebd):
    t_mps = pt_tebd._t_mps   # PtTebdBackend

    gammas = []
    for i in range(t_mps.n):
        g = np.array(t_mps.get_gamma(i), dtype=NpDtype)
        shape = g.shape
        rank = len(shape)

        if rank == 4:
            tmp_gamma = g
        elif rank == 3:
            # (L, P, R) → (L, P, A=1, R)
            tmp_gamma = g.reshape(shape[0], shape[1], 1, shape[2])
        elif rank == 2:
            tmp_gamma = g.reshape(1, shape[0]*shape[1], 1, 1)
        elif rank == 1:
            tmp_gamma = g.reshape(1, shape[0], 1, 1)
        else:
            raise ValueError(f"Unexpected gamma rank {rank}")
        gammas.append(tmp_gamma)

    bond_dims = []
    for g1, g2 in zip(gammas[:-1], gammas[1:]):
        assert g1.shape[3] == g2.shape[0], (g1.shape, g2.shape)
        bond_dims.append(g1.shape[3])

    lambdas = []
    for chi in bond_dims:
        lam = t_mps.get_lambda(len(lambdas))
        if lam is None:
            lam_vec = np.ones(chi, dtype=NpDtype)
        else:
            lam = np.array(lam, dtype=NpDtype)
            if lam.ndim == 2:
                lam_vec = np.diag(lam)
            elif lam.ndim == 1:
                lam_vec = lam
            else:
                raise ValueError(f"Unexpected lambda rank {lam.ndim}")
            assert lam_vec.shape[0] == chi
        lambdas.append(lam_vec)

    return SimpleAugMPS(gammas=gammas, lambdas=lambdas)

def augmented_mps_to_mpo_tensors(aug, vec_order="C", swap_ud=False):
    N = aug.n
    gammas_hat = []

    for n in range(N):
        G = np.array(aug.gammas[n], copy=True)
        if G.ndim != 4:
            raise ValueError(f"gamma[{n}] rank != 4: shape={G.shape}")
        L, P, A, R = G.shape
        if A != 1:
            raise NotImplementedError("only the case A=1 (process_tensors=None) is implemented")

        if n > 0:
            lam = np.array(aug.lambdas[n-1])
            if lam.ndim == 2:
                lam = np.diag(lam)
            assert lam.shape[0] == L, (lam.shape, L)
            G = lam[:, None, None, None] * G

        gammas_hat.append(G)

    mpo_tensors = []
    d = None
    for n, G in enumerate(gammas_hat):
        L, P, A, R = G.shape
        assert A == 1
        if d is None:
            d_guess = int(round(P ** 0.5))
            if d_guess * d_guess != P:
                raise ValueError(f"P={P} is not a perfect square and cannot be split into d x d")
            d = d_guess
        else:
            assert P == d * d

        G2 = G[:, :, 0, :]        # (L, P, R)

        if vec_order == "C":
            T = np.reshape(G2, (L, d, d, R), order="C")
        elif vec_order == "F":
            T = np.reshape(G2, (L, d, d, R), order="F")
        else:
            raise ValueError("vec_order must be 'C' or 'F'")

        if swap_ud:
            T = np.transpose(T, (0, 2, 1, 3))

        mpo_tensors.append(T)

    return mpo_tensors
import quimb.tensor as qtn

def build_mpo_matching_rho(pt_tebd):
    aug = get_augmented_mps_manual(pt_tebd)
    N = aug.n
    sites = list(range(N))

    rho_ABC = pt_tebd.get_current_density_matrix(sites)

    candidates = []
    for vec_order in ["C", "F"]:
        for swap_ud in [False, True]:
            mpo_tensors = augmented_mps_to_mpo_tensors(
                aug,
                vec_order=vec_order,
                swap_ud=swap_ud,
            )
            mpo = qtn.MatrixProductOperator(mpo_tensors, shape="ludr")
            rho_mpo = mpo.to_dense()
            err = np.linalg.norm(rho_mpo - rho_ABC)
            candidates.append(((vec_order, swap_ud), err, mpo_tensors, rho_mpo))

    (best_vec_order, best_swap_ud), best_err, best_tensors, best_rho = min(
        candidates,
        key=lambda x: x[1],
    )

    print(
        f"[build_mpo_matching_rho] best mapping: "
        f"vec_order={best_vec_order}, swap_ud={best_swap_ud}, "
        f"‖rho_mpo - rho_ABC‖_F = {best_err:.3e}"
    )

    mpo_best = qtn.MatrixProductOperator(best_tensors, shape="ludr")
    return mpo_best, rho_ABC, best_rho

def S(rho):
    rho = (rho + rho.conj().T) / 2
    rho = rho / np.trace(rho).real
    w = np.linalg.eigvalsh(rho)
    w = w[w > 0]
    return float(-np.sum(w * np.log2(w)))

def mpo_partial_trace(mpo, keep):
    tn = mpo.copy()
    for i in range(tn.L):
        #print(i)
        if i not in keep:
            bra, ket = tn[i].inds[-2:]
            tn[i].reindex_({bra:ket})
    mpo_data  = (tn^all).data
    phys_d = mpo_data.shape[0]
    sites = len(mpo_data.shape)//2

    x1 = np.arange(0,2*sites,2)
    x2 = np.arange(1,2*sites,2)
    mpo_data = mpo_data.transpose(np.append(x1,x2))
    
    return mpo_data.reshape([phys_d**sites, phys_d**sites])
N_pairs = 50
system_chain = build_bell_pair_chain(N_pairs)
initial_aug_mps = make_initial_augmented_mps_bell(N_pairs)

pt_tebd_params = oqupy.PtTebdParameters(
    dt=10**-3,
    order=1,
    epsrel=1.0e-7,
)
pt_tebd = oqupy.PtTebd(
    initial_augmented_mps=initial_aug_mps,
    system_chain=system_chain,
    process_tensors=[None]*N_pairs,    
    parameters=pt_tebd_params,
    dynamics_sites=list(range(N_pairs)),  
)

mpo_list=[]
for num_steps in [100]:



    results = pt_tebd.compute(num_steps, progress_type="bar")
    aug = get_augmented_mps_manual(pt_tebd)
    mpo_tensors = augmented_mps_to_mpo_tensors(aug, vec_order="C", swap_ud=False)
    tensor_list = []
    for i,tensor in enumerate(mpo_tensors):
        if i == 0:
            l,u,d,r = tensor.shape
            tensor = tensor.reshape(u,d,r)
        if i == len(mpo_tensors)-1:
            l,u,d,r = tensor.shape
            tensor = tensor.reshape(l,u,d)
        tensor_list.append(tensor)
    mpo=qtn.MatrixProductOperator(tensor_list, shape="ludr")
    mpo_list.append(mpo)


from _functions_ import *
from dynamics import *
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

def mpo_partial_trace(mpo, keep):
    tn = mpo.copy()
    for i in range(tn.L):
        #print(i)
        if i not in keep:
            bra, ket = tn[i].inds[-2:]
            tn[i].reindex_({bra:ket})
    mpo_data  = (tn^all).data
    phys_d = mpo_data.shape[0]
    sites = len(mpo_data.shape)//2

    x1 = np.arange(0,2*sites,2)
    x2 = np.arange(1,2*sites,2)
    mpo_data = mpo_data.transpose(np.append(x1,x2))
    
    return mpo_data.reshape([phys_d**sites, phys_d**sites])
def make_ds(rho_test):
    rho_test = 0.5* (rho_test+rho_test.conj().T)
    evals, evecs = np.linalg.eigh(rho_test)
    tol = 1e-12
    evals_clipped = np.maximum(evals, 0.0)
    rho_fixed = (evecs @ np.diag(evals_clipped) @ evecs.conj().T)
    tr = np.trace(rho_fixed).real
    if tr > 0:
        rho_fixed = rho_fixed / tr
    else:
        raise ValueError("all eigenvalues are 0")
    return rho_fixed
import scipy as sp
#beta_list = np.linspace(1,3,10) #Inverse temperature
phys_dim = 4
petz_dist_list = []
opt_dist_list = []
petz_dist_out_list = []
opt_dist_out_list = []
n_points = 21
t_max = 9
mpo_out_list = []
sigma_ABC_list=  []
J_list= []

num_mpo = int(sys.argv[1])
eta_list = list(np.logspace(-5, -2, 9))   # sigma of Gaussian coefficient noise
eta_i_record = []                          # per-window trace-distance errors
pauli_ops = generate_pauli_basis_choi_3qubit()

for num_list, eta in enumerate([eta_list[num_mpo]]):
    rho_mpo = mpo_list[-1]
    a = 1
    b = 1
    c = 1
    L = len(rho_mpo.tensors)
    dims = (phys_dim**a, phys_dim**b,phys_dim**c)
    dA, dB, dBout, dC, dOut = dims_from_abc(a, b, c, dloc=4) 
    for step in range(0,(L-a-b-c)//c+1,1):
    
        if step == 0:
        
            keep_ABC = np.arange(step*c, step*c + a + b + c)
            keep_AB  = np.arange(step*c, step*c + a + b)
            keep_B = np.arange(step*c+a, step*c+a+b)
            keep_BC = np.arange(step*c+a, step*c+ a+b +c)
            #print(f"keep_abc:{keep_ABC}, keep_AB:{keep_AB}, keep_B:{keep_B}, keep_BC:{keep_BC}")
            rho_ABC = mpo_partial_trace(rho_mpo, keep_ABC)#+np.random.random(size =(4**(a+b+c),4**(a+b+c)))*eta
            rho_ABC, eta_i = add_gauss_pauli_noise(rho_ABC, sigma=eta,
                                     pauli_ops=pauli_ops)
            eta_i_record.append(eta_i)
            rho_BC = qu.partial_trace(rho_ABC, keep=[1,2], dims=[dA,dB,dC])
            rho_B = qu.partial_trace(rho_ABC, keep=[1], dims=[dA,dB,dC])
            rho_AB = qu.partial_trace(rho_ABC, keep=[0,1], dims=[dA,dB,dC])        
            mpo_out = mpo_phys_to_last(qtn.MatrixProductOperator.from_dense(rho_AB,dims=phys_dim))
            mpo_petz_in = mpo_out

        else:
            keep_ABC = np.arange(step*c, step*c + a + b + c)
            keep_AB  = np.arange(step*c, step*c + a + b)
            keep_B = np.arange(step*c+a, step*c+a+b)
            keep_BC = np.arange(step*c+a, step*c+ a+b +c)

            rho_ABC = mpo_partial_trace(rho_mpo, keep_ABC)#+np.random.random(size =(4**(a+b+c),4**(a+b+c)))*eta
            rho_ABC, eta_i = add_gauss_pauli_noise(rho_ABC, sigma=eta,
                                     pauli_ops=pauli_ops)
            eta_i_record.append(eta_i)
            rho_AB = qu.partial_trace(rho_ABC, keep=[0,1], dims=[dA,dB,dC])
            rho_BC = qu.partial_trace(rho_ABC, keep=[1,2], dims=[dA,dB,dC])
            rho_B = qu.partial_trace(rho_ABC, keep=[1], dims=[dA,dB,dC])


        dimJ = dOut * dB
        rhoAB_blocks = blockify_rho_AB(rho_AB, dA, dB)
        J = cp.Variable((dimJ, dimJ), complex=True, hermitian=True)
        J_petz = choi_rotated_petz_from_rhos(rho_BC, rho_B, n_points=n_points, t_max=t_max)
        J.value = J_petz
        constraints = []
        constraints.append(J >> 0)  # Complete positivity
        tp_mat = ptr_out_trace_preserving_constraint(J, d_out=dOut, d_in=dB)
        constraints.append(tp_mat == np.eye(dB, dtype=complex))
        sigma_ABC = 0
        for i in range(dB):
            for ip in range(dB):
                Jii = J_block(J, d_out=dOut, d_in=dB, i=i, ip=ip)     # (dOut, dOut)
                RAB = rhoAB_blocks[i][ip]     # (dA, dA)
                #RAB = rhoAB_blocks[ip][i]
                sigma_ABC = sigma_ABC + cp.kron(RAB, Jii)
        diff = rho_ABC - sigma_ABC
        diffH = 0.5 * (diff + diff.H)


        objective = cp.Minimize(cp.norm(diff, 'fro'))

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, eps=1e-9, max_iters=200000)
        J_opt = J.value

        #J_petz = choi_rotated_petz_from_rhos(rho_BC, rho_B, n_points=n_points, t_max=t_max)
        #J_opt = J_petz
        #####grow up the mpo####
        mpo_out = mpo_grow(step,mpo_out, J_opt, a, b, c, d_phys =phys_dim )
        print("at step step=",step)
    mpo1 = rho_mpo
    
    distance_opt =  purity_norm(mpo1, mpo_out,purity=True)
    print(f"distance_opt={1-distance_opt}")
    mpo_out_list.append(mpo_out)    
    rep=os.environ.get('REP','0')
    np.save(f'eta_{num_mpo}_r{rep}.npy', np.array([np.max(eta_i_record), np.mean(eta_i_record), float(eta)]))
    print(f'sigma={eta:.3e}  eta_max={np.max(eta_i_record):.4e}  eta_mean={np.mean(eta_i_record):.4e}', flush=True)
    np.save(f'mpo_ref_{num_mpo}_r{rep}_a_{a}_b_{b}_c_{c}.npy', np.array(mpo1, dtype=object))
    np.save(f'mpo_out_{num_mpo}_r{rep}_a_{a}_b_{b}_c_{c}.npy', np.array(mpo_out, dtype=object))



