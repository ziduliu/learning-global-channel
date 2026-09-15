from _functions_ import *
from dynamics import *
import numpy as np
from tenpy.networks.site import SpinHalfSite, GroupedSite
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel
import copy
import numpy as np
from scipy.linalg import expm
import quimb as qu
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from oqupy.mps_mpo import compute_tebd_propagator
import sys


def compose_from_layers(tebd_obj, full,max_bond=512,cutoff=10**-6,dagger=False):
    gate_layers =  tebd_obj.gate_layers
    if dagger == True:
        gate_layers.reverse()
    for layer in gate_layers:
        layer_mpo = layer_to_chain_mpo(layer, L, d2)
        full = compose_mpo(layer_mpo, full, max_bond=max_bond, cutoff=cutoff)
    return full

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
    #print(np.append(x1,x2))
    mpo_data = mpo_data.transpose(np.append(x1,x2))
    return mpo_data.reshape([phys_d**sites, phys_d**sites])
def full_tebd_step_mpo(tebd_list, L: int, dloc: int,
                       max_bond: int = 512,
                       cutoff: float = 1e-12,
                       split: str = None):

    d2 = dloc * dloc
    def compose_from_layers(tebd_obj, full,max_bond,cutoff):
        for layer in tebd_obj.gate_layers:
            layer_mpo = layer_to_chain_mpo(layer, L, d2)
            full = compose_mpo(layer_mpo, full, max_bond=max_bond, cutoff=cutoff)
        return full



    full = identity_mpo(L, d2)
    for tebd in tebd_list:
        full = compose_from_layers(tebd, full, max_bond=max_bond, cutoff=cutoff)
    return full
def build_chain(L=6, dloc=2, Jx=1.0, Jy=1.0, Jz=1.0, gamma=0.2,epsilon_x=1.0, epsilon_y=1.0, epsilon_z=1.0,random=False):
    chain = SystemChain([dloc]*L)
    for i in range(L-1):
        if (i%2) == 0:
            if random == False:
                Jx_here, Jy_here, Jz_here = Jx, Jy, Jz
            else:
                rx,ry,rz = np.random.random(3)
                Jx_here, Jy_here, Jz_here = Jx*rx, Jy*ry, Jz*rz
        else:
            if random == True:
                rx,ry,rz = np.random.random(3)
            else:
                rx,ry,rz = 1,1,1
            Jx_here, Jy_here, Jz_here = epsilon_x*rx, epsilon_y*ry, epsilon_z*rz

        if Jx_here: chain.add_nn_hamiltonian(i, Jx_here*sigma('x'), sigma('x'))
        if Jy_here: chain.add_nn_hamiltonian(i, Jy_here*sigma('y'), sigma('y'))
        if Jz_here: chain.add_nn_hamiltonian(i, Jz_here*sigma('z'), sigma('z'))
    for i in range(L):
        if gamma: chain.add_site_dissipation(i,sigma('z'),gamma=gamma)
    return chain
L, dloc =20, 2
cmi_list = []
mpo_q_list = []
num_mpo = int(sys.argv[1])
gamma_list = np.linspace(0,0.01,10)
#t_list = np.linspace(0,1,10)
dt = 1
eta_x, eta_y, eta_z = [1,1,1]
for gamma in [0, gamma_list[num_mpo]]:
     
    Jx, Jy, Jz = 1,1,0
    chain_U = build_chain(L=L, dloc=dloc, Jx=Jx, Jy=Jy, Jz=Jz, gamma=0, epsilon_x=0, epsilon_y=0, epsilon_z=0)
    chain_U_err = build_chain(L=L, dloc=dloc, Jx=eta_x*gamma, Jy=eta_y*gamma, Jz=eta_z*gamma, gamma=0, epsilon_x=eta_x*gamma, epsilon_y=eta_y*gamma, epsilon_z=eta_z*gamma,random=True)


    chain_diss = build_chain(L=L, dloc=dloc, Jx=0, Jy=0, Jz=0, gamma=gamma,epsilon_x=0, epsilon_y=0, epsilon_z=0)
    tebd_U = compute_tebd_propagator(system_chain=chain_U,
                                   time_step=np.pi/4,
                                   epsrel=1e-7,
                                   order=1)
    tebd_U_err = compute_tebd_propagator(system_chain=chain_U_err,
                                   time_step=1,
                                   epsrel=1e-7,
                                   order=1)
    tebd_diss = compute_tebd_propagator(system_chain=chain_diss,
                                   time_step=1,
                                   epsrel=1e-7,
                                   order=1)
    d2 = dloc**2


    full = identity_mpo(L, d2)
    full = compose_from_layers(tebd_U, full)
    full = compose_from_layers(tebd_U_err,full)
    full = compose_from_layers(tebd_diss, full)
    
    #full = compose_from_layers(tebd_U_dagger, full, dagger=True)
    E_dt = full

    mpo_choi = tenpy_super_to_choi_mpo(E_dt, d=2,phys_axes=[1,2])
    site_tag_id='I{}'
    left_phys_id='k{}'
    right_phys_id='b{}'
    bond_id='I{}'
    mpo_data = [mpo_choi[i].transpose(0,-1,1,2) for i in range(len(mpo_choi))]
    Ts = []
    for i in range(L):
        W_i = mpo_data[i]/2 #normalization
        Dl, Dr, din, dout = W_i.shape
        inds = (bond_id.format(i-1), bond_id.format(i), left_phys_id.format(i), right_phys_id.format(i))
        tags = [site_tag_id.format(i), 'MPO']
        T = qtn.Tensor(data=W_i, inds=inds, tags=tags)
        Ts.append(T)
    arrays = [T.data for T in Ts]
    if len(arrays)!= 1:
        _,D,d1,d2 = arrays[0].shape
        arrays[0] = arrays[0].reshape(D,d1,d2)
        D,_,d1,d2 = arrays[-1].shape
        arrays[-1] =  arrays[-1].reshape(D,d1,d2)
    mpo_q = qtn.MatrixProductOperator(arrays,
        shape='lrud',
        upper_ind_id='k{}',
        lower_ind_id='b{}')

    mpo_q_list.append(mpo_q)


import scipy as sp
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
for num_list, rho_mpo in enumerate([mpo_q_list[-1]]):
    
    a = 1
    b = 1
    c = 1
    L = len(rho_mpo.tensors)
    dims = (phys_dim**a, phys_dim**b,phys_dim**c)

    for step in range(0,(L-a-b-c)//c+1,1):

        if step == 0:
            keep_ABC = np.arange(step*c, step*c + a + b + c)
            keep_AB  = np.arange(step*c, step*c + a + b)
            keep_B = np.arange(step*c+a, step*c+a+b)
            keep_BC = np.arange(step*c+a, step*c+ a+b +c)
            #print(f"keep_abc:{keep_ABC}, keep_AB:{keep_AB}, keep_B:{keep_B}, keep_BC:{keep_BC}")
            rho_ABC = mpo_partial_trace(rho_mpo, keep_ABC)#+np.random.random(size =(2**(a+b+c),2**(a+b+c)))*eta
            rho_BC = mpo_partial_trace(rho_mpo, keep_BC)#+np.random.random(size =(2**(b+c),2**(b+c)))*eta
            rho_B  = mpo_partial_trace(rho_mpo, keep_B)#+np.random.random(size =(2**b,2**b))*eta
            rho_AB = mpo_partial_trace(rho_mpo, keep_AB)#+np.random.random(size =(2**(a+b),2**(a+b)))*eta
            mpo_out = mpo_phys_to_last(qtn.MatrixProductOperator.from_dense(rho_AB,dims=phys_dim))
            mpo_petz_in = mpo_out
        else:
            keep_ABC = np.arange(step*c, step*c + a + b + c)
            keep_AB  = np.arange(step*c, step*c + a + b)
            keep_B = np.arange(step*c+a, step*c+a+b)
            keep_BC = np.arange(step*c+a, step*c+ a+b +c)
            #print(f"keep_abc:{keep_ABC}, keep_AB:{keep_AB}, keep_B:{keep_B}, keep_BC:{keep_BC}")

            #print(f"keep_ABC is {keep_ABC}, keep_B is {keep_B}")
            #print(f"length of mpo out is {len(mpo_out.tensors)}")
            #print(f"Keep_AB is {keep_AB}, keep ABC is {keep_ABC}")
            rho_ABC = mpo_partial_trace(rho_mpo, keep_ABC)

            #print(f"shape of rho_ABC is {rho_ABC.shape}")
            #rho_AB = mpo_partial_trace(mpo_out, keep_AB)
            rho_AB = mpo_partial_trace(mpo_out, keep_AB)
            rho_BC = mpo_partial_trace(rho_mpo, keep_BC)
            rho_B  = mpo_partial_trace(rho_mpo, keep_B)

        dA, dB, dBout, dC, dOut = dims_from_abc(a, b, c, dloc=4)
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
        #objective = cp.Minimize(cp.normNuc(diffH))
        objective = cp.Minimize(cp.norm(diff, 'fro'))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)
        J_opt = J.value
        print("status:", prob.status)
        print("opt value:", prob.value)
        #J_petz = choi_rotated_petz_from_rhos(rho_BC, rho_B, n_points=n_points, t_max=t_max)
        #J_opt = J_petz
        #####grow up the mpo####
        mpo_out = mpo_grow(step,mpo_out, J_opt, a, b, c, d_phys =phys_dim )
        print("at step step=",step)
    mpo1 = rho_mpo
    mpo_ref = mpo_q_list[0]
    distance_opt =  purity_norm(mpo_ref, mpo_out,purity=False)
    distance_ref = purity_norm(mpo_ref, mpo1,purity=False)
    print(f"distance_opt={1-distance_opt}")
    print(f"distance_ref={1-distance_ref}")
    mpo_out_list.append(mpo_out)
    np.save(f'mpo_ref_{num_mpo}_a_{a}_b_{b}_c_{c}.npy', np.array(mpo1, dtype=object))
    np.save(f'mpo_out_{num_mpo}_a_{a}_b_{b}_c_{c}.npy', np.array(mpo_out, dtype=object))




