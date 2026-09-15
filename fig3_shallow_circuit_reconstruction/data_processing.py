import numpy as np
import quimb as qu
import quimb.tensor as qtn
import numpy as np
import quimb.tensor as qtn
import cvxpy as cp
def pauli_vec(letter, normalize=False, order="col", dtype=complex):
    p = letter.upper()
    p = letter.upper()
    if p == 'I':
        v = np.array([1, 0,   0,  1], dtype=dtype)
    elif p == 'X':
        v = np.array([0, 1,   1,  0], dtype=dtype)
    elif p == 'Y':
        # Y = [[0,-i],[i,0]]
        v = np.array([0, 1j, -1j, 0], dtype=dtype) if order == 'col' \
            else np.array([0, -1j, 1j, 0], dtype=dtype)
    elif p == 'Z':
        v = np.array([1, 0,   0, -1], dtype=dtype)
    else:
        raise ValueError(f"Unknown Pauli '{letter}', use one of 'I','X','Y','Z'.")
    if normalize:
        v = v / np.sqrt(2.0)
    return v

def pauli_product_mps(paulis, normalize=False, order='col',
                      site_tag_id='LI{}', tags=('pauli-liou', 'product'),inds="k{}"):
    pstr = paulis
    arrays = []
    for i, ch in enumerate(pstr):
        v = pauli_vec(ch, normalize=normalize, order=order)

        arrays.append(v.reshape(1,1,4))  # (left bond, phys=4, right bond)
    mps = qtn.MatrixProductState(
        arrays,
        site_ind_id=inds,
        site_tag_id=site_tag_id,
        shape='lrp',
        tags=tags
    )
    return mps


mpo_q_list = []
for i in range(10):
    mpo_out = np.load(f"mpo_out_{i}_a_1_b_1_c_1.npy",allow_pickle=True).tolist()
    mpo_q_list.append(mpo_out)

mpo_out_list = []
for i in range(10):
    mpo_out = np.load(f"mpo_ref_{i}_a_1_b_1_c_1.npy",allow_pickle=True).tolist()
    mpo_out_list.append(mpo_out)

from itertools import combinations, product
from math import comb

def pauli_strings_by_weight(n, w, non_id=('X', 'Y', 'Z'), identity='I'):
    if w < 0 or w > n:
        return []
    if w == 0:
        return [identity * n]

    out = []
    for idxs in combinations(range(n), w):
        for letters in product(non_id, repeat=w):
            arr = [identity] * n
            for pos, let in zip(idxs, letters):
                arr[pos] = let
            out.append(''.join(arr))
    return out

def count_pauli_strings_by_weight(n, w, k=3):
    if w < 0 or w > n: 
        return 0
    return comb(n, w) * (k ** w)

weight_list = [0,1,2,3,4]
n = 20
out_list = []
for mpo_q in mpo_q_list:
    v_list=  []
    for weight in weight_list:
        print(f"weight is {weight}")
        str_list = pauli_strings_by_weight(n, weight)
        val_list = []
        for str in str_list:

            vecl= pauli_product_mps(str,inds="k{}",normalize=True)
            vecr= pauli_product_mps(str,inds="b{}",normalize=True)
            tn = vecl.H & mpo_q & vecr
            val = tn.contract(all, optimize='auto-hq')
            val_list.append(val)
        v_list.append(np.sum(np.abs(val_list)))
    out_list.append(v_list)

weight_list = [0,1,2,3,4]
n = 20
out_list_result = []
for mpo_q in mpo_out_list:
    v_list=  []
    for weight in weight_list:
        print(f"weight is {weight}")
        str_list = pauli_strings_by_weight(n, weight)
        val_list = []
        for str in str_list:

            vecl= pauli_product_mps(str,inds="k{}",normalize=True)
            vecr= pauli_product_mps(str,inds="b{}",normalize=True)
            tn = vecl.H & mpo_q & vecr
            val = tn.contract(all, optimize='auto-hq')
            val_list.append(val)
        v_list.append(np.sum(np.abs(val_list)))
    out_list_result.append(v_list)
print(f"out_list is {out_list}")
print(f"result is {out_list_result}")
np.save("out_list",np.array(out_list))
np.save("out_list_result",np.array(out_list_result))
