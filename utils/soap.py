# pip install ase dscribe numpy
from ase.io import read
import numpy as np
from dscribe.descriptors import SOAP

# === 1) 读入结构（用POSCAR/CONTCAR/CIF等；INCAR不包含几何，不能用） ===
atoms_A = read("/home/bingxing2/ailab/xiazeyu_p/Programs/vasp_benchmark/relax/Ti/relax/CONTCAR")  # 或 "CONTCAR_A" / "A.cif"
atoms_B = read("/home/bingxing2/ailab/xiazeyu_p/Programs/vasp_benchmark/material_evaluator/runs/2025_07_02/relax_deepseek-v3-12_59_04/Ti/CONTCAR")

# === 2) 定义SOAP描述符 ===
# 常用超参：rcut(截断半径), nmax, lmax, sigma(高斯宽度)
species = list(set(atoms_A.get_chemical_symbols() + atoms_B.get_chemical_symbols()))
soap = SOAP(
    species=species,
    r_cut=5.0,
    n_max=14,
    l_max=10,
    sigma=0.1,
    average='inner',
    sparse=False,
    periodic=True,
)



# === 3) 计算结构级SOAP向量 ===
desc_A = soap.create(atoms_A)  # shape: (1, d)
desc_B = soap.create(atoms_B)  # shape: (1, d)
vA, vB = desc_A[0], desc_B[0]

# === 4) 余弦相似度（也就是归一化点积） ===
def cosine_sim(x, y, eps=1e-12):
    nx = np.linalg.norm(x) + eps
    ny = np.linalg.norm(y) + eps
    return float(np.dot(x, y) / (nx * ny))

from scipy.optimize import linear_sum_assignment

def hungarian_similarity(A, B):
    # 归一化每一行
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)

    # 相似度矩阵 (cosine)
    sim_matrix = A_norm @ B_norm.T   # shape=(Na, Nb)

    # Hungarian 最大化 → 取 -sim 做最小化
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    return float(sim_matrix[row_ind, col_ind].mean())

# print(desc_A, desc_B)
sim = cosine_sim(desc_A, desc_B)
# sim = hungarian_similarity(desc_A, desc_B)
print(f"Average-SOAP similarity: {sim:.8f}")
