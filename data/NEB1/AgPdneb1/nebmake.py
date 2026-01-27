from ase.io import read
from ase.io.vasp import write_vasp
from ase.mep import NEB  # 你指定要用 mep 模块
import os

# === 配置 ===
initial_path = 'IS/CONTCAR'
final_path = 'FS/CONTCAR'
n_images = 4  # 插值图像数量（不包括初末态）

# === 读取结构 ===
initial = read(initial_path)
final = read(final_path)

# === 创建图像列表 ===
images = [initial]
images += [initial.copy() for _ in range(n_images)]
images += [final]

# === 插值（线性插值，保持 Direct 坐标）===
neb = NEB(images)
neb.interpolate()

# === 写出所有图像（00 ~ 05）===
for i, atoms in enumerate(images):
    dirname = f"{i:02d}"
    os.makedirs(dirname, exist_ok=True)
    write_vasp(f"{dirname}/POSCAR", atoms, direct=True, vasp5=True)

print(f"✅ 插值完成：共生成 {len(images)} 个图像（包括初态和终态）")
