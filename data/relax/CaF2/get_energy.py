from pymatgen.io.vasp import Vasprun

def get_energy(path='./vasprun.xml'):
    # 读取 vasprun.xml 文件
    vasprun = Vasprun(path)
    
    # 获取结构优化后的系统总能量
    total_energy = vasprun.final_energy
    
    print("Optimized total energy:", total_energy)
    return total_energy

# 调用函数
get_energy()
