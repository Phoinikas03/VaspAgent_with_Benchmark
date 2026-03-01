from pymatgen.io.vasp import Vasprun

def gap_calc(path='./vasprun.xml'):
    vasprun = Vasprun(path, parse_projected_eigen=True)
    band_structure = vasprun.get_band_structure()

    band_gap = band_structure.get_band_gap()
    print("Band gap:", band_gap)
    return band_gap

gap_calc()