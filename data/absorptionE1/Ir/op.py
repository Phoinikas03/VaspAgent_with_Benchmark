import subprocess
import sys, os
env_copy = os.environ.copy()
env_copy.pop("LD_PRELOAD", None)

directory = sys.argv[1]
POTCAR = sys.argv[2]
POSCAR = sys.argv[3]
result = subprocess.run(f"mkdir {directory} && cp * {directory}", shell=True, capture_output=True, text=True, timeout=999999)
os.chdir(directory)
subprocess.run(f"cp {POTCAR} POTCAR && cp {POSCAR} POSCAR", shell=True, capture_output=True, text=True, timeout=999999)
#result = subprocess.run("mpirun -n 1 /home/bingxing2/ailab/majinzhe/task/vasp/vasp.6.3.2/bin/vasp_std", shell=True, capture_output=True, text=True, timeout=6000)
result = subprocess.run("bash ./run.sh", shell=True, capture_output=True, text=True, timeout=999999, env=env_copy)
os.chdir("..")
sys.stdout.write(result.stdout)
sys.stderr.write(result.stderr)
sys.exit(result.returncode)