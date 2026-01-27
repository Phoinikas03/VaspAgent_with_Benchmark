#!/bin/bash
module purge
module load nvhpc/21.11_cuda10.2-11.0-11.5/nvhpc/21.11 compilers/gcc/9.3.0
export PATH=/home/bingxing2/ailab/majinzhe/task/vasp/vasp.6.3.2/bin:$PATH
mpirun -n 1 vasp_std
