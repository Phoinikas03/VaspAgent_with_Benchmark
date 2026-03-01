#!/bin/bash
# VASP 运行脚本模板 - 请根据本集群环境修改
# 复制到各材料 run 目录后会被 data.json 中的 "command": "bash run_vasp.sh" 调用

# ========== 集群调度（按需取消注释并修改）==========
# SLURM 示例:
# #SBATCH -J vasp
# #SBATCH -N 1
# #SBATCH -n 8
# #SBATCH -p normal
# #SBATCH -t 24:00:00

# PBS 示例:
# #PBS -N vasp
# #PBS -l nodes=1:ppn=8
# #PBS -l walltime=24:00:00
# #PBS -q batch

# ========== 环境与执行 ==========
# 若需加载模块（按本集群修改）:
# module load intel/2020
# module load openmpi/4.0
# module load vasp/6.x

# 并行进程数，请按本机/队列资源修改
NP=${NP:-8}

mpirun -np $NP vasp_std
