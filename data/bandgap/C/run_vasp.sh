#!/bin/bash

timestamp=$(date +"%Y%m%d_%H%M%S")
logname="log_${timestamp}.txt"

mpirun -n 4 vasp_std >> "$logname" 2>&1
