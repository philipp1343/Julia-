#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH -e error.txt
#SBATCH -o output.txt
#SBATCH --ntasks-per-node=32
#test
module load julia/1.10.3
module load intel/mkl/64/11.2/2015.5.223
module load openmpi/gcc/64/4.0.2
module load petsc/3.16.1

# Disable multi-threading
export OMP_NUM_THREADS=1
export NUM_THREADS=1

# This file is to be launched with sbatch:
# $ sbatch step_02_launch_job.sh

# Use the switches above to select the number of nodes
# and the number of tasks (processes) per node

# Use different values of parts_per_dir, nodes_per_dir

# NB! the value passed to np needs to coincide with prod(parts_per_dir)
MPIFLAGS="--map-by node:span --rank-by core"
JULIAFLAGS="--project=. --check-bounds=no -O3"
mpiexec -np 8 $MPIFLAGS  julia $JULIAFLAGS -e '
    include("experiment.jl")
    with_mpi() do distribute
        params = Dict(
            "parts_per_dir"=>(2,2,2),
            "nodes_per_dir"=>(100,100,100),
           )
        nruns = 4
        main(distribute,params,nruns)
    end
'
