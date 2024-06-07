#!/bin/bash

#SBATCH --time=00:05:00

#SBATCH -N 2

#SBATCH --ntasks-per-node=16

module load openmpi4/4.1.1

module load julia/1.7.3

pwd

mpiexec -np 32 julia --project=. hello_mpi.jl

