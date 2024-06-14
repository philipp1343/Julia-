module load openmpi/gcc/64/4.0.2
module load intel/mkl/64/11.2/2015.5.223
module load julia/1.10.3
module load petsc/3.16.1

set -e
rm -rf Manifest.toml
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using MPIPreferences;use_jll_binary(;force=true)'
julia --project=. -e 'using PetscCall;PetscCall.use_system_petsc()'
julia --project=. -e 'using MPIPreferences;use_system_binary(;force=true)'
julia --project=. -e 'using MPI; using PetscCall'
julia --project=. -O3 --check-bounds=no -e 'using Pkg; Pkg.precompile()'
