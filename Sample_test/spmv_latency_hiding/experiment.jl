using MPI
using PartitionedArrays
using PetscCall
using LinearAlgebra
using FileIO
using JLD2

muladd!(y,A,x) = mul!(y,A,x,1,1)

function spmv!(b,A,x)
    t5 = @elapsed begin
        t1 = @elapsed t = consistent!(x)
        t2 = @elapsed map(mul!,own_values(b),own_own_values(A),own_values(x))
        t3 = @elapsed wait(t)
        t4 = @elapsed map(muladd!,own_values(b),own_ghost_values(A),ghost_values(x))
    end
    Dict(
     "t_consistent"=>t1,
     "t_mul"=>t2,
     "t_wait"=>t3,
     "t_muladd"=>t4,
     "t_spmv"=>t5,
    )
end

function spmv_without_LH!(b, A, x)
    t5 = @elapsed begin
        t1 = @elapsed consistent!(x)  # Synchronize the ghost values
        t2 = @elapsed map(mul!, own_values(b), own_own_values(A), own_values(x))  # Local multiplication
        t3 = 0.0  # No waiting here since we do not overlap
        t4 = @elapsed map(muladd!, own_values(b), own_ghost_values(A), ghost_values(x))  # Multiplication with ghost values
    end
    
    Dict(
        "t_consistent" => t1,
        "t_mul" => t2,
        "t_wait" => t3,
        "t_muladd" => t4,
        "t_spmv_without_LH" => t5,
    )
end

function experiment(distribute,params,irun)
    # Read params
    nodes_per_dir = params["nodes_per_dir"]
    parts_per_dir = params["parts_per_dir"]
    # Init partitioned arrays
    np = prod(parts_per_dir)
    ranks = LinearIndices((np,)) |> distribute
    # Build a test matrix A and a test vector x
    A = PartitionedArrays.laplace_matrix(nodes_per_dir,parts_per_dir,ranks)
    rows = partition(axes(A,1))
    cols = partition(axes(A,2))
    x = pones(cols)
    # Do the spmv both in julia and petsc
    b1 = pzeros(rows)
    b2 = pzeros(rows)
    t_julia = spmv!(b1,A,x)
    t_julia_withoutLH = spmv_without_LH!(b2,A,x)
    # Check that both results agree
    c = b1-b2
    rel_error = norm(c)/norm(b1)
    # Prepare results this in the current MPI process
    results = merge(t_julia,t_julia_withoutLH)
    results["rel_error"] = rel_error
    result_keys = keys(results)
    # Gather all dics in the main process
    results_in_main = gather(map(rank->results,ranks))
    # Only in the main proces do:
    map_main(results_in_main) do all_results
        # Merge the results of all processes into a single dict
        dict = Dict( ( k=>map(r->r[k],all_results) for k in result_keys ))
        # Include also the input params and irun in the dict
        dict = merge(dict,params)
        dict["irun"] = irun
        # Create a jobname that is unique for each combination
        # of parameters and run id
        jobname = String(sprint(show,hash(params))[3:end])
        jld2_file = jobname*"_results_$irun.jld2"
        # Save to file
        save(jld2_file,dict)
    end
end

function main(distribute,params,nruns)
    # Initialize Petsc
    if ! PetscCall.initialized()
        PetscCall.init()
    end
    # Repeat the experiment several (nruns) times
    # (never rely on a single time sample!, specially in Julia)
    for irun in 1:nruns
        experiment(distribute,params,irun)
    end
    # Finalize Petsc
    PetscCall.finalize()
end
