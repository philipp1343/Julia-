using MPI
using PartitionedArrays
using PetscCall
using LinearAlgebra
using FileIO
using JLD2
using Random

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

function spmv_petsc!(b,A,x)
    # Convert the input to petsc objects
    t1 = @elapsed begin
        mat = Ref{PetscCall.Mat}()
        vec_b = Ref{PetscCall.Vec}()
        vec_x = Ref{PetscCall.Vec}()
        parts = linear_indices(partition(x))
        petsc_comm = PetscCall.setup_petsc_comm(parts)
        args_A = PetscCall.MatCreateMPIAIJWithSplitArrays_args(A,petsc_comm)
        args_b = PetscCall.VecCreateMPIWithArray_args(copy(b),petsc_comm)
        args_x = PetscCall.VecCreateMPIWithArray_args(copy(x),petsc_comm)
        ownership = (args_A,args_b,args_x)
        PetscCall.@check_error_code PetscCall.MatCreateMPIAIJWithSplitArrays(args_A...,mat)
        PetscCall.@check_error_code PetscCall.MatAssemblyBegin(mat[],PetscCall.MAT_FINAL_ASSEMBLY)
        PetscCall.@check_error_code PetscCall.MatAssemblyEnd(mat[],PetscCall.MAT_FINAL_ASSEMBLY)
        PetscCall.@check_error_code PetscCall.VecCreateMPIWithArray(args_b...,vec_b)
        PetscCall.@check_error_code PetscCall.VecCreateMPIWithArray(args_x...,vec_x)
    end
    # This line does the actual product
    t2 = @elapsed PetscCall.@check_error_code PetscCall.MatMult(mat[],vec_x[],vec_b[])
    # Move the result back to julia
    t3 = @elapsed PetscCall.VecCreateMPIWithArray_args_reversed!(b,args_b)
    # Cleanup
    t4 = @elapsed begin
        GC.@preserve ownership PetscCall.@check_error_code PetscCall.MatDestroy(mat)
        GC.@preserve ownership PetscCall.@check_error_code PetscCall.VecDestroy(vec_b)
        GC.@preserve ownership PetscCall.@check_error_code PetscCall.VecDestroy(vec_x)
    end
    Dict(
     "t_julia_to_petsc"=>t1,
     "t_spmv_petsc"=>t2,
     "t_petsc_to_julia"=>t3,
     "t_cleanup"=>t4)
end


function create_random_ijv(rank, num_rows, num_cols, num_entries)
    I = rand(1:num_rows, num_entries)
    J = rand(1:num_cols, num_entries)
    V = rand(num_entries)
    return I, J, V
end

function experiment(distribute, params, irun)
    # Read params
    nodes_per_dir = params["nodes_per_dir"]
    parts_per_dir = params["parts_per_dir"]
    num_rows = params["num_rows"]
    num_cols = params["num_cols"]
    num_entries_per_rank = params["num_entries_per_rank"]
    
    # Init partitioned arrays
    np = prod(parts_per_dir)
    ranks = LinearIndices((np,)) |> distribute

    IJV = map(ranks) do rank
        create_random_ijv(rank, num_rows, num_cols, num_entries_per_rank)
    end

    I, J, V = tuple_of_arrays(IJV)
    row_partition = uniform_partition(ranks, num_rows)
    A = psparse(I, J, V, row_partition, row_partition) |> fetch

    rows = partition(axes(A, 1))
    cols = partition(axes(A, 2))
    x = pones(cols)
    
    # Do the spmv both in julia and petsc
    b1 = pzeros(rows)
    b2 = pzeros(rows)
    t_julia = spmv!(b1, A, x)
    t_petsc = spmv_petsc!(b2, A, x)
    
    # Check that both results agree
    c = b1 - b2
    rel_error = norm(c) / norm(b1)
    
    # Prepare results in the current MPI process
    results = merge(t_julia, t_petsc)
    results["rel_error"] = rel_error
    result_keys = keys(results)
    
    # Gather all dicts in the main process
    results_in_main = gather(map(rank -> results, ranks))
    
    # Only in the main process do:
    map_main(results_in_main) do all_results
        # Merge the results of all processes into a single dict
        dict = Dict((k => map(r -> r[k], all_results) for k in result_keys))
        # Include also the input params and irun in the dict
        dict = merge(dict, params)
        dict["irun"] = irun
        # Create a jobname that is unique for each combination
        # of parameters and run id
        jobname = String(sprint(show, hash(params))[3:end])
        jld2_file = jobname * "_results_$irun.jld2"
        # Save to file
        save(jld2_file, dict)
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
