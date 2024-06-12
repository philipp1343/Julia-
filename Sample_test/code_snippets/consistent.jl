module Tmp

using PartitionedArrays
using LinearAlgebra

function consistent2!(v::PVector)
    vector_partition = partition(v)
    # Get the pre-computed communication pattern
    # stored in v.cache
    cache_for_assemble = v.cache
    cache_for_consistent = map(reverse,cache_for_assemble)
    # Fill the snd buffers
    function setup_snd!(cache,values)
        local_indices_snd = cache.local_indices_snd
        for (p,lid) in enumerate(local_indices_snd.data)
            cache.buffer_snd.data[p] = values[lid]
        end
    end
    map(setup_snd!,cache_for_consistent,vector_partition)
    # Get the exchange data
    buffer_snd = map(cache->cache.buffer_snd,cache_for_consistent)
    buffer_rcv = map(cache->cache.buffer_rcv,cache_for_consistent)
    neighbors_snd = map(cache->cache.neighbors_snd,cache_for_consistent)
    neighbors_rcv = map(cache->cache.neighbors_rcv,cache_for_consistent)
    graph = ExchangeGraph(neighbors_snd,neighbors_rcv)
    # Start the exchange
    t = exchange!(buffer_rcv,buffer_snd,graph)
    # read the rcv buffer asynchronously
    @async begin
        wait(t)
        # Copy values from the recive buffer to the appropriate location
        # in the vector
        function setup_rcv!(cache,values)
            local_indices_rcv = cache.local_indices_rcv
            for (p,lid) in enumerate(local_indices_rcv.data)
                values[lid] = cache.buffer_rcv.data[p]
            end
        end
        map(setup_rcv!,cache_for_consistent,vector_partition)
        nothing
    end
end

# Remarks:
# When moving to the GPU the "values" will be a CuVector
# You will need to create new buffers of the same structure
# as buffer_snd and buffer_rcv, but using CuVectors.
# You will need to create kernels to perform the data copy
# in functions setup_rcv! and setup_rcv! on the GPU.
# You can then copy the buffers on the GPU to the CPU and call exchange! on the CPU (as done now),
# or call exchange! directly (this will require CUDA-aware MPI and some further extension of the
# exchange! function)

# Implementation of spmv using the custom consistent2!
muladd!(y,A,x) = mul!(y,A,x,1,1)
function spmv!(b,A,x)
    t = consistent2!(x)
    map(mul!,own_values(b),own_own_values(A),own_values(x))
    wait(t)
    map(muladd!,own_values(b),own_ghost_values(A),ghost_values(x))
    b
end

# test

np = 3
ranks = DebugArray(LinearIndices((np,)))
IJV = map(ranks) do rank
    if rank == 1
        V = [2,1,6,4,5,3,1]
        I = [1,3,1,3,1,2,2]
        J = [1,2,3,3,5,5,7]
    elseif rank == 2
        V = [1,1,3,8,7,5]
        I = [5,4,5,4,5,6]
        J = [2,4,4,6,8,8]
    else
        V = [6,1,2,2,8,7]
        I = [7,9,8,7,9,8]
        J = [1,1,3,6,6,8]
    end
    I,J,V
end
AI,AJ,AV = tuple_of_arrays(IJV)
row_partition = uniform_partition(ranks,9)
A = psparse(AI,AJ,AV,row_partition,row_partition) |> fetch

x = pones(partition(axes(A,2)))
b1 = pzeros(partition(axes(A,1)))

spmv!(b1,A,x)

b2 = A*x

tol = 10.0e-13
@assert norm(b1-b2) / norm(b2) <= tol


end # module
