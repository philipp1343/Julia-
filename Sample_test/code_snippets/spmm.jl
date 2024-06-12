using PartitionedArrays
using SparseArrays
using LinearAlgebra

muladd!(C,A,B) = mul!(C,A,B,1,1)

function merge_blocks(own_own,own_ghost,rows,cols)
    n_ghost_rows = 0
    n_own_cols = size(own_own,2)
    n_ghost_cols = size(own_ghost,2)
    ghost_own = similar(own_own,n_ghost_rows,n_own_cols)
    ghost_ghost = similar(own_own,n_ghost_rows,n_ghost_cols)
    blocks = PartitionedArrays.split_matrix_blocks(
        own_own,own_ghost,ghost_own,ghost_ghost)
    row_perm = PartitionedArrays.local_permutation(rows)
    col_perm = PartitionedArrays.local_permutation(cols)
    A = PartitionedArrays.split_matrix(blocks,row_perm,col_perm)
    A
end

function spmm(A,B)
    rows_A = partition(axes(A,1))
    cols_A = partition(axes(A,2))
    t = consistent(B,cols_A;reuse=true)
    A_own_own = own_own_values(A)
    A_own_ghost = own_ghost_values(A)
    B_own_own = own_own_values(B)
    C_own_own = map(*,A_own_own,B_own_own)
    D,cacheD = fetch(t)
    cols_D = partition(axes(D,2))
    D_own_ghost = own_ghost_values(D)
    D_ghost_own = ghost_own_values(D)
    D_ghost_ghost = ghost_ghost_values(D)
    C_own_ghost = map(*,A_own_own,D_own_ghost)
    map(muladd!,C_own_own,A_own_ghost,D_ghost_own)
    map(muladd!,C_own_ghost,A_own_ghost,D_ghost_ghost)
    C_partition = map(merge_blocks,C_own_own,C_own_ghost,rows_A,cols_D)
    C = PSparseMatrix(C_partition,rows_A,cols_D,true)
    cacheC = (D,cacheD)
    C, cacheC
end

function spmm!(C,A,B,cacheD)
    D,cacheD = cacheC
    t = consistent!(D,B,cacheD)
    A_own_own = own_own_values(A)
    A_own_ghost = own_ghost_values(A)
    B_own_own = own_own_values(B)
    C_own_own = own_own_values(C)
    C_own_ghost = own_ghost_values(C)
    map(mul!,C_own_own,A_own_own,B_own_own)
    wait(t)
    D_own_ghost = own_ghost_values(D)
    D_ghost_own = ghost_own_values(D)
    D_ghost_ghost = ghost_ghost_values(D)
    map(mul!,C_own_ghost,A_own_own,D_own_ghost)
    map(muladd!,C_own_own,A_own_ghost,D_ghost_own)
    map(muladd!,C_own_ghost,A_own_ghost,D_ghost_ghost)
    C
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
A_seq = centralize(A)

C, cacheC = spmm(A,A)
@assert centralize(C) == A_seq*A_seq

map(partition(A)) do A
    nonzeros(A.blocks.own_own) .*= 4
    nonzeros(A.blocks.own_ghost) .*= 4
end
A_seq = centralize(A)

spmm!(C,A,A,cacheC)
@assert centralize(C) == A_seq*A_seq


