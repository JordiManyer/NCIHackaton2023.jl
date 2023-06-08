"""
  Tests the manual computation of flops against 
  the results provided by the GFlops.jl package
"""

using Test
using BenchmarkTools
using LinearAlgebra
using IterativeSolvers
using GFlops

using Gridap
using Gridap.TensorValues
using Gridap.Arrays

using NCIHackaton2023

# SQ  # Number of quadrature points per direction
# SB  # Number of degrees of freedom per direction
function matrix_free_vector_product_flops(SQ,SB,U,A_lazy,x; 
                                          structured_mesh_opt=true)
    # A) Scatter the input vector into blocks for each cell
    #
    
    # Cell DoF IDs: For each cell, contains the IDs of it's DoFs. 
    #      Free DoFs      -> Positive IDs, values contained in x
    #      Dirichlet DoFs -> Negative IDs, values set to 0.0
    cell_dof_ids = get_cell_dof_ids(U)
    cell_values = map(cell_dof_ids) do ids 
        xi = ones(length(ids))
        for (i,id) in enumerate(ids)
            (id > 0) ? (xi[i] = x[id]) : (xi[i] = 0.0)
        end
        return xi
    end

    # B) Local integration within each cell
    cell_wq, cell_jq, cell_djq = A_lazy.cell_quantities
    cell_integrals = map(1:num_cells(model)) do cell
        # Local input vector
        xi = cell_values[cell]

        # Local caches for each tensor index contraction
        Zk = [zeros(D,SQ[1:d-1]...,SB[d:D]...) for d in 1:D+1]

        # B.1) Vector -> Tensor 
        dof_map = m.dof_map
        @inbounds for r in 1:D, (i,I) in enumerate(dof_map)
            Zk[1][r,I] = xi[i]
        end

        # B.2) First Tensor contractions: Evaluation of ∇(u) on the quadrature points
        # Equivalent to 
        #  Zk[2][r,:,:] = mats[1,r]*Zk[1][r,:,:]
        #  Zk[3][r,:,:] = mats[2,r]*permutedims(Zk[2][r,:,:],(2,1))
        #  Zk[3][r,:,:] = permutedims(Zk[3][r,:,:],(2,1))
        _, mats = m.mats
        @inbounds for r in 1:D, i1 in 1:SQ[1], j1 in 1:SB[1], j2 in 1:SB[2]
            Zk[2][r,i1,j2] += mats[1,r][i1,j1] * Zk[1][r,j1,j2]
        end
        @inbounds for r in 1:D, i1 in 1:SQ[1], i2 in 1:SQ[2], j2 in 1:SB[2]
            Zk[3][r,i1,i2] += mats[2,r][i2,j2] * Zk[2][r,i1,j2]
        end

        #B.3) Apply integration weights and jacobian
        #   Zk[3][r,I] = wq[I] * djq[I] * jq[I][l,r] * jq[I][l,k] * Zk[3][k,I]
        wq  = cell_wq[cell]  # Weights
        jq  = cell_jq[cell]  # Jacobian matrix
        djq = cell_djq[cell] # Jacobian determinant (measure)
        if (structured_mesh_opt)
            @inbounds for i1 in 1:SQ[1], i2 in 1:SQ[2]
                idx = (i2-1)*SQ[1] + i1
                for r in 1:D
                    Zk[3][r,i1,i2] = wq[idx] * Zk[3][r,i1,i2]
                end
            end
        else 
            @inbounds for i1 in 1:SQ[1], i2 in 1:SQ[2]
                idx = (i2-1)*SQ[1] + i1
                change = transpose(jq[idx]) ⋅ jq[idx] # change[r,k] = jq[I][l,r]*jq[I][l,k]
                cr = zeros(D) # c_r = jq[i1,...,id][l,r]*jq[i1,...,id][l,k]*Z[k,i1,...,id]
                for r in 1:D, k in 1:D
                cr[r] += change[r,k]*Zk[3][k,i1,i2]
                end
                for r in 1:D
                Zk[3][r,i1,i2] = wq[idx] * djq[idx] * cr[r]
                end
            end
        end 

        # B.4) Second Tensor contractions: Applying the map ∇(v)⋅ 
        map(z->fill!(z,0.0),Zk[1:2])
        @inbounds for r in 1:D, i1 in 1:SQ[1], i2 in 1:SQ[2], j2 in 1:SB[2]
            Zk[2][r,i1,j2] += mats[2,r][i2,j2] * Zk[3][r,i1,i2]
        end
        @inbounds for r in 1:D, i1 in 1:SQ[1], j1 in 1:SB[1], j2 in 1:SB[2]
            Zk[1][r,j1,j2] += mats[1,r][i1,j1] * Zk[2][r,i1,j2]
        end

        yi = fill!(similar(xi),0.0)
        # @inbounds for r in 1:D, (i,I) in enumerate(dof_map)
        #   yi[i] += Zk[1][r,I]
        # end
        yi
    end 
    cell_integrals
end 

# Parameters
D = 2                             # Problem dimension
fe_orders   = Tuple(fill(1,D))    # FE element orders
quad_orders = Tuple(fill(4,D))    # Quadrature orders 

# Setup
n = 4
domain    = repeat([0,1],D)
partition = fill(n,D)
model     = CartesianDiscreteModel(domain,partition)

Ω  = Triangulation(model)
dΩ = Measure(Ω,quad_orders[1])

g(x)  = sum(x) 
reffe = ReferenceFE(lagrangian,Float64,fe_orders)
V = FESpace(Ω,reffe;dirichlet_tags=["boundary"])
U = TrialFESpace(V,g)

# Assembled matrix
f(x) = 1.0
op = AffineFEOperator((u,v)->∫(∇(u)⋅∇(v))dΩ,v->∫(v*f)dΩ,U,V)
A = get_matrix(op)
b = get_vector(op)

# Sum-Factorization based matrix
m = SumFactorizationMap(D,fe_orders,quad_orders)
A_lazy = LazyMatrix(m,U,V,dΩ)

cell_dof_ids = get_cell_dof_ids(U)
cell_wq, cell_jq, cell_djq = A_lazy.cell_quantities

SQi = Int(sqrt(length(first((cell_wq)))))
SBi = Int(sqrt(length(first((cell_dof_ids)))))
SQ=(SQi,SQi)
SB=(SBi,SBi)
x = ones(size(b))  # Input vector of free dofs
for structured_mesh_opt in (true,false)
   count_auto = @count_ops matrix_free_vector_product_flops(SQ,SB,U,A_lazy,x;structured_mesh_opt=structured_mesh_opt)
   println("multadd64=$(count_auto.muladd64)")
   println("fma64=$(count_auto.fma64)")

   nadds=count_manual_flops_poisson_matrix_free(16,2,SQ,SB;
                                                count_only_fma_ops=true,
                                                structured_mesh_opt=structured_mesh_opt)
   nadds_and_muls=count_manual_flops_poisson_matrix_free(16,2,SQ,SB;
                                             count_only_fma_ops=false,
                                             structured_mesh_opt=structured_mesh_opt) 
   @assert nadds == count_auto.add64
   @assert nadds_and_muls == count_auto.add64 + count_auto.mul64
end    
