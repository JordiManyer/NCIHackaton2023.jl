"""
  This demo provides a step-by-step simplified overview of 
  sum-factorization-based matrix-vector multiplication.
"""

using Test
using LinearAlgebra

using Gridap
using Gridap.TensorValues
using Gridap.Arrays

using CUDA
using Adapt
using NCIHackaton2023

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

############################################################################################
############################################################################################
# GPU implementation

SQ = (3,3)
SB = (2,2)
struct SFMap{D,SB,SQ} end
Adapt.adapt_structure(to, m::SFMap{D,SB,SQ}) where {D,SB,SQ} = SFMap{D,SB,SQ}()
gpu_m = SFMap{D,SB,SQ}()

Adapt.@adapt_structure Gridap.Arrays.Table
cell_dof_ids = get_cell_dof_ids(U)
gpu_cell_dof_ids = Table(CuArray(cell_dof_ids.data),CuArray(cell_dof_ids.ptrs));

cell_wq, cell_jq, cell_djq = A_lazy.cell_quantities
gpu_wq  = CuArray(cell_wq.value)
gpu_jq  = CuArray(cell_jq.value)
gpu_djq = CuArray(cell_djq.value)

gpu_dof_map = CuArray(m.dof_map)

mats = zeros((2,2,3,2))
for r in 1:D, k in 1:D
  mats[r,k,:,:] = m.mats[2][r,k]
end
gpu_mats = CuArray(mats)

# Caches
gpu_xi = CuArray(zeros(prod(SB)))
gpu_Zk = [CuArray(zeros(D,SQ[1:d-1]...,SB[d:D]...)) for d in 1:D+1]

function gpu_mul!(m::SFMap{D,SB,SQ},y,x,cell_ids,dof_map,mats,wq) where {D,SB,SQ}
  num_threads = blockDim().x
  cell = threadIdx().x
  #@cuprintln(num_threads, ", ", cell)

  # A) Select cell values from input array
  xi  = CuStaticSharedArray(Float64,64) # num_threads*SB[1]*SB[2]
  ids = view(cell_ids.data,cell_ids.ptrs[cell]:cell_ids.ptrs[cell]+1)
  for (i,id) in enumerate(ids)
    xi_idx = 4*(cell-1) + i
    (id > 0) ? (xi[xi_idx] = x[id]) : (xi[xi_idx] = 0.0)
  end

  # B)
  Z1 = CuStaticSharedArray(Float64,128) # num_threads*D*SB[1]*SB[2]
  Z2 = CuStaticSharedArray(Float64,192) # num_threads*D*SQ[1]*SB[2]
  Z3 = CuStaticSharedArray(Float64,288) # num_threads*D*SQ[1]*SQ[2]

  for r in 1:D, (i,I) in enumerate(dof_map)
    j1 = I[1]; j2 = I[2];
    xi_idx = 4*(cell-1) + i
    z1_idx = (cell-1)*SB[2]*SB[1]*D + (j2-1)*SB[1]*D + (j1-1)*D + r
    Z1[z1_idx] = xi[xi_idx]
  end

  for r in 1:D, i1 in 1:SQ[1], j1 in 1:SB[1], j2 in 1:SB[2]
    z1_idx = (cell-1)*SB[2]*SB[1]*D + (j2-1)*SB[1]*D + (j1-1)*D + r
    z2_idx = (cell-1)*SB[2]*SQ[1]*D + (j2-1)*SQ[1]*D + (i1-1)*D + r
    Z2[z2_idx] += mats[1,r,i1,j1] * Z1[z1_idx]
  end

  for r in 1:D, i1 in 1:SQ[1], i2 in 1:SQ[2], j2 in 1:SB[2]
    z2_idx = (cell-1)*SB[2]*SQ[1]*D + (j2-1)*SQ[1]*D + (i1-1)*D + r
    z3_idx = (cell-1)*SQ[2]*SQ[1]*D + (i2-1)*SQ[1]*D + (i1-1)*D + r
    Z3[z3_idx] += mats[2,r,i2,j2] * Z2[z2_idx]
  end

  for r in 1:D, i1 in 1:SQ[1], i2 in 1:SQ[2]
    idx    = (i2-1)*SQ[1] + i1
    z3_idx = (cell-1)*SQ[2]*SQ[1]*D + (i2-1)*SQ[1]*D + (i1-1)*D + r
    Z3[z3_idx] *= wq[idx]
  end

  fill!(Z2,0.0)
  for r in 1:D, i1 in 1:SQ[1], i2 in 1:SQ[2], j2 in 1:SB[2]
    z2_idx = (cell-1)*SB[2]*SQ[1]*D + (j2-1)*SQ[1]*D + (i1-1)*D + r
    z3_idx = (cell-1)*SQ[2]*SQ[1]*D + (i2-1)*SQ[1]*D + (i1-1)*D + r
    Z2[z2_idx] += mats[2,r,i2,j2] * Z3[z3_idx]
  end

  fill!(Z1,0.0)
  for r in 1:D, i1 in 1:SQ[1], j1 in 1:SB[1], j2 in 1:SB[2]
    z1_idx = (cell-1)*SB[2]*SB[1]*D + (j2-1)*SB[1]*D + (j1-1)*D + r
    z2_idx = (cell-1)*SB[2]*SQ[1]*D + (j2-1)*SQ[1]*D + (i1-1)*D + r
    Z1[z1_idx] += mats[1,r,i1,j1] * Z2[z2_idx]
  end

  fill!(xi,0.0)
  for r in 1:D, (i,I) in enumerate(dof_map)
    j1 = I[1]; j2 = I[2];
    xi_idx = 4*(cell-1) + i
    z1_idx = (cell-1)*SB[2]*SB[1]*D + (j2-1)*SB[1]*D + (j1-1)*D + r
    xi[xi_idx] += Z1[z1_idx]
  end

  fill!(y,0.0)
  for (i,id) in enumerate(ids)
    xi_idx = 4*(cell-1) + i
    (id > 0) && (y[id] += xi[xi_idx])
  end

  return
end


x = CuArray(ones(size(b)))
y = CuArray(zeros(size(b)))
nt = num_cells(Ω)
@cuda threads=nt gpu_mul!(gpu_m,
               y,
               x,
               gpu_cell_dof_ids,
               gpu_dof_map,
               gpu_mats,
               gpu_wq);

cpu_y = Array(y)
y_ref = A*ones(length(b))
