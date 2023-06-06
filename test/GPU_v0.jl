"""
  SUMFAC-GPU v0
  Basic kernel. In this version, everything comes from global memory. 
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
n = 512
domain    = repeat([0,1],D)
partition = fill(n,D)
model     = CartesianDiscreteModel(domain,partition)

Ω  = Triangulation(model)
dΩ = Measure(Ω,quad_orders[1])

g(x)  = 0.0 
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
nCells = num_cells(Ω)
nt = 256
nb = 8

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

dof_map = m.dof_map
gpu_dof_map = CuArray(m.dof_map)

mats = zeros((3,2,D,D))
for r in 1:D, k in 1:D
  mats[:,:,r,k] = m.mats[2][r,k]
end
gpu_mats = CuArray(mats)

# Caches
gpu_Zk = [CuArray(zeros(nb*nt*D*prod(SQ[1:d-1])*prod(SB[d:D]))) for d in 1:D+1]

"""
# TODO: 
1 - Check if we can use StaticArrays for Zk, check if we can fit it in shared mem/registers
2 - permute mat dims so that first indexes are the ones that change the most
3 - start profiling the v0
"""

function gpu_mul!(m::SFMap{D,SB,SQ},nCells,y,x,cell_ids,dof_map,mats,wq,Z1,Z2,Z3) where {D,SB,SQ}
  thread = (blockIdx().x-1) * blockDim().x + threadIdx().x

  cell = thread
  while cell <= nCells

    # Scatter
    ids = view(cell_ids.data,cell_ids.ptrs[cell]:cell_ids.ptrs[cell+1]-1)
    for (i,I) in enumerate(dof_map)
      j1 = I[1]; j2 = I[2];
      id = ids[i]
      for r in 1:D
        z1_idx = (thread-1)*SB[2]*SB[1]*D + (j2-1)*SB[1]*D + (j1-1)*D + r
        if id > 0
          Z1[z1_idx] = x[id]
        else
          Z1[z1_idx] = 0.0
        end
      end
    end

    # Forward pass
    for r in 1:D, i1 in 1:SQ[1], j2 in 1:SB[2]
      z2_idx = (thread-1)*SB[2]*SQ[1]*D + (j2-1)*SQ[1]*D + (i1-1)*D + r
      Z2[z2_idx] = 0.0
      for j1 in 1:SB[1]
        z1_idx = (thread-1)*SB[2]*SB[1]*D + (j2-1)*SB[1]*D + (j1-1)*D + r
        Z2[z2_idx] += mats[i1,j1,1,r] * Z1[z1_idx]
      end
    end

    for r in 1:D, i1 in 1:SQ[1], i2 in 1:SQ[2]
      z3_idx = (thread-1)*SQ[2]*SQ[1]*D + (i2-1)*SQ[1]*D + (i1-1)*D + r
      Z3[z3_idx] = 0.0
      for j2 in 1:SB[2]
        z2_idx = (thread-1)*SB[2]*SQ[1]*D + (j2-1)*SQ[1]*D + (i1-1)*D + r
        Z3[z3_idx] += mats[i2,j2,2,r] * Z2[z2_idx]
      end
    end

    # Apply weights 
    for r in 1:D, i1 in 1:SQ[1], i2 in 1:SQ[2]
      idx    = (i2-1)*SQ[1] + i1
      z3_idx = (thread-1)*SQ[2]*SQ[1]*D + (i2-1)*SQ[1]*D + (i1-1)*D + r
      Z3[z3_idx] *= wq[idx]
    end

    # Backward pass
    for r in 1:D, i1 in 1:SQ[1], j2 in 1:SB[2]
      z2_idx = (thread-1)*SB[2]*SQ[1]*D + (j2-1)*SQ[1]*D + (i1-1)*D + r
      Z2[z2_idx] = 0.0
      for i2 in 1:SQ[2]
        z3_idx = (thread-1)*SQ[2]*SQ[1]*D + (i2-1)*SQ[1]*D + (i1-1)*D + r
        Z2[z2_idx] += mats[i2,j2,2,r] * Z3[z3_idx]
      end
    end

    for r in 1:D, j1 in 1:SB[1], j2 in 1:SB[2]
      z1_idx = (thread-1)*SB[2]*SB[1]*D + (j2-1)*SB[1]*D + (j1-1)*D + r
      Z1[z1_idx] = 0.0
      for i1 in 1:SQ[1]
        z2_idx = (thread-1)*SB[2]*SQ[1]*D + (j2-1)*SQ[1]*D + (i1-1)*D + r
        Z1[z1_idx] += mats[i1,j1,1,r] * Z2[z2_idx]
      end
    end

    # Assemble
    for (i,I) in enumerate(dof_map)
      j1 = I[1]; j2 = I[2];
      id = ids[i]
      val = 0.0
      for r in 1:D
        z1_idx = (thread-1)*SB[2]*SB[1]*D + (j2-1)*SB[1]*D + (j1-1)*D + r
        val += Z1[z1_idx]
      end
      if id > 0
        CUDA.@atomic y[id] += val
      end
    end

    cell += gridDim().x * blockDim().x
  end

  return
end


x_ref = randn(size(b))
x = CuArray(x_ref)
y = CuArray(zeros(size(b)))
@cuda blocks=nb threads=nt gpu_mul!(gpu_m,nCells,
               y,
               x,
               gpu_cell_dof_ids,
               gpu_dof_map,
               gpu_mats,
               gpu_wq,
               gpu_Zk[1],gpu_Zk[2],gpu_Zk[3]);

cpu_y = Array(y)

y_ref = zeros(length(b))
mul!(y_ref,A_lazy,x_ref)

cpu_y ≈ y_ref

"""
function gpu_mul!()
  idx = (blockIdx().x-1)*blockDim().x + threadIdx().x
  for color in 1:num_colors
    while idx <= length(cells_per_color[color])
      cell = cells_per_color[color][idx]
      do_stuff
      assemble
      idx = idx + num_threads*num_blocks
    end
    sync_threads()
  end
end
"""