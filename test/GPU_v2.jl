"""
  SUMFAC-GPU v2
  Basic kernel. Starting to take advantage of Shared & Constant memory. 
"""

using Test
using LinearAlgebra

using Gridap
using Gridap.TensorValues
using Gridap.Arrays

using CUDA
using Adapt
using NCIHackaton2023

import NCIHackaton2023: @index_to_tuple

# Parameters
D           = 2                    # Problem dimension
fe_orders   = Tuple(fill(4, D))    # FE element orders
quad_orders = Tuple(fill(6, D))    # Quadrature orders 

# Setup
n         = 512
domain    = repeat([0, 1], D)
partition = fill(n, D)
model     = CartesianDiscreteModel(domain, partition)

Ω  = Triangulation(model)
dΩ = Measure(Ω, quad_orders[1])

g(x)  = 0.0
reffe = ReferenceFE(lagrangian, Float64, fe_orders)
V     = FESpace(Ω, reffe; dirichlet_tags = ["boundary"])
U     = TrialFESpace(V, g)

# Assembled matrix
f(x) = 1.0
op = AffineFEOperator((u, v) -> ∫(∇(u) ⋅ ∇(v))dΩ, v -> ∫(v * f)dΩ, U, V)
A = get_matrix(op)
b = get_vector(op)

# Sum-Factorization based matrix
m = SumFactorizationMap(D, fe_orders, quad_orders)
A_lazy = LazyMatrix(m, U, V, dΩ)

D, SB, SQ = get_dimensional_parameters(m)

############################################################################################
############################################################################################
# GPU implementation
nCells = num_cells(Ω)

gpu_m = to_gpu(m)
gpu_cell_dof_ids = to_gpu(get_cell_dof_ids(U));

cell_wq, cell_jq, cell_djq = A_lazy.cell_quantities
gpu_wq = CuArray(cell_wq.value)
gpu_jq = CuArray(cell_jq.value)
gpu_djq = CuArray(cell_djq.value)

# Caches
D, SB, SQ = get_dimensional_parameters(m)

# Comparison CPU vs GPU
x_ref = ones(size(b))
x = CuArray(x_ref)
y = CuArray(zeros(size(b)))
kernel_args = (gpu_m, nCells, y, x, gpu_cell_dof_ids, gpu_wq, gpu_Zk...)

kernel = @cuda name = "gpu_mul_v2" launch = false gpu_mul_v2!(kernel_args...);
config = launch_configuration(kernel.fun)


x_ref = ones(size(b))
x = CuArray(x_ref)
y = CuArray(zeros(size(b)))
mem = 16*D*(prod(SB) + SQ[1]*SB[2] + prod(SQ))*sizeof(Float64)
@cuda threads = (32, 16) blocks = 80 shmem=mem gpu_mul_v2!(gpu_m, nCells, y, x, gpu_cell_dof_ids, gpu_wq)

Array(y)


y_ref = zeros(length(b))
@elapsed mul!(y_ref, A_lazy, x_ref)

cpu_y = Array(y)
cpu_y ≈ y_ref


function gpu_mul_v2!(m::SumFactorizationMap{D, SB, SQ}, nCells, y, x, cell_ids, wq) where {D, SB, SQ}
	dof_map, mats = m.dof_map, m.mats
  CUDA.Const(mats)
  CUDA.Const(dof_map)
	#thread  = (blockIdx().x - 1) * blockDim().y + threadIdx().y
  thread  = threadIdx().y
	y_start = threadIdx().x
	y_step  = blockDim().x
	
  s1 = blockDim().y*D*SB[1]*SB[2]; s2 = blockDim().y*D*SQ[1]*SB[2]; s3 = blockDim().y*D*SQ[1]*SQ[2];
  Z  = @cuDynamicSharedMem(Float64,s1+s2+s3)
  Z1 = view(Z,1:s1)
  Z2 = view(Z,s1+1:s1+s2)
  Z3 = view(Z,s1+s2+1:s1+s2+s3)

	cell = (blockIdx().x - 1) * blockDim().y + threadIdx().y
	while cell <= nCells
    #@cuprintln(" > thread = (", threadIdx().x, ",", threadIdx().y ,") - cell = ", cell)
		# Scatter
		ids = view(cell_ids.data, cell_ids.ptrs[cell]:cell_ids.ptrs[cell+1]-1)

		idy = y_start
		s = D * SB[1] * SB[2]
		while idy <= s
			r, i = @index_to_tuple(idy, D, SB[1] * SB[2])
			I = dof_map[i]
			j1 = I[1]
			j2 = I[2]
			id = ids[i]
			z1_idx = (thread - 1) * s + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r

			Z1[z1_idx] = x[max(id, 1)] * (id > 0)
			idy += y_step
		end
		CUDA.sync_threads()

		# Forward pass
		idy = y_start
		s = D * SQ[1] * SB[2]
		while idy <= s
			r, i1, j2 = @index_to_tuple(idy, D, SQ[1], SB[2])

			z2_idx = (thread - 1) * s + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z2[z2_idx] = 0.0
			for j1 in 1:SB[1]
				z1_idx = (thread - 1) * SB[2] * SB[1] * D + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
				Z2[z2_idx] += mats[i1, j1, 1, r] * Z1[z1_idx]
			end
			idy += y_step
		end
		CUDA.sync_threads()

		idy = y_start
		s = D * SQ[1] * SQ[2]
		while idy <= s
			r, i1, i2 = @index_to_tuple(idy, D, SQ[1], SQ[2])

			z3_idx = (thread - 1) * s + (i2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z3[z3_idx] = 0.0
			for j2 in 1:SB[2]
				z2_idx = (thread - 1) * SB[2] * SQ[1] * D + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z3[z3_idx] += mats[i2, j2, 2, r] * Z2[z2_idx]
			end

			idy += y_step
		end
		CUDA.sync_threads()

		# Apply weights 
		idy = y_start
		s = D * SQ[1] * SQ[2]
		while idy <= s
			r, i1, i2 = @index_to_tuple(idy, D, SQ[1], SQ[2])
			idx = (i2 - 1) * SQ[1] + i1
			z3_idx = (thread - 1) * s + (idx - 1) * D + r
			Z3[z3_idx] *= wq[idx]
			idy += y_step
		end
		CUDA.sync_threads()

		# Backward pass
		idy = y_start
		s = D * SQ[1] * SB[2]
		while idy <= s
			r, i1, j2 = @index_to_tuple(idy, D, SQ[1], SB[2])
			z2_idx = (thread - 1) * s + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z2[z2_idx] = 0.0
			for i2 in 1:SQ[2]
				z3_idx = (thread - 1) * SQ[2] * SQ[1] * D + (i2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z2[z2_idx] += mats[i2, j2, 2, r] * Z3[z3_idx]
			end
			idy += y_step
		end
		CUDA.sync_threads()

		idy = y_start
		s = D * SB[1] * SB[2]
		while idy <= s
			r, j1, j2 = @index_to_tuple(idy, D, SB[1], SB[2])
			z1_idx = (thread - 1) * s + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
			Z1[z1_idx] = 0.0
			for i1 in 1:SQ[1]
				z2_idx = (thread - 1) * SB[2] * SQ[1] * D + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z1[z1_idx] += mats[i1, j1, 1, r] * Z2[z2_idx]
			end
			idy += y_step
		end
		CUDA.sync_threads()

		# Assemble
		idy = y_start
		s = D * SB[1] * SB[2]
		while idy <= s
			r, i = @index_to_tuple(idy, D, SB[1] * SB[2])
			I = dof_map[i]
			j1 = I[1]
			j2 = I[2]
			id = ids[i]
			z1_idx = (thread - 1) * s + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
			if id > 0
				CUDA.@atomic y[id] += Z1[z1_idx]
			end
			idy += y_step
		end
		CUDA.sync_threads()

		cell += gridDim().x * blockDim().y
	end

	return
end

############################################################################################

function my_test_kernel(v)
  Z = @cuDynamicSharedMem(Float64,blockDim().y)
  Q = @cuDynamicSharedMem(Float64,blockDim().y)

  tid = threadIdx().y
  CUDA.@atomic Z[tid] += 1.0
  CUDA.@atomic Q[tid] += 1.0

  if threadIdx().x == 1
    v[tid] = Z[tid] + Q[tid]
  end

  return 
end

v = CuArray(zeros(10))
mem = sizeof(Float64)*10*2
@cuda threads=(10,10) shmem=mem my_test_kernel(v);

Array(v)