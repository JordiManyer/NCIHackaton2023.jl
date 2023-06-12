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

function get_mats(m::SumFactorizationMap{D, SB, SQ}) where {D, SB, SQ}
	ji_mats = zeros((SB[1],D,SQ[1],D))
	ij_mats = zeros((SQ[1],D,SB[1],D))
	for r in 1:D, k in 1:D, i in 1:SQ[1], j in 1:SB[1]
		ji_mats[j,r,i,k] = m.mats[2][k,r][i,j]
		ij_mats[i,r,j,k] = m.mats[2][k,r][i,j]
	end
	return CuArray(ij_mats), CuArray(ji_mats)
end

############################################################################################
############################################################################################
# GPU implementation
nCells = num_cells(Ω)

gpu_m = to_gpu(m)
gpu_cell_dof_ids = to_gpu(get_cell_dof_ids(U));
ij_mats, ji_mats = get_mats(m)

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
kernel_args = (gpu_m, nCells, y, x, gpu_cell_dof_ids, gpu_wq, ij_mats, ji_mats,Val((16,2)))

kernel = @cuda name = "gpu_mul_v4" launch = false gpu_mul_v4!(kernel_args...);
config = launch_configuration(kernel.fun)

mem = 16*D*(max(prod(SB),prod(SQ)) + SQ[1]*SB[2])*sizeof(Float64)
config = (threads=(16,2),blocks=80)
kernel(kernel_args...;config...)
GC.gc(true); CUDA.reclaim(); 

time = CUDA.@elapsed kernel(kernel_args...;config...)
println(time)

#GC.gc(true)
#CUDA.reclaim()

y_ref = zeros(length(b))
@elapsed mul!(y_ref, A_lazy, x_ref)

cpu_y = Array(y)
cpu_y ≈ y_ref

#niter = 100
#time = NCIHackaton2023.benchmark_kernel(kernel, config, kernel_args, niter)
#prinltn("Time = ", time)
