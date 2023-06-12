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
kernel_args = (gpu_m, nCells, y, x, gpu_cell_dof_ids, gpu_wq)

kernel = @cuda name = "gpu_mul_v2" launch = false gpu_mul_v2!(kernel_args...);
config = launch_configuration(kernel.fun)

mem = 16*D*(prod(SB) + SQ[1]*SB[2] + prod(SQ))*sizeof(Float64)
config = (threads=(32,16),blocks=160,shmem=mem)
kernel(kernel_args...;config...)

y_ref = zeros(length(b))
@elapsed mul!(y_ref, A_lazy, x_ref)

cpu_y = Array(y)
cpu_y ≈ y_ref

niter = 100
time = NCIHackaton2023.benchmark_kernel(kernel, config, kernel_args, niter)
