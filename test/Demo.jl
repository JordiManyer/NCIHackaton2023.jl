
using BenchmarkTools
using LinearAlgebra
using IterativeSolvers

using Gridap
using Gridap.TensorValues
using Gridap.Arrays

using NCIHackaton2023

# Parameters
D = 2                             # Problem dimension
fe_orders   = Tuple(fill(4,D))    # FE element orders
quad_orders = Tuple(fill(6,D))    # Quadrature orders 

# Setup
n = 20
domain    = repeat([0,1],D)
partition = fill(4,D)
model     = CartesianDiscreteModel(domain,partition)

Ω  = Triangulation(model)
dΩ = Measure(Ω,quad_orders[1])

g(x)  = sum(x) 
reffe = ReferenceFE(lagrangian,Float64,fe_orders)
V = FESpace(Ω,reffe;dirichlet_tags=["boundary"])
U = TrialFESpace(V,g)

############################################################################################

# Assembled matrix
f(x) = 1.0
op = AffineFEOperator((u,v)->∫(∇(u)⋅∇(v))dΩ,v->∫(v*f)dΩ,U,V)
A = get_matrix(op)
b = get_vector(op)

# Sum-Factorization based matrix
m = SumFactorizationMap(D,fe_orders,quad_orders)
A_lazy = LazyMatrix(m,U,V,dΩ)

############################################################################################

# Check implementation
y  = ones(size(b))
x1 = zeros(size(b)); mul!(x1,A,y)
x2 = zeros(size(b)); mul!(x2,A_lazy,y)
x1 ≈ x2

############################################################################################

# Benchmark implementation vs sparse mat-vec multiplication
@benchmark mul!($x1,$A,$y)
@benchmark mul!($x2,$A_lazy,$y)
