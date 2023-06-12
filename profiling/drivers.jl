using IterativeSolvers
using LinearAlgebra
using Gridap
using NCIHackaton2023
using CUDA

function setup_sparse(n,D,fe_orders,quad_orders)
  domain    = repeat([0,1],D)
  partition = fill(n,D)
  model     = CartesianDiscreteModel(domain,partition)

  Ω  = Triangulation(model)
  dΩ = Measure(Ω,quad_orders[1])

  g(x)  = sum(x) 
  reffe = ReferenceFE(lagrangian,Float64,fe_orders)
  V = FESpace(Ω,reffe;dirichlet_tags=["boundary"])
  U = TrialFESpace(V,g)

  f(x) = 1.0
  op = AffineFEOperator((u,v)->∫(∇(u)⋅∇(v))dΩ,v->∫(v*f)dΩ,U,V)
  A = get_matrix(op)
  b = get_vector(op)

  return A, b
end

function setup_sumfac_cpu(n,D,fe_orders,quad_orders)
  domain    = repeat([0,1],D)
  partition = fill(n,D)
  model     = CartesianDiscreteModel(domain,partition)

  Ω  = Triangulation(model)
  dΩ = Measure(Ω,quad_orders[1])

  g(x)  = sum(x) 
  reffe = ReferenceFE(lagrangian,Float64,fe_orders)
  V = FESpace(Ω,reffe;dirichlet_tags=["boundary"])
  U = TrialFESpace(V,g)

  f(x) = 1.0
  op = AffineFEOperator((u,v)->∫(∇(u)⋅∇(v))dΩ,v->∫(v*f)dΩ,U,V)
  b = get_vector(op)

  m = SumFactorizationMap(D,fe_orders,quad_orders)
  A_lazy = LazyMatrix(m,U,V,dΩ)

  return A_lazy, b
end

function setup_sumfac_gpu(n,D,fe_orders,quad_orders)
  domain    = repeat([0,1],D)
  partition = fill(n,D)
  model     = CartesianDiscreteModel(domain,partition)

  Ω  = Triangulation(model)
  dΩ = Measure(Ω,quad_orders[1])

  g(x)  = sum(x) 
  reffe = ReferenceFE(lagrangian,Float64,fe_orders)
  V = FESpace(Ω,reffe;dirichlet_tags=["boundary"])
  U = TrialFESpace(V,g)

  f(x) = 1.0
  op = AffineFEOperator((u,v)->∫(∇(u)⋅∇(v))dΩ,v->∫(v*f)dΩ,U,V)
  b = get_vector(op)

  m = SumFactorizationMap(D,fe_orders,quad_orders)
  A_lazy = LazyMatrix(m,U,V,dΩ)

  nCells = num_cells(Ω)
  nDofs  = length(b)

  gpu_m = to_gpu(m)
  gpu_cell_dof_ids = to_gpu(get_cell_dof_ids(U));

  cell_wq, cell_jq, cell_djq = A_lazy.cell_quantities
  gpu_wq = CuArray(cell_wq.value)
  #gpu_jq = CuArray(cell_jq.value)
  #gpu_djq = CuArray(cell_djq.value)

  return nCells, nDofs, gpu_m, gpu_cell_dof_ids, gpu_wq
end

function compute_solution(A::AbstractMatrix,b::AbstractVector)
  x = zeros(size(b))
  x = cg!(x,A,b)
  return x  
end

function compute_solution!(x::AbstractVector,A::AbstractMatrix,b::AbstractVector)
  fill!(x,0.0)
  x = cg!(x,A,b)
  return x  
end
