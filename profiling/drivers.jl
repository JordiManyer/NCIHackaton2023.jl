using IterativeSolvers
using Gridap
using NCIHackaton2023

function setup_sparse(D,fe_orders,quad_orders)
  n = 10
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

function setup_sumfac_cpu(D,fe_orders,quad_orders)
  n = 10
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

function compute_solution(A::AbstractMatrix,b::AbstractVector)
  x = zeros(size(b))
  x = cg!(x,A,b)
  return x  
end
