
using Gridap

for D in [2,3]
  n = 10
  domain    = repeat([0,1],D)
  partition = fill(n,D)
  model     = CartesianDiscreteModel(domain,partition)

  Ω  = Triangulation(model)
  dΩ = Measure(Ω,3)

  g(x)  = sum(x) 
  reffe = ReferenceFE(lagrangian,Float64,2)
  V = FESpace(Ω,reffe;dirichlet_tags=["boundary"])
  U = TrialFESpace(V,g)

  f(x) = 1.0
  op = AffineFEOperator((u,v)->∫(∇(u)⋅∇(v))dΩ,v->∫(v*f)dΩ,U,V)
  A = get_matrix(op)
  b = get_vector(op)
end
