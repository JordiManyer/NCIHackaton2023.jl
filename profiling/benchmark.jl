
include("drivers.jl")

using FileIO, DrWatson
using BenchmarkTools

function benchmark_mul(x::AbstractVector,A::AbstractMatrix,b::AbstractVector;niter=100)
  times = Vector{Float64}(undef,niter)
  mul!(x,A,b) # warmup
  for it in 1:niter
    times[it] = @elapsed mul!(x,A,b)
  end
  return minimum(times), maximum(times)
end

function benchmark_cg(x::AbstractVector,A::AbstractMatrix,b::AbstractVector;niter=10)
  times = Vector{Float64}(undef,niter)
  fill!(x,0.0)
  cg!(x,A,b,maxiter=10) # warmup
  for it in 1:niter
    fill!(x,0.0)
    times[it] = @elapsed cg!(x,A,b,maxiter=10)
  end
  return minimum(times), maximum(times)
end

function benchmark(D::Int,fe::Int,quad::Int)
  out = Dict{String,Any}()
  fe_orders   = Tuple(fill(fe,D))
  quad_orders = Tuple(fill(quad,D))

  A,b = setup_sparse(D,fe_orders,quad_orders)
  x = zeros(size(b))
  tmin_mul_sp, tmax_mul_sp = benchmark_mul(x,A,b;niter=100)
  tmin_cg_sp, tmax_cg_sp   = benchmark_cg(x,A,b;niter=10)

  A_lazy,b = setup_sumfac_cpu(D,fe_orders,quad_orders)
  x = zeros(size(b))
  tmin_mul_sf, tmax_mul_sf = benchmark_mul(x,A_lazy,b;niter=100)
  tmin_cg_sf, tmax_cg_sf   = benchmark_cg(x,A_lazy,b;niter=10)

  out["D"] = D
  out["fe_order"]    = fe
  out["quad_order"]  = quad
  out["tmin_mul_sp"] = tmin_mul_sp
  out["tmax_mul_sp"] = tmax_mul_sp
  out["tmin_cg_sp"]  = tmin_cg_sp
  out["tmax_cg_sp"]  = tmax_cg_sp
  out["tmin_mul_sf"] = tmin_mul_sf
  out["tmax_mul_sf"] = tmax_mul_sf
  out["tmin_cg_sf"]  = tmin_cg_sf
  out["tmax_cg_sf"]  = tmax_cg_sf

  return out
end

############################################################################################

dimensions = [2,3]
fe_orders_vec = [2,3,4,5,6,7]
quad_orders_vec = map(o->2*(o-1),fe_orders_vec)

for D in dimensions
  for (fe,quad) in zip(fe_orders_vec,quad_orders_vec)
    out = benchmark(D,fe,quad)

    aux = Dict("D" => D,"fe_order" => fe,"quad_order" => quad)
    filename = datadir("benchmarks/"*savename(aux,connector="_"))*".bson"
    save(filename,out)
  end
end
