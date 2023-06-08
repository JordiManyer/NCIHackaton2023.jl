include("drivers.jl")

using FileIO, DrWatson
using BenchmarkTools

function benchmark_kernel(kernel, config, args; niter=100)
  times = Vector{Float64}(undef, niter)
  CUDA.@sync kernel(args...; config...)
  for it in 1:niter
    times[it] = CUDA.@elapsed begin
      for i in 1:10
        CUDA.@sync kernel(args...; config...)
      end
    end
    times[it] /= 10.0
  end
	return minimum(times), maximum(times)
end

function get_kernel_caches(::SumFactorizationMap{D,SB,SQ},config) where {D,SB,SQ}
  nb = prod(config.blocks)
  nt = prod(config.threads)
  return [CuArray(zeros(nb * nt * D * prod(SQ[1:d-1]) * prod(SB[d:D]))) for d in 1:D+1]
end

function benchmark(n::Int, D::Int, fe::Int, quad::Int)
	out         = Dict{String, Any}()
	fe_orders   = Tuple(fill(fe, D))
	quad_orders = Tuple(fill(quad, D))

	nCells, nDofs, gpu_m, gpu_cell_dof_ids, gpu_wq = setup_sumfac_gpu(n, D, fe_orders, quad_orders)

  x = CuArray(ones(size(nDofs)))
  y = CuArray(zeros(size(nDofs)))

  # v0
  config = (threads=384,blocks=80)
  gpu_Zk = get_kernel_caches(gpu_m,config)
  kernel_args = (gpu_m, nCells, y, x, gpu_cell_dof_ids, gpu_wq, gpu_Zk...)
#  kernel = @cuda name = "gpu_mul_v0" launch = false gpu_mul_v0!(kernel_args...);
#	tmin_mul_gpu0, tmax_mul_gpu0 = benchmark_kernel(kernel, config, kernel_args; niter=100)

  # v1
  config = (threads = (4, 192), blocks = 80)
  gpu_Zk = get_kernel_caches(gpu_m,config)
  kernel_args = (gpu_m, nCells, y, x, gpu_cell_dof_ids, gpu_wq, gpu_Zk...)
  kernel = @cuda name = "gpu_mul_v1" launch = false gpu_mul_v1!(kernel_args...);
	tmin_mul_gpu1, tmax_mul_gpu1 = benchmark_kernel(kernel, config, kernel_args; niter=100)

	out["D"]           = D
	out["fe_order"]    = fe
	out["quad_order"]  = quad
	out["tmin_mul_gpu0"] = tmin_mul_gpu0
	out["tmax_mul_gpu0"] = tmax_mul_gpu0
  out["tmin_mul_gpu1"] = tmin_mul_gpu1
	out["tmax_mul_gpu1"] = tmax_mul_gpu1

	return out
end

############################################################################################

n = 512
dimensions = [2]
fe_orders_vec = [1] #[2, 3, 4]
quad_orders_vec = [4] #map(o -> 2 * (o - 1), fe_orders_vec)

for D in dimensions
	for (fe, quad) in zip(fe_orders_vec, quad_orders_vec)
		out = benchmark(n, D, fe, quad)

		aux = Dict("D" => D, "fe_order" => fe, "quad_order" => quad)
		filename = datadir("benchmarks/gpu_" * savename(aux, connector = "_")) * ".bson"
		save(filename, out)
	end
end
