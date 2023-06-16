
using CUDA
using Plots

function benchmark_dot(niter,x,y)
  dot(x,y)
  times = zeros(niter)
  for iter in 1:niter
    times[iter] = @elapsed dot(x,y)
  end
  minimum(times), sum(times)/niter, maximum(times)
end

function benchmark_cuda_dot(niter,x,y)
  CUDA.@sync dot(x,y)
  times = zeros(niter)
  for iter in 1:niter
    times[iter] = CUDA.@sync CUDA.@elapsed dot(x,y)
  end
  minimum(times), sum(times)/niter, maximum(times)
end

niter = 50
sizes = [10000,100000,1000000,10000000,10000000]

t1 = zeros(length(sizes))
t2 = zeros(length(sizes))
for (i,n) in enumerate(sizes)
  x_cpu, y_cpu = randn(n), randn(n)
  x_gpu, y_gpu = CuArray(x_cpu), CuArray(y_cpu)

  t1[i], _, _ = benchmark_dot(niter,x_cpu,y_cpu)
  t2[i], _, _ = benchmark_cuda_dot(niter,x_gpu,y_gpu)
end

plt = plot(xlabel="log(n)",ylabel="log(t)");
plot!(log10.(sizes),log10.(t1),labels="cpu");
plot!(log10.(sizes),log10.(t2),labels="gpu");
@show plt
