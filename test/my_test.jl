using CUDA
using StaticArrays

function my_kernel(v)
  Z = MVector{3,Float64}(0.0,0.0,0.0)
  Z[threadIdx().x] += 1
  v[threadIdx().x] = Z[threadIdx().x]
  return
end

function main()
  v = CuArray(zeros(3));
  @cuda threads=3 my_kernel(v)
  v_cpu = Array(v)
  println(v_cpu)
end

main()

