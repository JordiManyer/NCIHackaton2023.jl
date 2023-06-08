
# Index management
macro index_to_tuple(idx, D1, D2)
	return esc(:(($idx - 1) % $D1 + 1, ($idx - 1) ÷ $D1 + 1))
end

macro index_to_tuple(idx, D1, D2, D3)
	return esc(:(($idx - 1) % $D1 + 1, (($idx - 1) ÷ $D1) % $D2 + 1, ($idx - 1) ÷ ($D1 * $D2) + 1))
end

# Porting Gridap strutures to GPUs
Adapt.@adapt_structure Gridap.Arrays.Table
function to_gpu(t::Gridap.Arrays.Table)
	return Table(CuArray(t.data), CuArray(t.ptrs))
end

# Benchmark kernel 
function benchmark_kernel(kernel, config, args, niter)
	time = CUDA.@elapsed begin
		for i in 1:niter
			CUDA.@sync kernel(args...; config...)
		end
	end
	return time / niter
end

# if count_only_fma_ops==true, only count FMA operations 
# else count all flops 

# if structured_mesh_opt==true, assumes Cartesian mesh optimizations
# else no Cartesian mesh optimizations are applied
function count_manual_flops_poisson_matrix_free(ncells,D,SQ,SB;
                                                count_only_fma_ops=true,
                                                structured_mesh_opt=true)

  # B.2) (first tensor contractions)
  b_2_flops = 0
  for d=1:D
    b_2_flops += prod((D,SQ[1:d]...,SB[d:D]...))
  end
  if !(count_only_fma_ops)
    b_2_flops=2*b_2_flops
  end  

  # B.3) Apply integration weights and jacobian
  b3_flops = 0
  if !(structured_mesh_opt)
	# adds
	b3_flops = (D-1)*prod((D for i=1:D)) + D*D
    if !(count_only_fma_ops)
		                     # muls only
       b3_flops=2*b3_flops + prod((D for i=1:D)) + 2*D
    end
  else 
	if !(count_only_fma_ops)
	   b3_flops=D
	end
  end 	
  b3_flops = b3_flops * prod(SQ)
   
  # B.4) (second tensor contractions)
  b_4_flops = 0 
  for d=1:D 
    b_4_flops += prod((D,SQ[1:d]...,SB[d:D]...))
  end 
  if !(count_only_fma_ops)
    b_4_flops=2*b_4_flops
  end
  total_flops_x_cell = b_2_flops + b3_flops + b_4_flops
  return ncells*total_flops_x_cell
end

