
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
