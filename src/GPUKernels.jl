
"""
  SUMFAC-GPU Kernel v0
  Basic kernel. In this version, everything comes from global memory. Computations
  are parallelized cell-wise, i.e each cell is handled by a different thread. 
"""
function gpu_mul_v0!(m::SumFactorizationMap{D, SB, SQ}, nCells, y, x, cell_ids, wq, Zk...) where {D, SB, SQ}
	dof_map, mats = m.dof_map, m.mats
	thread = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	Z1, Z2, Z3 = Zk

	cell = thread
	while cell <= nCells

		# Scatter
		ids = view(cell_ids.data, cell_ids.ptrs[cell]:cell_ids.ptrs[cell+1]-1)
		for (i, I) in enumerate(dof_map)
			j1 = I[1]
			j2 = I[2]
			id = ids[i]
			for r in 1:D
				z1_idx = (thread - 1) * SB[2] * SB[1] * D + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
				if id > 0
					Z1[z1_idx] = x[id]
				else
					Z1[z1_idx] = 0.0
				end
			end
		end

		# Forward pass
		for r in 1:D, i1 in 1:SQ[1], j2 in 1:SB[2]
			z2_idx = (thread - 1) * SB[2] * SQ[1] * D + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z2[z2_idx] = 0.0
			for j1 in 1:SB[1]
				z1_idx = (thread - 1) * SB[2] * SB[1] * D + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
				Z2[z2_idx] += mats[i1, j1, 1, r] * Z1[z1_idx]
			end
		end

		for r in 1:D, i1 in 1:SQ[1], i2 in 1:SQ[2]
			z3_idx = (thread - 1) * SQ[2] * SQ[1] * D + (i2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z3[z3_idx] = 0.0
			for j2 in 1:SB[2]
				z2_idx = (thread - 1) * SB[2] * SQ[1] * D + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z3[z3_idx] += mats[i2, j2, 2, r] * Z2[z2_idx]
			end
		end

		# Apply weights 
		for r in 1:D, i1 in 1:SQ[1], i2 in 1:SQ[2]
			idx = (i2 - 1) * SQ[1] + i1
			z3_idx = (thread - 1) * SQ[2] * SQ[1] * D + (i2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z3[z3_idx] *= wq[idx]
		end

		# Backward pass
		for r in 1:D, i1 in 1:SQ[1], j2 in 1:SB[2]
			z2_idx = (thread - 1) * SB[2] * SQ[1] * D + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z2[z2_idx] = 0.0
			for i2 in 1:SQ[2]
				z3_idx = (thread - 1) * SQ[2] * SQ[1] * D + (i2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z2[z2_idx] += mats[i2, j2, 2, r] * Z3[z3_idx]
			end
		end

		for r in 1:D, j1 in 1:SB[1], j2 in 1:SB[2]
			z1_idx = (thread - 1) * SB[2] * SB[1] * D + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
			Z1[z1_idx] = 0.0
			for i1 in 1:SQ[1]
				z2_idx = (thread - 1) * SB[2] * SQ[1] * D + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z1[z1_idx] += mats[i1, j1, 1, r] * Z2[z2_idx]
			end
		end

		# Assemble
		for (i, I) in enumerate(dof_map)
			j1 = I[1]
			j2 = I[2]
			id = ids[i]
			val = 0.0
			for r in 1:D
				z1_idx = (thread - 1) * SB[2] * SB[1] * D + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
				val += Z1[z1_idx]
			end
			if id > 0
				CUDA.@atomic y[id] += val
			end
		end

		cell += gridDim().x * blockDim().x
	end

	return
end


"""
  SUMFAC-GPU Kernel v1
  Second version. Still reading everything from global memory. Added a new layer of parallelism,
  in which each cell is handled by multiple threads at the same time.
"""
function gpu_mul_v1!(m::SumFactorizationMap{D, SB, SQ}, nCells, y, x, cell_ids, wq, Zk...) where {D, SB, SQ}
	dof_map, mats = m.dof_map, m.mats
	thread = (blockIdx().x - 1) * blockDim().y + threadIdx().y
	y_start = threadIdx().x
	y_step = blockDim().x
	Z1, Z2, Z3 = Zk

	cell = thread
	while cell <= nCells

		# Scatter
		ids = view(cell_ids.data, cell_ids.ptrs[cell]:cell_ids.ptrs[cell+1]-1)

		idy = y_start
		s = D * SB[1] * SB[2]
		while idy <= s
			r, i = @index_to_tuple(idy, D, SB[1] * SB[2])
			I = dof_map[i]
			j1 = I[1]
			j2 = I[2]
			id = ids[i]
			z1_idx = (thread - 1) * s + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r

			Z1[z1_idx] = x[max(id, 1)] * (id > 0)
			idy += y_step
		end
		CUDA.sync_threads()

		# Forward pass
		idy = y_start
		s = D * SQ[1] * SB[2]
		while idy <= s
			r, i1, j2 = @index_to_tuple(idy, D, SQ[1], SB[2])

			z2_idx = (thread - 1) * s + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z2[z2_idx] = 0.0
			for j1 in 1:SB[1]
				z1_idx = (thread - 1) * SB[2] * SB[1] * D + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
				Z2[z2_idx] += mats[i1, j1, 1, r] * Z1[z1_idx]
			end
			idy += y_step
		end
		CUDA.sync_threads()

		idy = y_start
		s = D * SQ[1] * SQ[2]
		while idy <= s
			r, i1, i2 = @index_to_tuple(idy, D, SQ[1], SQ[2])

			z3_idx = (thread - 1) * s + (i2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z3[z3_idx] = 0.0
			for j2 in 1:SB[2]
				z2_idx = (thread - 1) * SB[2] * SQ[1] * D + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z3[z3_idx] += mats[i2, j2, 2, r] * Z2[z2_idx]
			end

			idy += y_step
		end
		CUDA.sync_threads()

		# Apply weights 
		idy = y_start
		s = D * SQ[1] * SQ[2]
		while idy <= s
			r, i1, i2 = @index_to_tuple(idy, D, SQ[1], SQ[2])
			idx = (i2 - 1) * SQ[1] + i1
			z3_idx = (thread - 1) * s + (idx - 1) * D + r
			Z3[z3_idx] *= wq[idx]
			idy += y_step
		end
		CUDA.sync_threads()

		# Backward pass
		idy = y_start
		s = D * SQ[1] * SB[2]
		while idy <= s
			r, i1, j2 = @index_to_tuple(idy, D, SQ[1], SB[2])
			z2_idx = (thread - 1) * s + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z2[z2_idx] = 0.0
			for i2 in 1:SQ[2]
				z3_idx = (thread - 1) * SQ[2] * SQ[1] * D + (i2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z2[z2_idx] += mats[i2, j2, 2, r] * Z3[z3_idx]
			end
			idy += y_step
		end
		CUDA.sync_threads()

		idy = y_start
		s = D * SB[1] * SB[2]
		while idy <= s
			r, j1, j2 = @index_to_tuple(idy, D, SB[1], SB[2])
			z1_idx = (thread - 1) * s + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
			Z1[z1_idx] = 0.0
			for i1 in 1:SQ[1]
				z2_idx = (thread - 1) * SB[2] * SQ[1] * D + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z1[z1_idx] += mats[i1, j1, 1, r] * Z2[z2_idx]
			end
			idy += y_step
		end
		CUDA.sync_threads()

		# Assemble
		idy = y_start
		s = D * SB[1] * SB[2]
		while idy <= s
			r, i = @index_to_tuple(idy, D, SB[1] * SB[2])
			I = dof_map[i]
			j1 = I[1]
			j2 = I[2]
			id = ids[i]
			z1_idx = (thread - 1) * s + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
			if id > 0
				CUDA.@atomic y[id] += Z1[z1_idx]
			end
			idy += y_step
		end
		CUDA.sync_threads()

		cell += gridDim().x * blockDim().y
	end

	return
end

"""
  SUMFAC-GPU Kernel v2
  Third version. Started leveraging shared and constant memory to reduce latency.
	 - We process blockDim().y cells at the same time.
	 - We use blockDim().x threads to process each cell.
"""
function gpu_mul_v2!(m::SumFactorizationMap{D, SB, SQ}, nCells, y, x, cell_ids, wq) where {D, SB, SQ}
	dof_map, mats = m.dof_map, m.mats
  CUDA.Const(mats)
  CUDA.Const(dof_map)
  tidy = threadIdx().y
	tidx = threadIdx().x
	tidx_step  = blockDim().x
	
  s1 = blockDim().y*D*SB[1]*SB[2]; s2 = blockDim().y*D*SQ[1]*SB[2]; s3 = blockDim().y*D*SQ[1]*SQ[2];
  Z  = @cuDynamicSharedMem(Float64,s1+s2+s3)
  Z1 = view(Z,1:s1)
  Z2 = view(Z,s1+1:s1+s2)
  Z3 = view(Z,s1+s2+1:s1+s2+s3)

	cell = (blockIdx().x - 1) * blockDim().y + threadIdx().y
	while cell <= nCells
		# Scatter
		ids = view(cell_ids.data, cell_ids.ptrs[cell]:cell_ids.ptrs[cell+1]-1)

		loop_idx = tidx
		s = D * SB[1] * SB[2]
		while loop_idx <= s
			r, i = @index_to_tuple(loop_idx, D, SB[1] * SB[2])
			I = dof_map[i]
			j1 = I[1]
			j2 = I[2]
			id = ids[i]
			z1_idx = (tidy - 1) * s + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r

			Z1[z1_idx] = x[max(id, 1)] * (id > 0)
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		# Forward pass
		loop_idx = tidx
		s = D * SQ[1] * SB[2]
		while loop_idx <= s
			r, i1, j2 = @index_to_tuple(loop_idx, D, SQ[1], SB[2])

			z2_idx = (tidy - 1) * s + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z2[z2_idx] = 0.0
			for j1 in 1:SB[1]
				z1_idx = (tidy - 1) * SB[2] * SB[1] * D + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
				Z2[z2_idx] += mats[i1, j1, 1, r] * Z1[z1_idx]
			end
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		loop_idx = tidx
		s = D * SQ[1] * SQ[2]
		while loop_idx <= s
			r, i1, i2 = @index_to_tuple(loop_idx, D, SQ[1], SQ[2])

			z3_idx = (tidy - 1) * s + (i2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z3[z3_idx] = 0.0
			for j2 in 1:SB[2]
				z2_idx = (tidy - 1) * SB[2] * SQ[1] * D + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z3[z3_idx] += mats[i2, j2, 2, r] * Z2[z2_idx]
			end

			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		# Apply weights 
		loop_idx = tidx
		s = D * SQ[1] * SQ[2]
		while loop_idx <= s
			r, i1, i2 = @index_to_tuple(loop_idx, D, SQ[1], SQ[2])
			idx = (i2 - 1) * SQ[1] + i1
			z3_idx = (tidy - 1) * s + (idx - 1) * D + r
			Z3[z3_idx] *= wq[idx]
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		# Backward pass
		loop_idx = tidx
		s = D * SQ[1] * SB[2]
		while loop_idx <= s
			r, i1, j2 = @index_to_tuple(loop_idx, D, SQ[1], SB[2])
			z2_idx = (tidy - 1) * s + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z2[z2_idx] = 0.0
			for i2 in 1:SQ[2]
				z3_idx = (tidy - 1) * SQ[2] * SQ[1] * D + (i2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z2[z2_idx] += mats[i2, j2, 2, r] * Z3[z3_idx]
			end
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		loop_idx = tidx
		s = D * SB[1] * SB[2]
		while loop_idx <= s
			r, j1, j2 = @index_to_tuple(loop_idx, D, SB[1], SB[2])
			z1_idx = (tidy - 1) * s + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
			Z1[z1_idx] = 0.0
			for i1 in 1:SQ[1]
				z2_idx = (tidy - 1) * SB[2] * SQ[1] * D + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z1[z1_idx] += mats[i1, j1, 1, r] * Z2[z2_idx]
			end
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		# Assemble
		loop_idx = tidx
		s = D * SB[1] * SB[2]
		while loop_idx <= s
			r, i = @index_to_tuple(loop_idx, D, SB[1] * SB[2])
			I = dof_map[i]
			j1 = I[1]
			j2 = I[2]
			id = ids[i]
			z1_idx = (tidy - 1) * s + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
			if id > 0
				CUDA.@atomic y[id] += Z1[z1_idx]
			end
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		cell += gridDim().x * blockDim().y
	end

	return
end