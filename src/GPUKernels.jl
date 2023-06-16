
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
  Z  = @cuDynamicSharedMem(Float64,max(s1,s3)+s2)
  Z1 = view(Z,1:s1)
  Z2 = view(Z,max(s1,s3)+1:s1+s2)
  Z3 = view(Z,1:s3)

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

			Z1[z1_idx] = x[abs(id)] * (id > 0)
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
			@inbounds for j1 in 1:SB[1]
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
			@inbounds for j2 in 1:SB[2]
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
			@inbounds for i2 in 1:SQ[2]
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
			@inbounds for i1 in 1:SQ[1]
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
			CUDA.@atomic y[abs(id)] += Z1[z1_idx] * (id > 0)
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		cell += gridDim().x * blockDim().y
	end

	return
end



"""
  SUMFAC-GPU Kernel v3
  Fourth version. Reordering of the matrices to obtain better memory access patterns.
	Matrices are still in ConstantMemory.
	 - We process blockDim().y cells at the same time.
	 - We use blockDim().x threads to process each cell.
"""
function gpu_mul_v3!(m::SumFactorizationMap{D, SB, SQ}, nCells, y, x, cell_ids, wq, ij_mats, ji_mats,dof_map) where {D, SB, SQ}
	dof_map = m.dof_map
	CUDA.Const(ij_mats)
  CUDA.Const(ji_mats)
  CUDA.Const(dof_map)
  tidy = threadIdx().y
	tidx = threadIdx().x
	tidx_step  = blockDim().x
	
  s1 = blockDim().y*D*SB[1]*SB[2]; s2 = blockDim().y*D*SQ[1]*SB[2]; s3 = blockDim().y*D*SQ[1]*SQ[2];
  Z  = @cuDynamicSharedMem(Float64,max(s1,s3)+s2)
  Z1 = view(Z,1:s1)
  Z2 = view(Z,max(s1,s3)+1:s1+s2)
  Z3 = view(Z,1:s3)

	cell = (blockIdx().x - 1) * blockDim().y + threadIdx().y
	while cell <= nCells
		# Scatter
		ids = view(cell_ids.data, cell_ids.ptrs[cell]:cell_ids.ptrs[cell+1]-1)

		loop_idx = tidx
		s = D * SB[1] * SB[2]
		while loop_idx <= s
			r, i = @index_to_tuple(loop_idx, D, SB[1] * SB[2])
			I = dof_map[i]; j1 = Int32(I[1]); j2 = Int32(I[2])
			id = ids[i]
			z1_idx = (tidy - 1) * s + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
			Z1[z1_idx] = x[abs(id)] * (id > 0)
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
			@inbounds for j1 in 1:SB[1]
				z1_idx = (tidy - 1) * SB[2] * SB[1] * D + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
				Z2[z2_idx] += ji_mats[j1,r,i1,1] * Z1[z1_idx]
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
			@inbounds for j2 in 1:SB[2]
				z2_idx = (tidy - 1) * SB[2] * SQ[1] * D + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z3[z3_idx] += ji_mats[j2,r,i2,2] * Z2[z2_idx]
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
			@inbounds for i2 in 1:SQ[2]
				z3_idx = (tidy - 1) * SQ[2] * SQ[1] * D + (i2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z2[z2_idx] += ij_mats[i2,r,j2,2] * Z3[z3_idx]
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
			@inbounds for i1 in 1:SQ[1]
				z2_idx = (tidy - 1) * SB[2] * SQ[1] * D + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z1[z1_idx] += ij_mats[i1,r,j1,1] * Z2[z2_idx]
			end
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		# Assemble
		loop_idx = tidx
		s = D * SB[1] * SB[2]
		while loop_idx <= s
			r, i = @index_to_tuple(loop_idx, D, SB[1] * SB[2])
			I = dof_map[i]; j1 = Int32(I[1]); j2 = Int32(I[2])
			id = ids[i]
			z1_idx = (tidy - 1) * s + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
			CUDA.@atomic y[abs(id)] += Z1[z1_idx] * (id > 0)
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		cell += gridDim().x * blockDim().y
	end

	return
end


"""
  SUMFAC-GPU Kernel v4
  Fifth version. Using MVectors instead of shared memory for the work vectors. 
	 - We process blockDim().y cells at the same time.
	 - We use blockDim().x threads to process each cell.
"""
function gpu_mul_v4!(m::SumFactorizationMap{D, SB, SQ}, nCells, y, x, cell_ids, wq, ij_mats, ji_mats,::Val{blockdim}) where {D, SB, SQ, blockdim}
	dof_map = m.dof_map
	CUDA.Const(ij_mats)
  CUDA.Const(ji_mats)
  CUDA.Const(dof_map)
  tidy = threadIdx().y
	tidx = threadIdx().x
	tidx_step  = blockdim[1]
	
  s1 = D*SB[1]*SB[2]
	s2 = D*SQ[1]*SB[2]
	s3 = D*SQ[1]*SQ[2]
  Z1 = zero(MVector{max(s1,s3),Float64})
  Z2 = zero(MVector{s2,Float64})
  Z3 = Z1

	cell = (blockIdx().x - 1) * blockdim[2] + threadIdx().y
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
			z1_idx = (j2 - 1) * SB[1] * D + (j1 - 1) * D + r

			Z1[z1_idx] = x[max(id, 1)] * (id > 0)
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		# Forward pass
		loop_idx = tidx
		s = D * SQ[1] * SB[2]
		while loop_idx <= s
			r, i1, j2 = @index_to_tuple(loop_idx, D, SQ[1], SB[2])

			z2_idx = (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z2[z2_idx] = 0.0
			@inbounds for j1 in 1:SB[1]
				z1_idx = (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
				Z2[z2_idx] += ji_mats[j1,r,i1,1] * Z1[z1_idx]
			end
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		loop_idx = tidx
		s = D * SQ[1] * SQ[2]
		while loop_idx <= s
			r, i1, i2 = @index_to_tuple(loop_idx, D, SQ[1], SQ[2])

			z3_idx = (i2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z3[z3_idx] = 0.0
			@inbounds for j2 in 1:SB[2]
				z2_idx = (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z3[z3_idx] += ji_mats[j2,r,i2,2] * Z2[z2_idx]
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
			z3_idx = (idx - 1) * D + r
			Z3[z3_idx] *= wq[idx]
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		# Backward pass
		loop_idx = tidx
		s = D * SQ[1] * SB[2]
		while loop_idx <= s
			r, i1, j2 = @index_to_tuple(loop_idx, D, SQ[1], SB[2])
			z2_idx = (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z2[z2_idx] = 0.0
			@inbounds for i2 in 1:SQ[2]
				z3_idx = (i2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z2[z2_idx] += ij_mats[i2,r,j2,2] * Z3[z3_idx]
			end
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		loop_idx = tidx
		s = D * SB[1] * SB[2]
		while loop_idx <= s
			r, j1, j2 = @index_to_tuple(loop_idx, D, SB[1], SB[2])
			z1_idx = (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
			Z1[z1_idx] = 0.0
			@inbounds for i1 in 1:SQ[1]
				z2_idx = (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z1[z1_idx] += ij_mats[i1,r,j1,1] * Z2[z2_idx]
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
			z1_idx = (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
			if id > 0
				CUDA.@atomic y[id] += Z1[z1_idx]
			end
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		cell += gridDim().x * blockdim[2]
	end

	return
end


"""
  SUMFAC-GPU Kernel v5
  Try to reorder some of the indexes to obtain better accesses to SharedMemory. 
	Matrices still in ConstantMemory. 
	 - We process blockDim().y cells at the same time.
	 - We use blockDim().x threads to process each cell.
"""
function gpu_mul_v5!(m::SumFactorizationMap{D, SB, SQ}, nCells, y, x, cell_ids, wq, ij_mats, ji_mats, dof_map) where {D, SB, SQ}
	CUDA.Const(ij_mats)
  CUDA.Const(ji_mats)
  CUDA.Const(dof_map)
  tidy = threadIdx().y
	tidx = threadIdx().x
	tidx_step  = blockDim().x
	
  s1 = blockDim().y*D*SB[1]*SB[2]; s2 = blockDim().y*D*SQ[1]*SB[2]; s3 = blockDim().y*D*SQ[1]*SQ[2];
  Z  = @cuDynamicSharedMem(Float64,max(s1,s3)+s2)
  Z1 = view(Z,1:s1)
  Z2 = view(Z,max(s1,s3)+1:s1+s2)
  Z3 = view(Z,1:s3)

	cell = (blockIdx().x - 1) * blockDim().y + threadIdx().y
	while cell <= nCells
		# Scatter
		ids = view(cell_ids.data, cell_ids.ptrs[cell]:cell_ids.ptrs[cell+1]-1)

		loop_idx = tidx
		s = SB[1] * SB[2]
		while loop_idx <= s
			i   = dof_map[loop_idx]
			id  = ids[i]
			val = x[abs(id)] * (id > 0)
			@inbounds for r in 1:D
				z1_idx = (tidy - 1) * s * D + (loop_idx-1)*D + r
				Z1[z1_idx] = val
			end
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		# Forward pass
		loop_idx = tidx
		s = D * SQ[1] * SB[2]
		while loop_idx <= s
			r, i1, j2 = @index_to_tuple(loop_idx, D, SQ[1], SB[2])
			val = 0.0
			@inbounds for j1 in 1:SB[1]
				z1_idx = (tidy - 1) * SB[2] * SB[1] * D + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
				val += ji_mats[j1,r,i1,1] * Z1[z1_idx]
			end
			z2_idx = (tidy - 1) * s + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z2[z2_idx] = val
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		loop_idx = tidx
		s = D * SQ[1] * SQ[2]
		while loop_idx <= s
			r, i1, i2 = @index_to_tuple(loop_idx, D, SQ[1], SQ[2])
			val = 0.0
			@inbounds for j2 in 1:SB[2]
				z2_idx = (tidy - 1) * SB[2] * SQ[1] * D + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				val += ji_mats[j2,r,i2,2] * Z2[z2_idx]
			end
			z3_idx = (tidy - 1) * s + (i2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z3[z3_idx] = val
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
			val = 0.0
			@inbounds for i2 in 1:SQ[2]
				z3_idx = (tidy - 1) * SQ[2] * SQ[1] * D + (i2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				val += ij_mats[i2,r,j2,2] * Z3[z3_idx]
			end
			z2_idx = (tidy - 1) * s + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
			Z2[z2_idx] = val
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		loop_idx = tidx
		s = D * SB[1] * SB[2]
		while loop_idx <= s
			r, j1, j2 = @index_to_tuple(loop_idx, D, SB[1], SB[2])
			val = 0.0
			@inbounds for i1 in 1:SQ[1]
				z2_idx = (tidy - 1) * SB[2] * SQ[1] * D + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r
				val += ij_mats[i1,r,j1,1] * Z2[z2_idx]
			end
			z1_idx = (tidy - 1) * s + (j2 - 1) * SB[1] * D + (j1 - 1) * D + r
			Z1[z1_idx] = val
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		# Assemble
		loop_idx = tidx
		s = SB[1] * SB[2]
		while loop_idx <= s
			i  = dof_map[loop_idx]
			id = ids[i]
			val = 0.0
			@inbounds for r in 1:D
				z1_idx = ((tidy - 1) * s + loop_idx-1) * D + r
				val += Z1[z1_idx]
			end
			CUDA.@atomic y[abs(id)] += val * (id > 0)
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		cell += gridDim().x * blockDim().y
	end

	return
end



function gpu_mul_v6!(m::SumFactorizationMap{D, SB, SQ}, nCells, y, x, cell_ids, wq,ij_mats,ji_mats) where {D, SB, SQ}
	dof_map = m.dof_map
  #CUDA.Const(mats)
  #CUDA.Const(dof_map)
  tidy = threadIdx().y
	tidx = threadIdx().x
	tidx_step  = blockDim().x
	
  s13 = blockDim().y*D*max(SB[1]*SB[2],SQ[1]*SQ[2])
	s2 = blockDim().y*D*SQ[1]*SB[2]
	smats = D*D*SB[1]*SQ[1]

  Z  = @cuDynamicSharedMem(Float64,s13+s2+2*smats)
  Z1 = view(Z,1:s13)
  Z2 = view(Z,s13+1:s13+s2)
  Z3 = view(Z,1:s13)

	# Bring matrices to shared memory
	f_mats = view(Z,s13+s2+1:s13+s2+smats)
	b_mats = view(Z,s13+s2+smats+1:s13+s2+2*smats)
	loop_idx = (tidy-1) * tidx_step + tidx
	while loop_idx <= smats
		f_mats[loop_idx] = ji_mats[loop_idx]
		b_mats[loop_idx] = ij_mats[loop_idx]
		loop_idx += tidx_step*blockDim().y
	end
	CUDA.sync_threads()

	cell = (blockIdx().x - 1) * blockDim().y + threadIdx().y
	while cell <= nCells
		# Scatter
		ids = view(cell_ids.data, cell_ids.ptrs[cell]:cell_ids.ptrs[cell+1]-1)

		loop_idx = tidx
		s = D * SB[1] * SB[2]
		while loop_idx <= s
			r, i = @index_to_tuple(loop_idx, D, SB[1] * SB[2])
			I = dof_map[i]; j1 = I[1]; j2 = I[2]
			id = ids[i]
			z1_idx = (tidy - 1) * s + (j1 - 1) * SB[2] * D + (j2 - 1) * D + r

			Z1[z1_idx] = x[abs(id)] * (id > 0)
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		# Forward pass
		loop_idx = tidx
		s = D * SQ[1] * SB[2]
		while loop_idx <= s
			r, j2, i1 = @index_to_tuple(loop_idx, D, SB[2], SQ[1])

			z2_idx = (tidy - 1) * s + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r # Not coalesced
			Z2[z2_idx] = 0.0
			@inbounds for j1 in 1:SB[1]
				z1_idx = (tidy - 1) * SB[1] * SB[2] * D + (j1 - 1) * SB[2] * D + (j2 - 1) * D + r # Coalesced
				#fm_idx = (j1 - 1) * SQ[1] * D + (i1 - 1) * D + r
				bm_idx = (i1 - 1) * SB[1] * D + (j1 - 1) * D + r
				Z2[z2_idx] += f_mats[bm_idx] * Z1[z1_idx]
			end
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		loop_idx = tidx
		s = D * SQ[1] * SQ[2]
		while loop_idx <= s
			r, i1, i2 = @index_to_tuple(loop_idx, D, SQ[1], SQ[2])

			z3_idx = (tidy - 1) * s + (i2 - 1) * SQ[1] * D + (i1 - 1) * D + r # Coalesced
			Z3[z3_idx] = 0.0
			@inbounds for j2 in 1:SB[2]
				z2_idx = (tidy - 1) * SB[2] * SQ[1] * D + (j2 - 1) * SQ[1] * D + (i1 - 1) * D + r # Coalesced
				#fm_idx = SB[2]*SQ[2]*D + (j2 - 1) * SQ[2] * D + (i2 - 1) * D + r
				bm_idx = SQ[2]*SB[2]*D + (i2 - 1) * SB[2] * D + (j2 - 1) * D + r
				Z3[z3_idx] += f_mats[bm_idx] * Z2[z2_idx]
			end
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		# Apply weights 
		loop_idx = tidx
		s = D * SQ[1] * SQ[2]
		while loop_idx <= s
			r, idx = @index_to_tuple(loop_idx, D, SQ[1]*SQ[2])
			z3_idx = (tidy - 1) * s + loop_idx
			Z3[z3_idx] *= wq[idx]
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		# Backward pass
		loop_idx = tidx
		s = D * SQ[1] * SB[2]
		while loop_idx <= s
			r, i1, j2 = @index_to_tuple(loop_idx, D, SQ[1], SB[2])
			z2_idx = (tidy - 1) * s + (i1 - 1) * SB[2] * D + (j2 - 1) * D + r # Not coalesced
			Z2[z2_idx] = 0.0
			@inbounds for i2 in 1:SQ[2]
				z3_idx = (tidy - 1) * SQ[2] * SQ[1] * D + (i2 - 1) * SQ[1] * D + (i1 - 1) * D + r # Coalesced
				#bm_idx = SQ[2]*SB[2]*D + (i2 - 1) * SB[2] * D + (j2 - 1) * D + r
				fm_idx = SB[2]*SQ[2]*D + (j2 - 1) * SQ[2] * D + (i2 - 1) * D + r
				Z2[z2_idx] += b_mats[fm_idx] * Z3[z3_idx]
			end
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		loop_idx = tidx
		s = D * SB[1] * SB[2]
		while loop_idx <= s
			r, j2, j1 = @index_to_tuple(loop_idx, D, SB[2], SB[1])
			z1_idx = (tidy - 1) * s + loop_idx
			Z1[z1_idx] = 0.0
			@inbounds for i1 in 1:SQ[1]
				z2_idx = (tidy - 1) * SB[2] * SQ[1] * D + (i1 - 1) * SB[2] * D + (j2 - 1) * D + r # Coalesced
				#bm_idx = (i1 - 1) * SB[1] * D + (j1 - 1) * D + r
				fm_idx = (j1 - 1) * SQ[1] * D + (i1 - 1) * D + r
				Z1[z1_idx] += b_mats[fm_idx] * Z2[z2_idx]
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
			z1_idx = (tidy - 1) * s + (j1 - 1) * SB[2] * D + (j2 - 1) * D + r
			CUDA.@atomic y[abs(id)] += Z1[z1_idx] * (id > 0)
			loop_idx += tidx_step
		end
		CUDA.sync_threads()

		cell += gridDim().x * blockDim().y
	end

	return
end


@generated function gpu_mul_v7!(m::SumFactorizationMap{D, SB, SQ}, nCells, y, x, cell_ids, wq,ij_mats,ji_mats,_dof_map,::Val{blockdims}) where {D, SB, SQ,blockdims}
	s13   = blockdims[2]*D*max(SB[1]*SB[2],SQ[1]*SQ[2])
	s2    = blockdims[2]*D*SQ[1]*SB[2]
	smats = D*D*SB[1]*SQ[1]
	smaps = prod(SB)
	
	return quote 
		tidy = threadIdx().y
		tidx = threadIdx().x

		Z  = @cuStaticSharedMem(Float64,$(s13+s2+2*smats))
		Z1 = view(Z,1:$(s13))
		Z2 = view(Z,$(s13+1):$(s13+s2))
		Z3 = view(Z,1:$(s13))

		# Bring matrices to shared memory
		f_mats = view(Z,$(s13+s2+1):$(s13+s2+smats))
		b_mats = view(Z,$(s13+s2+smats+1):$(s13+s2+2*smats))
		loop_idx = (tidy-1) * $(blockdims[1]) + tidx
		while loop_idx <= $(smats)
			f_mats[loop_idx] = ij_mats[loop_idx]
			b_mats[loop_idx] = ji_mats[loop_idx]
			loop_idx += $(blockdims[1]*blockdims[2])
		end

		dof_map = @cuStaticSharedMem(Int32,$smaps)
		loop_idx = (tidy-1) * $(blockdims[1]) + tidx
		while loop_idx <= $(smaps)
			dof_map[loop_idx] = _dof_map[loop_idx]
			loop_idx += $(blockdims[1]*blockdims[2])
		end

		CUDA.sync_threads()

		cell = (blockIdx().x - 1) * $(blockdims[2]) + threadIdx().y
		while cell <= nCells
			# Scatter
			ids = view(cell_ids.data, cell_ids.ptrs[cell]:cell_ids.ptrs[cell+1]-1)

			loop_idx = tidx
			while loop_idx <= $(SB[1] * SB[2])
				id = ids[dof_map[loop_idx]]
				xi = x[abs(id)] * (id > 0)
				@inbounds for r in 1:$D
					z1_idx = (tidy - 1) * $(D * SB[1] * SB[2]) + (loop_idx - 1) * $D + r
					Z1[z1_idx] = xi
				end
				loop_idx += blockdims[1]
			end
			CUDA.sync_threads()

			# Forward pass
			loop_idx = tidx
			while loop_idx <= $(D * SQ[1] * SB[2])
				r, j2, i1 = @index_to_tuple(loop_idx, D, SB[2], SQ[1])

				val = 0.0
				@inbounds for j1 in 1:$(SB[1])
					z1_idx = (tidy - 1) * $(SB[1] * SB[2] * D) + (j1 - 1) * $(SB[2] * D) + (j2 - 1) * $D + r # Coalesced
					fm_idx = (j1 - 1) * SQ[1] * D + (i1 - 1) * D + r
					#bm_idx = (i1 - 1) * $(SB[1] * D) + (j1 - 1) * $D + r
					val += f_mats[fm_idx] * Z1[z1_idx]
				end
				z2_idx = (tidy - 1) * $(D * SQ[1] * SB[2]) + (j2 - 1) * $(SQ[1] * D) + (i1 - 1) * $D + r # Not coalesced
				Z2[z2_idx] = val
				loop_idx += blockdims[1]
			end
			CUDA.sync_threads()

			loop_idx = tidx
			while loop_idx <= $(D * SQ[1] * SQ[2])
				r, i1, i2 = @index_to_tuple(loop_idx, $D, $(SQ[1]), $(SQ[2]))

				val = 0.0
				@inbounds for j2 in 1:$(SB[2])
					z2_idx = (tidy - 1) * $(SB[2] * SQ[1] * D) + (j2 - 1) * $(SQ[1] * D) + (i1 - 1) * $D + r # Coalesced
					fm_idx = SB[2]*SQ[2]*D + (j2 - 1) * SQ[2] * D + (i2 - 1) * D + r
					#bm_idx = $(SQ[2]*SB[2]*D) + (i2 - 1) * $(SB[2] * D) + (j2 - 1) * $D + r
					val += f_mats[fm_idx] * Z2[z2_idx]
				end
				z3_idx = (tidy - 1) * $(D * SQ[1] * SQ[2]) + (i2 - 1) * $(SQ[1] * D) + (i1 - 1) * $D + r # Coalesced
				Z3[z3_idx] = val
				loop_idx += blockdims[1]
			end
			CUDA.sync_threads()

			# Apply weights 
			loop_idx = tidx
			while loop_idx <= $(D * SQ[1] * SQ[2])
				r, idx = @index_to_tuple(loop_idx, $D, $(SQ[1]*SQ[2]))
				z3_idx = (tidy - 1) * $(D * SQ[1] * SQ[2]) + loop_idx
				Z3[z3_idx] *= wq[idx]
				loop_idx += blockdims[1]
			end
			CUDA.sync_threads()

			# Backward pass
			loop_idx = tidx
			while loop_idx <= $(D * SQ[1] * SB[2])
				r, i1, j2 = @index_to_tuple(loop_idx, $D, $(SQ[1]), $(SB[2]))
				
				val = 0.0
				@inbounds for i2 in 1:$(SQ[2])
					z3_idx = (tidy - 1) * $(SQ[2] * SQ[1] * D) + (i2 - 1) * $(SQ[1] * D) + (i1 - 1) * $D + r # Coalesced
					bm_idx = SQ[2]*SB[2]*D + (i2 - 1) * SB[2] * D + (j2 - 1) * D + r
					#fm_idx = $(SB[2]*SQ[2]*D) + (j2 - 1) * $(SQ[2] * D) + (i2 - 1) * $D + r
					val += b_mats[bm_idx] * Z3[z3_idx]
				end
				z2_idx = (tidy - 1) * $(D * SQ[1] * SB[2]) + (i1 - 1) * $(SB[2] * D) + (j2 - 1) * $D + r # Not coalesced
				Z2[z2_idx] = val
				loop_idx += blockdims[1]
			end
			CUDA.sync_threads()

			loop_idx = tidx
			while loop_idx <= $(D * SB[1] * SB[2])
				r, j2, j1 = @index_to_tuple(loop_idx, $D, $(SB[2]), $(SB[1]))
				
				val = 0.0
				@inbounds for i1 in 1:$(SQ[1])
					z2_idx = (tidy - 1) * $(SB[2] * SQ[1] * D) + (i1 - 1) * $(SB[2] * D) + (j2 - 1) * $D + r # Coalesced
					bm_idx = (i1 - 1) * SB[1] * D + (j1 - 1) * D + r
					#fm_idx = (j1 - 1) * $(SQ[1] * D) + (i1 - 1) * $D + r
					val += b_mats[bm_idx] * Z2[z2_idx]
				end
				z1_idx = (tidy - 1) * $(D * SB[1] * SB[2]) + loop_idx
				Z1[z1_idx] = val
				loop_idx += blockdims[1]
			end
			CUDA.sync_threads()

			# Assemble
			loop_idx = tidx
			while loop_idx <= $(SB[1] * SB[2])
				id = ids[dof_map[loop_idx]]
				yi = 0.0
				@inbounds for r in 1:D
					z1_idx = (tidy - 1) * $(D * SB[1] * SB[2]) + (loop_idx - 1) * $D + r
					yi += Z1[z1_idx]
				end
				CUDA.@atomic y[abs(id)] += yi * (id > 0)
				loop_idx += blockdims[1]
			end
			CUDA.sync_threads()

			cell += gridDim().x * blockDim().y
		end

		return
	end
end