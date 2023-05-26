
"""
Applies the sum-factorization kernel, going from DOFs to quadrature points.

* If it is a scalar quantity, applies 
    `Z[d+1][i1...id,jdp1,...jD] = mats[d][id,jd] * Z[d][i1...idm1,jd,...jD]`
  for every direction `d = 1:D`.
* If is a vector quantity, applies 
    `Z[d+1][r,i1...id,jdp1,...jD] = mats[d,r][id,jd] * Z[d][r,i1...idm1,jd,...jD]`
  for every direction `d = 1:D` and component `r = 1:D`.
        
In these formulas `iX` and `jX` iterate over quadrature points and DOFs in direction X 
respectively.
"""
@generated function _sumfac_dof2quad!(::SumFactorizationMap{D,SB,SQ},mats::AbstractVector,Z::AbstractVector) where {D,SB,SQ}
  lines = Vector{Expr}()
  
  # Z[d+1][i1...id,jdp1,...jD] = mats[d][id,jd] * Z[d][i1...idm1,jd,...jD] for d = 1:D
  for d in 1:D
    idx1  = map(x -> Symbol("i$x"),1:(d-1)) # Left indexes for Z[d]
    idx2  = map(x -> Symbol("j$x"),(d+1):D) # Right indexes for Z[d]
    idx3  = [idx1...,Symbol("i$d"),idx2...] # Indexes for Z[d+1]
    sizes = [SQ[1:d]...,SB[d+1:D]...]
    loop_lims   = map((xi,si,spacer) -> "$xi in 1:$si $spacer",idx3,sizes,[fill(", ",D-1)...,""]) # Limits of the for loops

    body = :(0.0)
    id = Symbol("i$d")
    for jd in 1:SB[d]
      body = :( $body + mats[$d][$id,$jd] * Z[$d][$(idx1...),$jd,$(idx2...)] )
    end
    body = :(Z[$(d+1)][$(idx3...)] = @fastmath $body)
    
    line = Meta.parse(string("@turbo for ", loop_lims..., string(body), " end"))
    push!(lines,line)
  end
  
  return Expr(:block, lines... ,:(return Z[$(D+1)]))
end

@generated function _sumfac_dof2quad!(::SumFactorizationMap{D,SB,SQ},mats::AbstractMatrix,Z::AbstractVector) where D where SB where SQ
  lines = Vector{Expr}()
  
  # Z[d+1][r,i1...id,jdp1,...jD] = mats[d,r][id,jd] * Z[d][r,i1...idm1,jd,...jD] for d = 1:D, r in 1:D
  for d in 1:D
    idx1  = map(x -> Symbol("i$x"),1:(d-1)) # Left indexes for Z[d]
    idx2  = map(x -> Symbol("j$x"),(d+1):D) # Right indexes for Z[d]
    idx3  = [idx1...,Symbol("i$d"),idx2...] # Indexes for Z[d+1]
    sizes = [SQ[1:d]...,SB[d+1:D]...]  
    loop_lims   = map((xi,si,spacer) -> "$xi in 1:$si $spacer",idx3,sizes,[fill(", ",D-1)...,""]) # Limits of the for loops
    
    for k in 1:D
      body = :(0.0)
      id = Symbol("i$d")
      for jd in 1:SB[d]
          body = :( $body + mats[$d,$k][$id,$jd] * Z[$d][$k,$(idx1...),$jd,$(idx2...)] )
      end
      body = :(Z[$(d+1)][$k,$(idx3...)] = @fastmath $body)

      line = Meta.parse(string("@turbo for ", loop_lims..., string(body), " end"))
      push!(lines,line)
    end
  end
  
  return Expr(:block, lines... ,:(return Z[$(D+1)]))
end

"""
Applies the sum-factorization kernel, going from quadrature points to DOFs.

* If it is a scalar quantity, applies 
    `Z[d][i1...idm1,jd,...jD] = mats[d][id,jd] * Z[d+1][i1...id,jdp1,...jD]`
  for every direction `d = D:-1:1`.
* If is a vector quantity, applies 
    `Z[d][r,i1...idm1,jd,...jD] = mats[d,r][id,jd] * Z[d+1][r,i1...id,jdp1,...jD]`
  for every direction `d = D:-1:1` and component `r = 1:D`.
        
In these formulas `iX` and `jX` iterate over quadrature points and DOFs in direction X 
respectively.
"""
@generated function _sumfac_quad2dof!(::SumFactorizationMap{D,SB,SQ},mats::AbstractVector,Z::AbstractVector) where {D,SB,SQ}
  lines = Vector{Expr}()
  
  # Z[d][i1...idm1,jd,...jD] = mats[d][id,jd] * Z[d+1][i1...id,jdp1,...jD] for d = D:1
  for d in D:-1:1
    idx1  = map(x -> Symbol("i$x"),1:(d-1)) # Left indexes for Z[d]
    idx2  = map(x -> Symbol("j$x"),(d+1):D) # Right indexes for Z[d]
    idx3  = [idx1...,Symbol("j$d"),idx2...] # Indexes for Z[d-1]
    sizes = [SQ[1:d-1]...,SB[d:D]...]  
    loop_lims   = map((xi,si,spacer) -> "$xi in 1:$si $spacer",idx3,sizes,[fill(", ",D-1)...,""]) # Limits of the for loops
    
    body = :(0.0)
    jd = Symbol("j$d")
    for id in 1:SQ[d]
      body = :( $body + mats[$d][$id,$jd] * Z[$(d+1)][$(idx1...),$id,$(idx2...)] )
    end
    body = :(Z[$d][$(idx3...)] = @fastmath $body)
    
    line = Meta.parse(string("@turbo for ", loop_lims..., string(body), " end"))
    push!(lines,line)
  end
  
  return Expr(:block, lines... ,:(return Z[1]))
end

@generated function _sumfac_quad2dof!(::SumFactorizationMap{D,SB,SQ},mats::AbstractMatrix,Z::AbstractVector) where D where SB where SQ
  lines = Vector{Expr}()
  
  # Z[d][r,i1...idm1,jd,...jD] = mats[d,r][id,jd] * Z[d+1][r,i1...id,jdp1,...jD] for d = D:1, r = 1:D
  for d in D:-1:1
    idx1  = map(x -> Symbol("i$x"),1:(d-1)) # Left indexes for Z[d]
    idx2  = map(x -> Symbol("j$x"),(d+1):D) # Right indexes for Z[d]
    idx3  = [idx1...,Symbol("j$d"),idx2...] # Indexes for Z[d+1]
    sizes = [SQ[1:d-1]...,SB[d:D]...]  
    loop_lims   = map((xi,si,spacer) -> "$xi in 1:$si $spacer",idx3,sizes,[fill(", ",D-1)...,""]) # Limits of the for loops
    
    for k in 1:D
      body = :(0.0)
      jd = Symbol("j$d")
      for id in 1:SQ[d]
          body = :( $body + mats[$d,$k][$id,$jd] * Z[$(d+1)][$k,$(idx1...),$id,$(idx2...)] )
      end
      body = :(Z[$d][$k,$(idx3...)] = @fastmath $body)

      line = Meta.parse(string("@turbo for ", loop_lims..., string(body), " end"))
      push!(lines,line)
    end
  end
  
  return Expr(:block, lines... ,:(return Z[1]))
end

"""
Copies DOFs from array `y` into tensor form using the dofMap. 
"""
@generated function _sumfac_array2tensor!(::SumFactorizationMap{D,SB,SQ},Z::Array{T,DZ},y,dofMap) where {D,SB,SQ,T,DZ}
  isVector = (DZ == 1 + D)
  if isVector
    body = quote 
      @turbo warn_check_args=false for r in 1:D , (k,idx) in enumerate(dofMap)
        Z[r,idx] = y[k]
      end
    end
  else
    body = quote 
      @turbo warn_check_args=false for (k,idx) in enumerate(dofMap)
        Z[idx] = y[k]
      end
    end
  end
  return Expr(:block,body,:(return Z[1]))
end

"""
Copies back the DOFs from tensor form into array `y` using the dofMap. 
"""
@generated function _sumfac_tensor2array!(::SumFactorizationMap{D,SB,SQ},Z::Array{T,DZ},y,dofMap) where {D,SB,SQ,T,DZ}
  isVector = (DZ == 1 + D)
  if isVector 
    line = :(0.0)
    for r in 1:D
        line = :($line + Z[$r,idx])
    end
  else
    line = :(Z[idx])
  end
  body = quote 
    @turbo warn_check_args=false for (k,idx) in enumerate(dofMap)
      y[k] = @fastmath $line
    end
  end
  return Expr(:block,body,:(return y))
end

@generated function _sumfac_apply_weights!(::SumFactorizationMap{D,SB,SQ},Z,wq,jq,djq) where D where SB where SQ
  idx = map(x -> Symbol("i$x"),1:D)
  sizes = SQ[1:D]
  loop_lims   = map((xi,si,spacer) -> "$xi in 1:$si $spacer",idx,sizes,[fill(", ",D-1)...,""])
  
  lines = Vector{Expr}()
  for r in 1:D # Computes the new components
      # c_r = jq[i1,...,id][l,r]*jq[i1,...,id][l,k]*Z[k,i1,...,id]
      c_r = :(0.0)
      for k in 1:D
          j_rk = :(0.0)
          for l in 1:D
              j_rk = :( $j_rk + jq[$(idx...)][$l,$r] * jq[$(idx...)][$l,$k] )
          end
          c_r = :($c_r + $j_rk * Z[$k,$(idx...)])
      end
      z_r = Symbol("z_$r")
      line = :( $z_r = @fastmath $c_r)
      push!(lines,line)
  end
  
  for r in 1:D # Overwrite old components with the new ones.
      z_r = Symbol("z_$r")
      line = :( Z[$r,$(idx...)] = $z_r * wq[$(idx...)] * djq[$(idx...)] )
      push!(lines,line)
  end
  
  strlines = string(quote $(lines...) end)
  body = Meta.parse(string("for ", loop_lims..., strlines..., " end"))
  
  return Expr(:block, body ,:(return Z))
end