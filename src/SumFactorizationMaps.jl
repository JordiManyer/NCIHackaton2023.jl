
struct SumFactorizationMap{D,SB,SQ} <: Arrays.Map
  mats
  dof_map
end

function _get_terms(poly::Polytope,orders)
  _nodes, facenodes = Gridap.ReferenceFEs._compute_nodes(poly,orders)
  terms = Gridap.ReferenceFEs._coords_to_terms(_nodes,orders)
  return terms
end

function _get_dof_map(poly::Polytope,orders) 
  maps1D = map(ord -> map(t -> t[1], _get_terms(SEGMENT,[ord])), orders)
  terms  = _get_terms(poly,orders)
  dofmap = map(t -> CartesianIndex(map((x,m) -> findfirst(mi -> mi == x,m),Tuple(t),maps1D)), terms)
  return dofmap
end

function SumFactorizationMap(D::Int,fe_orders::Tuple,quad_orders::Tuple)
  poly = (D == 2) ? QUAD : HEX

  basis1D  = map(ord -> get_shapefuns(LagrangianRefFE(Float64,SEGMENT,ord)), fe_orders)
  quads1D  = map(ord -> Quadrature(SEGMENT,ord), quad_orders)
  points1D = map(q -> get_coordinates(q), quads1D)

  A0 = map(evaluate,basis1D,points1D)
  B0 = map((f,x)-> map(y->y[1],evaluate(f,x)),map(Broadcasting(gradient),basis1D),points1D)

  evalMats = A0
  gradMats = [(q != r) ? A0[q] : B0[q] for q in 1:D, r in 1:D]

  dof_map = _get_dof_map(poly,fe_orders)

  SB = map(b->length(b),basis1D)
  SQ = map(q->length(get_weights(q)),quads1D)
  return SumFactorizationMap{D,SB,SQ}([evalMats,gradMats],dof_map)
end

function Arrays.return_cache(::SumFactorizationMap{D,SB,SQ},x,wq,jq,djq) where {D,SB,SQ}
  r        = similar(x)
  Z_scalar = [zeros(SQ[1:d-1]...,SB[d:D]...) for d in 1:D+1]
  Z_vector = [zeros(D,SQ[1:d-1]...,SB[d:D]...) for d in 1:D+1]
  return r, Z_scalar, Z_vector
end

function Arrays.return_value(::SumFactorizationMap{D,SB,SQ},x,wq,jq,djq) where {D,SB,SQ}
  return similar(x)
end

function Arrays.evaluate!(cache,m::SumFactorizationMap{D,SB,SQ},x,wq,jq,djq) where {D,SB,SQ}
  y, Z_scalar, Z_vector = cache
  evalMats, gradMats = m.mats
  dof_map = m.dof_map

  _sumfac_array2tensor!(m,Z_vector[1],x,dof_map)    # Copy dofs to tensor
  _sumfac_dof2quad!(m,gradMats,Z_vector)            # Compute the gradient of u
  _sumfac_apply_weights!(m,Z_vector[end],wq,jq,djq) # Apply pullback and quad weights
  _sumfac_quad2dof!(m,gradMats,Z_vector)            # Apply gradient of v
  _sumfac_tensor2array!(m,Z_vector[1],y,dof_map)    # Transfer result back to y
  return y
end
