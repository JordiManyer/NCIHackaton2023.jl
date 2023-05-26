
struct LazyMatrix{T} <: AbstractMatrix{T}
  m
  test
  trial
  trian
  assem
  rows
  dir_zeros
  cell_quantities
  function LazyMatrix(m,test,trial,trian,assem,rows,dir_zeros,cell_quantities)
    T = eltype(dir_zeros)
    new{T}(m,test,trial,trian,assem,rows,dir_zeros,cell_quantities)
  end
end

Base.size(A::LazyMatrix) = (num_free_dofs(A.test),num_free_dofs(A.trial))
Base.eltype(::LazyMatrix{T}) where T = T

function Base.show(io::IO,::MIME{Symbol("text/plain")},A::LazyMatrix{T}) where T
  s = Base.size(A)
  print(io,"$(s[1])x$(s[2]) LazyMatrix{$T}\n")
end

function Base.show(io::IO, A::LazyMatrix{T}) where T
  s = Base.size(A)
  print(io,"$(s[1])x$(s[2]) LazyMatrix{$T}")
end

function LazyMatrix(m::SumFactorizationMap,trial::FESpace,test::FESpace,dΩ::Measure)
  return LazyMatrix(m,trial,test,dΩ.quad)
end

function LazyMatrix(m::SumFactorizationMap,trial::FESpace,test::FESpace,quad::CellQuadrature)
  trian    = get_triangulation(quad)
  cell_map = get_cell_map(trian)
  cell_Jt  = lazy_map(∇,cell_map)
  cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
  
  cell_djq = lazy_map(Broadcasting(meas),cell_Jtx)
  cell_jq  = lazy_map(Broadcasting(inv),cell_Jtx)
  cell_wq  = quad.cell_weight
  cell_quantities = (cell_wq,cell_jq,cell_djq)

  assem = SparseMatrixAssembler(trial,test)
  rows = get_cell_dof_ids(test,trian)

  dir_zeros = zero_dirichlet_values(trial)
  return LazyMatrix(m,test,trial,trian,assem,rows,dir_zeros,cell_quantities)
end

function LinearAlgebra.mul!(y::AbstractVector,A::LazyMatrix,x::AbstractVector)
  cell_dof = scatter_free_and_dirichlet_values(A.trial,x,A.dir_zeros)

  cell_vec   = lazy_map(A.m,cell_dof,A.cell_quantities...)
  cell_vec_r = Gridap.FESpaces.attach_constraints_rows(A.test,cell_vec,A.trian)
  vecdata    = ([cell_vec_r],[A.rows])
  assemble_vector!(y,A.assem,vecdata)

  return y
end
