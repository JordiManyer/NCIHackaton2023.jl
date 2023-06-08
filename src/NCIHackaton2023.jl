module NCIHackaton2023

using FillArrays
using LoopVectorization
using LinearAlgebra

using CUDA
using Adapt

using Gridap
using Gridap.Arrays
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.TensorValues
using Gridap.FESpaces

include("SumFactorizationMaps.jl")
include("SumFactorizationKernels.jl")
include("LazyMatrices.jl")

include("Utils.jl")
include("GPUKernels.jl")

export SumFactorizationMap, get_dimensional_parameters
export LazyMatrix

export to_gpu
export gpu_mul_v0!
export gpu_mul_v1!
export gpu_mul_v2!

export count_manual_flops_poisson_matrix_free

end
