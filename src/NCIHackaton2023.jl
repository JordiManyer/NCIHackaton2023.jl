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

export SumFactorizationMap
export LazyMatrix

end
