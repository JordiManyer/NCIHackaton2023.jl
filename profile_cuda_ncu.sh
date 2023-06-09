#!/bin/bash
#PBS -P vp91
#PBS -q gpuvolta
#PBS -l walltime=01:00:00
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=10GB
#PBS -l storage=scratch/vp91

module load cuda
module load nvidia-hpc-sdk/22.11

export LD_LIBRARY_PATH=~/bin/julia-1.8.5/lib/julia/:$LD_LIBRARY_PATH

ncu --set full -k regex:gpu_mul --target-processes all -o result_v1 julia --project=. test/GPU_v1.jl
ncu --set full -k regex:gpu_mul --target-processes all -o result_v3 julia --project=. test/GPU_v3.jl

#ncu launch --trace cuda julia --project=. -e'include("test/GPU_v0.jl")'
#ncu launch --trace cuda julia --project=. -e'include("test/GPU_v1.jl")'

