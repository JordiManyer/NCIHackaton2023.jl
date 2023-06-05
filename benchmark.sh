#!/bin/bash
#PBS -P vp91
#PBS -q normal
#PBS -l walltime=06:00:00
#PBS -l ncpus=1
#PBS -l mem=16GB
#PBS -N benchmark_sumfac
#PBS -l wd

PROJECT_DIR=/home/552/jm3247/hackaton2023/NCIHackaton2023.jl

julia --project=$PROJECT_DIR -O3 --check-bounds=no $PROJECT_DIR/profiling/benchmark.jl

