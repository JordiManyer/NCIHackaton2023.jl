#!/bin/bash

module load cuda

nsys launch --trace cuda julia --project=. -e'include("test/GPU_v1.jl")'
