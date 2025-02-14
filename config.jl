using ProgressMeter
using LinearAlgebra
using Lux
using LuxCUDA # uncomment this to use CUDA
# using LuxAMDGPU # uncomment this to use AMDGPU
using Random
using Statistics
using Distributions
using Optimisers
using Zygote
using ComponentArrays
using Plots
using ProgressMeter
using CUDA
using Functors
using ADTypes
using MLUtils
using LinearAlgebra

SAMPLE_SIZE = 500;
LEARNING_RATE = 1e-3;
EPOCHS = 500;
BATCH_SIZE = 32;