# Core packages for numerical computations and statistics
using ProgressMeter
using LinearAlgebra
using Statistics
using Distributions

# Neural network and optimization packages
using Lux
using LuxCUDA # For CUDA GPU acceleration
# using LuxAMDGPU # Alternative for AMD GPU acceleration
using Optimisers
using Zygote  # Automatic differentiation
using ComponentArrays

# Utility packages
using Random
using Plots
using ProgressMeter
using CUDA
using Functors
using ADTypes
using MLUtils

# Training hyperparameters
SAMPLE_SIZE = 500;      # Number of houses in the dataset
LEARNING_RATE = 1e-3;   # Learning rate for Adam optimizer
EPOCHS = 500;           # Number of training epochs
BATCH_SIZE = 32;        # Number of samples per batch