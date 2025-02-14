# ML_LUX
 An example of machine learning in Julia, with GPU support.
## Overview
This project demonstrates a simple machine learning example in Julia using the Lux neural network framework. It implements a house price prediction model with GPU acceleration support.

## Features
- Neural network implementation using Lux.jl
- GPU acceleration support (CUDA and AMD GPU)
- Synthetic data generation for house prices
- Training and validation split
- Model prediction capabilities
- Data normalization and denormalization

## Project Structure
- `config.jl` - Configuration and package imports
- `data_generator.jl` - Synthetic data generation
- `network.jl` - Neural network model and training logic
- `main.jl` - Main execution script
- `graphs.jl` - Visualization utilities

## Requirements
The following Julia packages are required:
- Lux
- LuxCUDA (for CUDA GPU support)
- LuxAMDGPU (for AMD GPU support)
- Zygote
- Optimisers
- MLUtils
- Statistics
- Random
- CUDA
- Plots

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ML_LUX.git
   cd ML_LUX
   ```
2. Install Julia if you haven't already:
   - Download from https://julialang.org/downloads/
   - Follow installation instructions for your OS

3. Install required packages:
   ```julia
   using Pkg
   Pkg.add([
       "Lux",
       "LuxCUDA",  # For CUDA GPU support
       "LuxAMDGPU", # For AMD GPU support
       "Zygote",
       "Optimisers", 
       "MLUtils",
       "Statistics",
       "Random",
       "CUDA",
       "Plots"
   ])
   ```

## Usage
1. Configure parameters in `config.jl`:
   - `SAMPLE_SIZE`: Number of training examples
   - `LEARNING_RATE`: Learning rate for optimization
   - `EPOCHS`: Number of training epochs
   - `BATCH_SIZE`: Batch size for training

2. Run the model:
   ```julia
   julia main.jl
   ```

3. Example output:
   ```
   Prediction for house with:
   Size: 3000 sq ft
   Bedrooms: 3
   Age: 20 years
   Predicted price: 750000
   True price: 752500
   Prediction error: 2500 (0.3%)
   
   Data Statistics:
   Average house size: 3000 sq ft
   Average bedrooms: 3.0
   Average age: 50 years
   Average price: 700000
   ```

## Model Details
The model uses a simple neural network to predict house prices based on:
- House size (square feet)
- Number of bedrooms
- Age of house (years)

Features are normalized using Z-score normalization before training.

## Performance
The model typically achieves:
- Training loss < 0.1
- Validation loss < 0.1
- Prediction accuracy within 5% of true values


