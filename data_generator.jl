function generate_data(SAMPLE_SIZE)
    # Create features matrix with features in rows, samples in columns
    X = zeros(3, SAMPLE_SIZE)  # Changed back to (features, samples)
    X[1, :] = rand(1000:5000, SAMPLE_SIZE)  # House size in sq ft
    X[2, :] = rand(1:5, SAMPLE_SIZE)        # Number of bedrooms
    X[3, :] = rand(1:100, SAMPLE_SIZE)      # Age of the house in years
    
    # Generate house prices with some noise
    true_weights = zeros(1,3)  # Changed back to matrix
    true_weights[1,1] = 0.2
    true_weights[1,2] = 50
    true_weights[1,3] = -2.5
    true_bias = 100                 # Base price

    Y = true_weights * X .+ true_bias .+ randn(1, SAMPLE_SIZE) * 50  # Add noise
    
    # Normalize features (Z-score normalization)
    X_means = mean(X, dims=2)
    X_stds = std(X, dims=2)
    X_norm = (X .- X_means) ./ X_stds
    
    # Normalize labels
    Y_mean = mean(Y)
    Y_std = std(Y)
    Y_norm = (Y .- Y_mean) ./ Y_std
    
    # Store normalization parameters for later use
    normalization_params = (
        X_means=X_means,
        X_stds=X_stds,
        Y_mean=Y_mean,
        Y_std=Y_std
    )
    
    return X_norm, Y_norm, normalization_params
end

# Helper function to denormalize predictions
function denormalize_predictions(Y_norm, norm_params)
    return Y_norm .* norm_params.Y_std .+ norm_params.Y_mean
end




