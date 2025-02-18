# Generates synthetic house price data with features:
# - House size (sq ft)
# - Number of bedrooms
# - Age of house (years)
function generate_data(SAMPLE_SIZE)
    # Create features matrix with features in rows, samples in columns
    X = zeros(3, SAMPLE_SIZE)  
    X[1, :] = rand(1000:5000, SAMPLE_SIZE)  # House size in sq ft
    X[2, :] = rand(1:5, SAMPLE_SIZE)        # Number of bedrooms
    X[3, :] = rand(1:100, SAMPLE_SIZE)      # Age of the house in years
    
    # Define true relationship between features and price
    true_weights = zeros(1,3)  
    true_weights[1,1] = 0.2    # Price increase per sq ft
    true_weights[1,2] = 50     # Price increase per bedroom
    true_weights[1,3] = -2.5   # Price decrease per year of age
    true_bias = 100            # Base price

    # Generate prices using true relationship plus random noise
    Y = true_weights * X .+ true_bias .+ randn(1, SAMPLE_SIZE) * 50
    
    # Normalize features using Z-score normalization
    X_means = mean(X, dims=2)
    X_stds = std(X, dims=2)
    X_norm = (X .- X_means) ./ X_stds
    
    # Normalize labels (prices)
    Y_mean = mean(Y)
    Y_std = std(Y)
    Y_norm = (Y .- Y_mean) ./ Y_std
    
    # Store normalization parameters for later denormalization
    normalization_params = (
        X_means=X_means,
        X_stds=X_stds,
        Y_mean=Y_mean,
        Y_std=Y_std
    )
    
    return X_norm, Y_norm, normalization_params
end

# Converts normalized predictions back to original price scale
function denormalize_predictions(Y_norm, norm_params)
    return Y_norm .* norm_params.Y_std .+ norm_params.Y_mean
end




