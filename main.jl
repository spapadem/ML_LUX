# Import required modules
include("config.jl")
include("data_generator.jl")
include("network.jl")
include("graphs.jl")

# Generate synthetic house price dataset
inputs, labels, norm_params = generate_data(SAMPLE_SIZE)

# Train the neural network model
model, parameters, layer_states = train_network(inputs, labels, EPOCHS, BATCH_SIZE, LEARNING_RATE)

# Make prediction for a sample house
# Example house specifications:
# - Size: 3000 sq ft
# - Bedrooms: 3
# - Age: 20 years
new_house = reshape([3000, 3, 20], :, 1)  # Reshape to column vector
# Normalize the input using same parameters as training data
new_house_normalized = (new_house .- norm_params.X_means) ./ norm_params.X_stds

# Get price prediction and denormalize it
predicted_price_norm, _ = model(new_house_normalized, parameters, layer_states)
predicted_price = denormalize_predictions(predicted_price_norm, norm_params)

# Print prediction details
println("\nPrediction for house with:")
println("Size: ", Int(new_house[1]), " sq ft")
println("Bedrooms: ", Int(new_house[2]))
println("Age: ", Int(new_house[3]), " years")
println("Predicted price: ", round(Int, predicted_price[1]))

# Calculate true price using the original pricing formula
true_weights = [0.2 50 -2.5]
true_bias = 100
true_price = (true_weights * new_house .+ true_bias)[1]
println("True price: ", round(Int, true_price))
println("Prediction error: ", round(Int, abs(predicted_price[1] - true_price)), 
        " (", round(abs(predicted_price[1] - true_price)/true_price * 100, digits=1), "%)")

# Print dataset statistics
println("\nData Statistics:")
println("Average house size: ", round(Int, norm_params.X_means[1]), " sq ft")
println("Average bedrooms: ", round(norm_params.X_means[2], digits=1))
println("Average age: ", round(Int, norm_params.X_means[3]), " years")
println("Average price: ", round(Int, norm_params.Y_mean))

