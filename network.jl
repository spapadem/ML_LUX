# Loss function that computes Mean Squared Error between predictions and true values
function loss_function(parameters, layer_states, x, y, model) 
    # Forward pass through the model to get predictions
    y_pred, new_layer_states = model(x, parameters, layer_states)
    
    # Calculate Mean Squared Error loss: 1/2 * mean((predicted - actual)^2)
    loss = 0.5*mean((y_pred - y).^2)
    
    return loss, y_pred, new_layer_states
end

# Creates a simple neural network model with one dense layer
# Input size: 3 features, Output size: 1 (regression)
function create_model()
    return Chain(
        Dense(3 => 1)
    )
end

# Main training function that handles the training loop and validation
function train_network(inputs, labels, epochs, batch_size, learning_rate)
    # Ensure inputs and labels are properly shaped
    @assert size(inputs, 2) == size(labels, 2) "Number of samples must match between inputs and labels"
    
    # Split the data keeping the (features, samples) format
    (x_train, y_train), (x_val, y_val) = splitobs((inputs, labels); at=0.8, shuffle=true)
    
    train_loader = DataLoader((x_train, y_train); batchsize=batch_size, shuffle=true)
    val_loader = DataLoader((x_val, y_val); batchsize=batch_size, shuffle=false)
    
    # Initialize model, parameters, and move to GPU if available
    model = create_model()
    rng = Random.default_rng()
    parameters, layer_states = Lux.setup(rng, model)
    
    dev = gpu_device()
    parameters = parameters |> dev
    layer_states = layer_states |> dev

    # Setup Adam optimizer with specified learning rate
    opt = Optimisers.Adam(learning_rate)
    opt_state = Optimisers.setup(opt, parameters)
    
    # Training loop
    for epoch in 1:epochs
        # Training phase
        total_train_loss = 0.0
        n_train_batches = 0
        for (x, y) in train_loader
            # Move batch to GPU
            x = x |> dev
            y = y |> dev
            
            # Compute loss and gradients
            (loss, y_pred, layer_states), back = pullback(p -> loss_function(p, layer_states, x, y, model), parameters)
            gs = back((one(loss), nothing, nothing))[1]
            
            # Update parameters using optimizer
            opt_state, parameters = Optimisers.update(opt_state, parameters, gs)
            
            total_train_loss += loss
            n_train_batches += 1
        end
        avg_train_loss = total_train_loss / n_train_batches

        # Validation phase
        total_val_loss = 0.0
        n_val_batches = 0
        layer_states_ = Lux.testmode(layer_states)  # Switch to test mode for validation
        for (x, y) in val_loader
            x = x |> dev
            y = y |> dev
            (loss, y_pred, layer_states_) = loss_function(parameters, layer_states, x, y, model)
            total_val_loss += loss
            n_val_batches += 1
        end
        avg_val_loss = total_val_loss / n_val_batches

        # Print progress every 50 epochs
        if epoch % 50 == 0
            println("Epoch [$epoch]: Train Loss: $(round(avg_train_loss, digits=5)) Val Loss: $(round(avg_val_loss, digits=5))")
        end
    end
    
    # Move model back to CPU before returning
    return (model, parameters, layer_states) |> cpu_device()
end


