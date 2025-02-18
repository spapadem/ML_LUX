# Visualizes input data samples and their corresponding labels
function plot_data(inputs, labels, SAMPLE_SIZE)
    # Create a figure with 5 random samples, 3 plots per sample
    indices = rand(1:SAMPLE_SIZE, 5);
    p = plot(layout=(5,3), size=(1600,1000))

    for (i, idx) in enumerate(indices)
        # Plot 1: Input matrix heatmap
        heatmap!(p[i,1], inputs[:,:,idx], 
                title="Sample $idx - D = $(round(D[idx], digits=3))",
                ylabel="Matrix",
                colorbar=true)
        
        # Plot 2: Singular values line plot
        plot!(p[i,2], 1:M, labels[:,1,idx], 
                title="Singular Values",
                ylabel="Value",
                label=false)
        
        # Plot 3: Singular vectors heatmap
        heatmap!(p[i,3], labels[:,2:M+1,idx], 
                title="Left singular vectors",
                colorbar=true)
    end

    # Configure plot layout and style
    plot!(p, plot_title="Sample Data Visualization", 
        plot_titlefontsize=14, 
        margin=5Plots.mm)

    display(p)
end

# Compares original and predicted values
function plot_predictions(labels, predictions, test_indices)
    for (i, idx) in enumerate(test_indices)
        fig = Figure(size=(1200, 800))
        
        # Plot 1: Compare original vs predicted singular values
        ax1 = Axis(fig[1, 1], title="Singular Values")
        lines!(ax1, labels[:,1,idx], label="Original")
        lines!(ax1, predictions[i][:,1], label="Predicted")
        axislegend(ax1)
    
        # Plot 2: Original singular vectors
        ax2 = Axis(fig[1, 2], title="Original Singular Vectors")
        heatmap!(ax2, labels[:,2:M+1,idx])
    
        # Plot 3: Predicted singular vectors
        ax3 = Axis(fig[1, 3], title="Predicted Singular Vectors")
        heatmap!(ax3, predictions[i][:,2:end])
    
        display(fig)
    end
end
