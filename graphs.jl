function plot_data(inputs, labels, SAMPLE_SIZE)
    # Create a figure with 5 rows and 4 columns (input + 3 label components)
    indices = rand(1:SAMPLE_SIZE, 5);
    p = plot(layout=(5,3), size=(1600,1000))

    for (i, idx) in enumerate(indices)
        # Plot input matrix
        heatmap!(p[i,1], inputs[:,:,idx], 
                title="Sample $idx - D = $(round(D[idx], digits=3))",
                ylabel="Matrix",
                colorbar=true)

        
        # Plot eigenvalues (first M values in labels)
        plot!(p[i,2], 1:M, labels[:,1,idx], 
                title="Singular Values",
                ylabel="Value",
                label=false)


        
        # Plot eigenvectors (next M values)
        heatmap!(p[i,3], labels[:,2:M+1,idx], 
                title="Left singular vectors",
                colorbar=true)
        
    end

    # Adjust layout
    plot!(p, plot_title="Sample Data Visualization", 
        plot_titlefontsize=14, 
        margin=5Plots.mm)

    # Save or display the plot
    display(p)
end

# savefig("samples.png")  # uncomment to save
function plot_predictions(labels, predictions, test_indices)
    # Plot original vs predicted results
    for (i, idx) in enumerate(test_indices)
        fig = Figure(size=(1200, 800))
        

        # Original singular values
        ax1 = Axis(fig[1, 1], title="Singular Values")
        lines!(ax1, labels[:,1,idx], label="Original")
        lines!(ax1, predictions[i][:,1], label="Predicted")
        axislegend(ax1)

    
        # Original singular vectors
        ax2 = Axis(fig[1, 2], title="Original Singular Vectors")
        heatmap!(ax2, labels[:,2:M+1,idx])

    
        # Predicted singular vectors
        ax3 = Axis(fig[1, 3], title="Predicted Singular Vectors")
        heatmap!(ax3, predictions[i][:,2:end])
    
        display(fig)
end
end
