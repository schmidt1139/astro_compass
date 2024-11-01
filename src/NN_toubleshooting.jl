using Plots;
using LinearAlgebra;
using Random;
using Statistics;
using Base.Math;
using Flux;
using CSV;
using StaticArrays;
using Zygote;

function NN_Test();

    println("Neural Network Troubleshooting");

    learn_rate = 0.01;
    optimizer = Descent(learn_rate);
    training_samples = 100;
    plot_mod = 1;
    training_eps = 2000;

    # Construct a Neural Network Model
    σ = tanh # Can replace custom fcn if desired 
    m = Chain(Dense(1=>16,σ), Dense(16=>16,σ), Dense(16=>16,σ), Dense(16=>1))

    for epoch in range(1,training_samples)

        n = 100;                                         #sample size
        x = rand(n);                                    #arguments
        x = sort(x);
        x_nn_plot = sort(x);                            #NN plotting
        x_plot = range(0,1,1000);                       #plot
        truth(x) = (sin(x) + cos(8*x)^4 );              #truth function
        y = truth.(x)                                   #function to approx sample
        y_plot = truth.(x_plot)                         #function to plot

        x = reshape(x,1,n);
        y = reshape(y,1,n);

        #println("\nx");
        #display(x);
        #println("\ny");
        #display(y);

        #println("\ny =_nn");
        y_nn = m( x );
        #display(y_nn);

        #define loss function
        loss_model( m, x ,y ) = mean( abs2.(m( x ) .- y ) );
        l = loss_model( m, x, y )
        #display(l);


        #=
        data = [];

        for i in range(1,n)
            data_tuple = ( [x[i]], [y[i]] );
            push!(data, data_tuple );
        end
        =#

        data = [(x,y)];

        #display(data);

        for e_i in range(1,training_eps)
            Flux.train!( loss_model, m, data, optimizer )
        end

        if ( mod(epoch,plot_mod) == 0 )

            # Plot the data
            println("Training epoch: " * string(epoch) );
            p1 = scatter(vec(x), vec(y), label = "Sample" );
            plot!( x_plot, y_plot, label = "Truth", ylims = [0.0,2.0] );
            plot!( vec(x), vec(m(x)), label = "NN Approx" );
            display(p1);

        end

    end

end

NN_Test();