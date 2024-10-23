include("Celestial_Body.jl");
include("Ephemeris.jl");
include("Spacecraft.jl");
include("Spacecraft_Propagation.jl");
include("Policy.jl");
include("Reward.jl");
include("Hohmann_Transfer_RL_Support_Functions.jl");

using Plots;
using LinearAlgebra;
using Base.Math;
using Flux;
using CSV;

function Hohmann_Transfer_RL();

    println("Hohmann Transfer Reinforcement Learning");

    #define celestial objects and Spacecraft
    Moon = Celestial_Body( "Moon", [0.0,0.0], [0.0,0.0], 4903, 1740 );
    list_celestial_bodies::Vector{Celestial_Body} = [ Moon ];

    #propagator settings
    timepsan_step::Float64 = 60.0;
    propagator_settings = Dict();
    propagator_settings["flag_fixed_step"] = true;
    propagator_settings["flag_write_ephemeris_states"] = false;
    timespan_extra_prop_days = 8.0;

    #RL parameters
    γ::Float64                  = 0.99;
    ϵ_high                      = 0.9;
    learn_rate                  = 0.001; #learn_rate
    total_episodes              = 799;
    episodes_between_training   = 40;
    training_epochs             = 4000;
    training_sample_size        = 100;
    ϵ_floor                     = 0.0;

    ACTOR = Chain(
        Dense(6, 6, tanh),
        Dense(6, 6, tanh),
        Dense(6, 4 )
    )

    display(ACTOR);

    #parameters
    isp_SC = 220.0;

    #initial state
    initial_epoch   = 0.0;
    position_0      = [0.0, 7000.0];
    velocity_0      = [sqrt(Moon.mu/7000), 0.0];
    total_mass_0    = 1000.0;
    fuel_mass_0     = 999.0;
    r_mag_0         = norm( position_0 );
    v_mag_0         = norm( velocity_0 );
    v_escape_0      = sqrt( 2 * Moon.mu / r_mag_0 );
    target_sma      = 14 * Moon.r;

    #inital action
    timespan_wait_0::Float64 = 0.4;
    dV_1::Float64 = 0.1;
    timespan_transfer_0::Float64 = 0.6;
    dV_2::Float64 = 0.11;

    #allowable action ranges
    range_timespan_wait::Vector{Float64} = [0.0, 10.0];
    range_timespan_transfer::Vector{Float64} = [0.0, 10.0];
    range_dv1::Vector{Float64} = [ 0.0, 0.5 ];
    range_dv2::Vector{Float64} = [-1.0, 1.0];
    list_ranges::Vector{Array{Float64}} = [ range_timespan_wait, range_dv1, range_timespan_transfer, range_dv2 ];

    #plotting settings
    count_plot_final_eps = total_episodes - 1;

    count_episodes = 1;
    count_episodes_since_training = 0;
    buffer = [];
    arr_episodes = [];
    arr_r = [];

    while ( count_episodes <= total_episodes )

        println("Episode " * string(count_episodes) );

        #reset the environment
    
        #spacecraft constructor
        SC1 = Spacecraft( "SC1", initial_epoch, position_0, velocity_0, total_mass_0, fuel_mass_0, isp_SC, [0.0, 0.0], timepsan_step );
        eph = Ephemeris( [], [], [], [], [], 0 );
        
        #initial state_space
        S::Vector{Float64} = [ SC1.position[1], SC1.position[2], SC1.velocity[1], SC1.velocity[2], Moon.mu, target_sma  ];

        #determine epsilon
        m_ϵ = ( ϵ_floor - ϵ_high ) / ( total_episodes );
        ϵ = m_ϵ * count_episodes + ϵ_high;
        println(" ϵ: " * string(ϵ) );

        #Action (either random or sampled from actor network)
        if ( rand() < ϵ )
            A = HT_RL_rand_action( list_ranges );
            println(" Random action");
            #display(A);
        else
            S_nn = convert( Vector{Float32}, S );
            A_nn = ACTOR(S_nn);
            A = convert( Vector{Float64}, A_nn );
            A = HT_RL_apply_actor_constraints( A, list_ranges );
            println(" Actor action");
            #display(A);
        end
        
        #A = [ timespan_wait_0, dV_1, timespan_transfer_0, dV_2 ];
        #display(A);

        #Hohmann Transfer step function
        propagator_settings["flag_write_ephemeris_states"] = false;
        S_prime, r, flag_terminal = HT_RL_step( S, A, γ, SC1, list_celestial_bodies, propagator_settings );

        #experience tuple
        experience_tuple = ( S, A, S_prime, r, flag_terminal );

        #add experience to buffer
        push!(buffer, experience_tuple);

        #=
        println( "S:  " * string(S) );
        println( "A:  " * string(A) );
        println( "S': " * string(S_prime) );
        println( "r:  " * string(r) );
        println( "" );
        =#

        if ( count_episodes > count_plot_final_eps )
            #Optional Plotting
            #---------------------------------------------------------------------------------------------------------------------------------
            #Hohmann transfer step function
            propagator_settings["flag_write_ephemeris_states"] = true;
            S_prime, r, flag_terminal, eph = HT_RL_step( S, A, γ, SC1, list_celestial_bodies, propagator_settings, eph, timespan_extra_prop_days );


            #plot the ephem
            list_plots = plot_ephem( eph, Moon.r, true, "Moon" );

            for plot_i in range(1, length( list_plots ) );
                display(list_plots[plot_i]);
            end
            #---------------------------------------------------------------------------------------------------------------------------------

        end
        
        push!(arr_episodes, count_episodes);
        push!(arr_r, r);

        #increment counters
        count_episodes                  = count_episodes + 1;
        count_episodes_since_training   = count_episodes_since_training + 1;


        if (count_episodes_since_training == episodes_between_training )

            println("Episode " * string(count_episodes-1) * "... time to train");
            count_episodes_since_training   = 0;

            #pull from experience buffer
            data = rand( buffer, training_sample_size );

            function HT_RL_loss( ACTOR, S, A, S_prime, r, flag_terminal )
                
                l = 1 - r;
                return l;

            end

            for i_train in 1:training_epochs
                Flux.Optimise.train!( HT_RL_loss, ACTOR, data, Descent(learn_rate) );
            end

            #dump the old buffer
            buffer = [];

            #evaluate the actor
            p_r = plot(arr_episodes, arr_r );
            display(p_r);


        end

        #sleep(4);

    end

end

Hohmann_Transfer_RL();