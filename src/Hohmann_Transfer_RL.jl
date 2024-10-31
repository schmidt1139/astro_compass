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
    timepsan_step::Float64 = 3600.0;
    propagator_settings = Dict();
    propagator_settings["flag_fixed_step"] = true;
    propagator_settings["flag_write_ephemeris_states"] = false;
    propagator_settings["timepsan_step"] = timepsan_step;

    #RL parameters
    RL_settings                                = Dict();
    γ::Float64                                 = 0.9999;
    ϵ_high::Float64                            = 0.9;
    learn_rate::Float64                        = 0.1;
    total_episodes::Int                        = 10;
    episodes_between_training::Int             = 10;
    training_epochs::Int                       = 1;
    training_sample_size::Int                  = 10;
    eval_episodes::Int                         = 1;
    ϵ_floor::Float64                           = 0.8;
    traj_depth::Float64                        = 24;
    range_dv1::Vector{Float64}                 = [ -0.001, 0.001 ];
    range_dv2::Vector{Float64}                 = [-0.001, 0.001 ];
    list_ranges::Vector{Array{Float64}}        = [  range_dv1, range_dv2 ];
    target_sma::Float64                        = 14 * Moon.r;


    RL_settings["γ"]                           = γ;
    RL_settings["ϵ_high"]                      = ϵ_high;
    RL_settings["learn_rate"]                  = learn_rate;
    RL_settings["total_episodes"]              = total_episodes;
    RL_settings["episodes_between_training"]   = episodes_between_training;
    RL_settings["training_epochs"]             = training_epochs;
    RL_settings["training_sample_size"]        = training_sample_size;
    RL_settings["ϵ_floor"]                     = ϵ_floor;
    RL_settings["traj_depth"]                  = traj_depth;
    RL_settings["list_ranges"]                 = list_ranges;
    RL_settings["flag_plot_ephem_episode"]     = false;
    RL_settings["count_plot_final_eps"]        = RL_settings["total_episodes"];
    RL_settings["target_sma"]                  = target_sma;

    ACTOR = Chain(
        Dense(6, 6, tanh),
        Dense(6, 6, tanh),
        Dense(6, 2 )
    )

    CRITIC = Chain(
        Dense(8, 8, tanh),
        Dense(8, 8, tanh),
        Dense(8, 1 )
    )

    display(ACTOR);
    display(CRITIC);

    #spacecraft parameters
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

    #inital action
    #timespan_wait_0::Float64 = 0.4;
    dV_1::Float64 = 0.1;
    #timespan_transfer_0::Float64 = 0.6;
    dV_2::Float64 = 0.0;

    #plotting settings
    count_plot_final_eps = total_episodes - 1;

    #pre-propagation setup
    count_episodes = 1;
    count_total_eval_eps = 0;
    count_episodes_since_training = 0;
    count_evals = 0;
    buffer = [];
    arr_episodes = [];
    arr_r = [];
    arr_steps = [];
    flag_plot_ephem_episode = false;

    while ( count_episodes <= total_episodes )

        if ( count_episodes >= RL_settings["count_plot_final_eps"] )
            RL_settings["flag_plot_ephem_episode"] = true;
        end

        #determine epsilon
        m_ϵ = ( ϵ_floor - ϵ_high ) / ( total_episodes );
        ϵ = m_ϵ * count_episodes + ϵ_high;
        RL_settings["ϵ"] = ϵ;

        #reset the environment
        traj_step_count = 0;
        γ_current = 1.0;
        R = 0.0;
        SC1 = Spacecraft( "SC1", initial_epoch, position_0, velocity_0, total_mass_0, fuel_mass_0, isp_SC, [0.0, 0.0], timepsan_step );
        eph = Ephemeris( [], [], [], [], [], 0 );

        #propagate a trajectory
        R, buffer, eph, total_steps = HT_RL_Trajectory( RL_settings, propagator_settings, SC1, eph, list_celestial_bodies, ACTOR, buffer );

        #println("Ep: " * string(total_steps) ;)

        if ( RL_settings["flag_plot_ephem_episode"] == true )
            list_plots = plot_ephem( eph, Moon.r, true, "Moon" );
            for plot in list_plots
                display(plot);
            end
        end


        #increment counters
        count_episodes                  = count_episodes + 1;
        count_episodes_since_training   = count_episodes_since_training + 1;

        #training
        if (count_episodes_since_training == episodes_between_training )

            println("Episode " * string(count_episodes-1) * "... time to train");
            println(" Buffer size: " * string( length(buffer) ) );
            println(" ϵ: " * string(ϵ) );
            count_episodes_since_training   = 0;

            #pull from experience buffer
            data = rand( buffer, training_sample_size );

            function HT_RL_loss( ACTOR, S, A, S_prime, r, flag_terminal )
                
                l = -r;

                #=
                println("S: " * string(S) );
                println("A: " * string(A) );
                println("S': " * string(S_prime) );
                println("r: " * string(r) );
                println("l: " * string(l) );
                =#

                return l;

            end

            for i_train in 1:training_epochs
                Flux.Optimise.train!( HT_RL_loss, ACTOR, data, Descent(learn_rate) );
            end

            #dump the old buffer
            buffer = [];

            #eval the actor
            count_eval_episode = 1;
            sum_R = 0.0;
            sum_steps = 0;
            RL_settings["ϵ"] = 0.0;


            while ( count_eval_episode <= eval_episodes )

                #reset the environment
                traj_step_count = 0;
                γ_current = 1.0;
                R = 0.0;
                SC1 = Spacecraft( "SC1", initial_epoch, position_0, velocity_0, total_mass_0, fuel_mass_0, isp_SC, [0.0, 0.0], timepsan_step );
                eph = Ephemeris( [], [], [], [], [], 0 );

                S::Vector{Float64} = [ SC1.position[1], SC1.position[2], SC1.velocity[1], SC1.velocity[2], Moon.mu, target_sma  ];
                S_nn = convert( Vector{Float32}, S );
                A_nn = ACTOR(S_nn);
                A = convert( Vector{Float64}, A_nn );

                R, buffer, eph, total_steps = HT_RL_Trajectory( RL_settings, propagator_settings, SC1, eph, list_celestial_bodies, ACTOR, buffer );

                count_eval_episode = count_eval_episode + 1;
                count_total_eval_eps = count_total_eval_eps + 1;

                display(R);

                sum_R = sum_R + R;
                sum_steps = sum_steps + total_steps;

            end

            RL_settings["ϵ"] = ϵ;

            count_evals = count_evals + 1;
            mean_R = sum_R / eval_episodes;
            mean_steps = sum_steps / eval_episodes;

            push!( arr_episodes, count_evals );
            push!( arr_r, mean_R );
            push!( arr_steps, mean_steps );

            #plot traj steps
            p_steps = plot(arr_episodes, arr_steps, color = :red, xlabel="Eval Episodes", ylabel="Steps" );
            display( p_steps )

            #plot traj rewards
            p_r = plot(arr_episodes, arr_r , color = :blue, xlabel="Eval Episodes", ylabel="R" );
            display(p_r);

        end

    end

    #plot traj steps
    p_steps = plot(arr_episodes, arr_steps, color = :red, xlabel="Eval Episodes", ylabel="Steps" );
    display( p_steps )

    #plot traj rewards
    p_r = plot(arr_episodes, arr_r , color = :blue, xlabel="Eval Episodes", ylabel="R" );
    display(p_r);

end

Hohmann_Transfer_RL();