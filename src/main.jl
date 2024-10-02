

include("Celestial_Body.jl");
include("Ephemeris.jl");
include("Spacecraft.jl");
include("Policy.jl");

using Plots;
using LinearAlgebra;
using Base.Math;
using Flux;
using CSV;

mutable struct prop_settings
    sim_duration::Float64;
    time_per_step::Float64;
    max_steps_rollout::Int;
    time_limit_periapsis_step::Float64;
    num_max_steps_between_mans::Int;
end


function main()
    
    println("Main");

    #define celestial objects and Spacecraft
    Moon = Celestial_Body( "Moon", [0.0,0.0], [0.0,0.0], 4903, 1740 );
    Earth = Celestial_Body( "Earth", [384400.0,0.0], [0.0,0.0], 398600/10, 6378 );

    #spacecraft constructor
    SC1 = Spacecraft( "SC1", [0.0, 9000.0], [sqrt(Moon.mu/7000), 0.0], 1000.0, 500.0, 220.0, [0.0, 0.0], 5 );
    eph = Ephemeris( [], [], [], [], [], 0 );

    #simulation parameters
    sim_duration::Float64 = 60*60*24*100;
    elapsed_time::Float64 = 0.0;
    time_per_step::Float64 = 60*60*24*1;
    time_limit_periapsis_step::Float64 = 60*60*24*10;
    max_steps_rollout::Int = 4;
    list_celestial_bodies::Array{Celestial_Body} = [ Moon, Earth ];
    flag_terminal::Bool = false;
    flag_maneuvers_enabled::Bool = true;
    max_rad = -Inf;
    num_steps::Int = 0;
    max_steps::Int = 30;
    num_steps_GA::Int = 0;
    max_steps_GA::Int = 32;
    num_steps_between_mans::Int = 0;
    num_max_steps_between_mans::Int = 4;
    r::Float64 = 0.0;
    γ::Float64 = 0.99;
    α::Float64 = 0.0005; #gradient ascent learning rate
    ϵ::Float64 = 0.005; #gradient ascent restriction
    δ_R_frac::Float64 = 0.05;
    δ_min_dV::Float64 = sqrt( 2 * ϵ ) / 100;
    arr_sc_x_plot::Array{Float64} = [];
    arr_sc_y_plot::Array{Float64} = [];
    maneuver_history::Array{String} = [];
    path_out_dir = "../output";
    R_TRAJ::Float64 = 0.0;

    #propagation settings
    propagation_config = prop_settings( sim_duration, time_per_step, max_steps_rollout, time_limit_periapsis_step, num_max_steps_between_mans );

    #initialize policy parameters
    Θ_params = Policy_Params( [0.0, 0.0], [0.001, 0.001] );

    #=
    #neural network constructor
    Q = Chain( 
        Dense( 4, 4, tanh ),
        Dense( 4, 4, tanh ),
        Dense( 4, 2 )
     )
    =#

    #create state tuple
    #create state tuple at time t
    r_rel_1 = SC1.position - Moon.position;
    r_rel_2 = SC1.position - Earth.position;
    v_rel_1 = SC1.velocity - Moon.velocity;
    v_rel_2 = SC1.velocity - Earth.velocity;
    S = ( r_rel_1, v_rel_1, r_rel_2, v_rel_2, Moon.mu, Earth.mu, SC1.mass, SC1.fuel_mass, flag_terminal );

    #R = ValueRollout( Θ_params, S, γ, propagation_config, SC1, list_celestial_bodies );

    
    while ( elapsed_time < sim_duration && num_steps < max_steps )
        
        #create state tuple at time t
        r_rel_1 = SC1.position - Moon.position;
        r_rel_2 = SC1.position - Earth.position;
        v_rel_1 = SC1.velocity - Moon.velocity;
        v_rel_2 = SC1.velocity - Earth.velocity;
        S = ( r_rel_1, v_rel_1, r_rel_2, v_rel_2, Moon.mu, Earth.mu, SC1.mass, SC1.fuel_mass, flag_terminal );

        #perform policy search w/ gradient ascent
            #start with fresh dV
        Θ_params = Policy_Params( [0.0, 0.0], [0.001, 0.001] );


        if ( flag_maneuvers_enabled == true )
        
            num_steps_GA = 0;
            ϵ_input::Float64 = ϵ;
            ΔΘ = ones( length(Θ_params.dV) )

            while (num_steps_GA < max_steps_GA && δ_min_dV < norm(ΔΘ) );


                Θ_params, ϵ_input  = policy_gradient_ascent( Θ_params, SC1, list_celestial_bodies, propagation_config, S, γ, α, ϵ_input, δ_R_frac );
                num_steps_GA = num_steps_GA + 1;
                println("      Gradient ascent dV change: " * string( norm(ΔΘ) ) * "   Desired exit condition: " * string(δ_min_dV) );
                
            end

        end
        
        

        #action vector is applied dV in radial and tangential directions
        A_in = Policy( S, Θ_params );

        #SC1, eph, elapsed_time, flag_terminal, r = StepForward( SC1, list_celestial_bodies, elapsed_time, time_per_step, eph, A, true );

        num_steps_between_mans = 0;

        while ( num_steps_between_mans < num_max_steps_between_mans && flag_terminal == false )

            #only maneuver on 0
            if ( num_steps_between_mans != 0 )
                A = zeros( length(A_in) );
            else
                A = A_in;
            end

            SC1, eph, elapsed_time, flag_terminal, r = StepToPeriapsis( SC1, list_celestial_bodies, elapsed_time, time_limit_periapsis_step, eph, A, true, maneuver_history );
            num_steps_between_mans = num_steps_between_mans + 1;
            num_steps = num_steps + 1;
            R_TRAJ = R_TRAJ + r*γ^(num_steps);
            println("Step: " * string( num_steps ) * "   t: " * string(round((elapsed_time/86400)*100)/100) * "   r: " * string(r) * "   dv: " * string(A) * "   fuel mass [kg]: " * string(SC1.fuel_mass) );
        end

        if ( flag_terminal == true )
            println("Terminal state reached");
            break;
        end

    end







    #plotting
    arr_sc_x_plot = eph.arr_X;
    arr_sc_y_plot = eph.arr_Y;
    p = plot( arr_sc_x_plot/Moon.r, arr_sc_y_plot/Moon.r, label="Spacecraft" );

    arr_a = [];
    arr_e = [];
    arr_w = [];
    arr_theta = [];
    arr_elapsed_t = [];

    println("\n\n\n\n\n\n");
    println("Propagation Complete");
    println("R Trajectory: " * string( R_TRAJ ) );
    println("Number of vectors in ephemeris: " * string( eph.num_vectors ) );

    for i in 1:eph.num_vectors

        push!( arr_elapsed_t, eph.arr_elapsed_time[i] );

        SC1.position[1] = eph.arr_X[i];
        SC1.position[2] = eph.arr_Y[i];
        SC1.velocity[1] = eph.arr_VX[i];
        SC1.velocity[2] = eph.arr_VY[i];

        a, e, ω, θ = Calculate_Planar_OE( SC1, Moon );

        push!( arr_a, a );
        push!( arr_e, e );
        push!( arr_w, ω );
        push!( arr_theta, θ );

    end

    arr_moon_rad_x = [];
    arr_moon_rad_y = [];

    for i in 1:360

        x = Moon.r * cosd( i );
        y = Moon.r * sind( i );

        push!( arr_moon_rad_x, x );
        push!( arr_moon_rad_y, y );

    end

    max_rad = maximum( abs.(arr_sc_x_plot) );
    if ( maximum( abs.(arr_sc_y_plot) ) > max_rad ) max_rad = maximum( abs.(arr_sc_y_plot) ) end

    plot!( p, arr_moon_rad_x/Moon.r, arr_moon_rad_y/Moon.r, ratio=:equal, label="Moon" );
    xlims!(p, -max_rad*1.2/Moon.r, max_rad*1.2/Moon.r);
    ylims!(p, -max_rad*1.2/Moon.r, max_rad*1.2/Moon.r);
    xlabel!(p, "Lunar Centered Inertial X (Moon Radii)");
    ylabel!(p, "Lunar Centered Inertial Y (Moon Radii)");

    display(p);
    savefig( path_out_dir * "/orbit_plot.png");

    p2 = plot( arr_elapsed_t/86400, arr_a, linewidth=2 );
    xlabel!(p2, "Elapsed Days");
    ylabel!(p2, "Lunar Orbit Semi-Major Axis (km)");
    display(p2);
    savefig( path_out_dir * "/sma_plot.png");

    p3 = plot( arr_elapsed_t/86400, arr_e, linewidth=2 );
    xlabel!(p3, "Elapsed Days");
    ylabel!(p3, "Lunar Orbit Eccentricity");
    display(p3);
    savefig( path_out_dir * "/ecc_plot.png");

    p4 = plot( arr_elapsed_t/86400, arr_w*180/π, linewidth=2 );
    xlabel!(p4, "Elapsed Days");
    ylabel!(p4, "Argument of Periapsis (deg)");
    display(p4);
    savefig( path_out_dir * "/omega_plot.png");

    p5 = plot( arr_elapsed_t/86400, arr_theta*180/π, linewidth=2 );
    xlabel!(p5, "Elapsed Days");
    ylabel!(p5, "True Anomaly (deg)");
    display(p5);
    savefig( path_out_dir * "/ta_plot.png");

    #write maneuver summary
    f_path = path_out_dir * "/maneuver_summary.csv";
    open( f_path, "w" ) do file
        for str in maneuver_history
            write(file, str * "\n" );
        end
    end

    

    #CSV.write( path_out_dir * "/maneuver_summary.csv",  ["test"] )

    #println("Press any key to exit");
    #input = readline();



end

main();

