include("Celestial_Body.jl");
include("Ephemeris.jl");
include("Spacecraft.jl");
include("Policy.jl");

using Plots;
using LinearAlgebra;
using Base.Math;
using Flux;
using CSV;
using DifferentialEquations;

mutable struct prop_settings
    sim_duration::Float64;
    time_per_step::Float64;
    max_steps_rollout::Int;
    time_limit_periapsis_step::Float64;
    num_max_steps_between_mans::Int;
end

function SimpleProp();

    println("Simple Propagation");

    #define celestial objects and Spacecraft
    Moon = Celestial_Body( "Moon", [0.0,0.0], [0.0,0.0], 4903, 1740 );
    Earth = Celestial_Body( "Earth", [384400.0,0.0], [0.0,0.0], 398600, 6378 );
    #list_celestial_bodies::Array{Celestial_Body} = [ Moon, Earth ];
    list_celestial_bodies::Array{Celestial_Body} = [ Moon ];

    #prop parameters
    timespan_propduration::Float64 = 60*60*24;
    timespan_elpased_time::Float64 = 0.0;
    timepsan_step::Float64 = 60.0;
    epoch_0 = 86400.0;

    #spacecraft constructor
    SC1 = Spacecraft( "SC1", [0.0, 7000.0], [sqrt(Moon.mu/7000), 0.0], 1000.0, 500.0, 220.0, [0.0, 0.0], timepsan_step );
    eph = Ephemeris( [], [], [], [], [], 0 );

    #other data
    list_plots = [];

    #ode settings ----------------------------------------------------------------------------------------------------------------

    #ode timespan
    t_span = (0.0, timespan_propduration);

    #parameters for ODE propagator
    params = [ list_celestial_bodies ];

    #initial state vec
    u0 = [ SC1.position[1], SC1.position[2], SC1.velocity[1], SC1.velocity[2] ];

    #ode problem container
    prob = ODEProblem( ode_prop_f!, u0, t_span, params );

    #-----------------------------------------------------------------------------------------------------------------------------
    #Solve ODE equation
    sol = solve( prob, Tsit5(), reltol = 1e-13, abstol = 1e-13 );

    num_vectors = length(sol);

    println("Solution vector length: " * string( num_vectors ) );

    for vec_i in range(1,num_vectors)

        vector = sol[vec_i];

        t_eph = sol.t[vec_i];
        x_eph = vector[1];
        y_eph = vector[2];
        vx_eph = vector[3];
        vy_eph = vector[4];

        eph = add_data( eph, t_eph, x_eph, y_eph, vx_eph, vy_eph );
        
        #line = "t: " * string( sol.t[vec_i] ) * "   x: " * string( x_eph ) * "   y: " * string( y_eph ) * "   vx: " * string( vx_eph ) * "   vy: " * string( vy_eph );
        #println( line );

    end

    list_plots = plot_ephem( eph, Moon.r, true, "Moon" );

    for plot_i in range(1, length( list_plots ) );
        display(list_plots[plot_i]);
    end


end

SimpleProp();