include("Celestial_Body.jl");
include("Ephemeris.jl");
include("Spacecraft.jl");
include("Spacecraft_Propagation.jl");
include("Policy.jl");

using Plots;
using LinearAlgebra;
using Base.Math;
using Flux;
using CSV;


function SimpleProp();

    println("Simple Propagation");

    #define celestial objects and Spacecraft
    Moon = Celestial_Body( "Moon", [0.0,0.0], [0.0,0.0], 4903, 1740 );
    #Earth = Celestial_Body( "Earth", [384400.0,0.0], [0.0,0.0], 398600, 6378 );
    #list_celestial_bodies::Array{Celestial_Body} = [ Moon, Earth ];
    list_celestial_bodies::Array{Celestial_Body} = [ Moon ];

    #prop parameters
    timespan_propduration::Float64 = 60*60*3 + 1;
    timespan_elpased_time::Float64 = 0.0;
    timepsan_step::Float64 = 60.0;
    epoch_0 = 86400.0;

    #spacecraft constructor
    SC1 = Spacecraft( "SC1", [0.0, 7000.0], [sqrt(Moon.mu/7000), 0.0], 1000.0, 500.0, 220.0, [0.0, 0.0], timepsan_step );
    eph = Ephemeris( [], [], [], [], [], 0 );

    #other data
    list_plots = [];

    #ode settings ----------------------------------------------------------------------------------------------------------------

    #step to elapsed_time
    SC1, sol, eph = step_SC_to_elapsed_time( SC1, 0.0, timespan_propduration, list_celestial_bodies, true, true, eph );

    println("Solution vector length: " * string( eph.num_vectors ) );

    list_plots = plot_ephem( eph, Moon.r, true, "Moon" );

    for plot_i in range(1, length( list_plots ) );
        display(list_plots[plot_i]);
    end


end

SimpleProp();