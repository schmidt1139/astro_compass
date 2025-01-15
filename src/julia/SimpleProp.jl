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
    timespan_propduration_1::Float64 = 1*60*60*24;
    timespan_propduration_2::Float64 = 3*60*60*12;
    timespan_propduration_3::Float64 = 0*60*60*12;
    timespan_elpased_time::Float64 = 0.0;
    timepsan_step::Float64 = 60.0;
    initial_epoch::Float64 = 0.0;
    flag_fixed_step::Bool = true;
    flag_write_ephemeris_states::Bool = true;

    #spacecraft constructor
    SC1 = Spacecraft( "SC1", initial_epoch, [0.0, 7000.0], [sqrt(Moon.mu/7000), 0.0], 1000.0, 999.0, 220.0, [0.0, 0.0], timepsan_step );
    eph = Ephemeris( [], [], [], [], [], 0 );

    #maneuver related
    str_mnvr::String = "";
    arr_dV1::Array{Float64} = [ 0.200, 0.0 ];
    arr_dV2::Array{Float64} = [ 0.0, 0.0 ];

    #other data
    list_plots = [];

    #state_space
    S = [ SC1.position[1], SC1.position[2], SC1.velocity[1], SC1.velocity[2], Moon.mu  ];
    A = [ arr_dV1[1], arr_dV1[2], arr_dV2[1], arr_dV2[2], timespan_propduration_2 ];

    #ode settings ----------------------------------------------------------------------------------------------------------------

    #step to elapsed_time
    SC1, sol, eph = step_SC_to_elapsed_time( SC1, initial_epoch, timespan_propduration_1, list_celestial_bodies, flag_fixed_step, 
    flag_write_ephemeris_states, eph );

    #maneuver spacecraft
    SC1, flag_mnvr, str_mnvr = ManeuverSpacecraft( SC1, arr_dV1, Moon, "LVLH" );

    #step to elapsed_time
    SC1, sol, eph = step_SC_to_elapsed_time( SC1, initial_epoch, timespan_propduration_2, list_celestial_bodies, flag_fixed_step, 
    flag_write_ephemeris_states, eph );

    #maneuver spacecraft
    SC1, flag_mnvr, str_mnvr = ManeuverSpacecraft( SC1, arr_dV2, Moon, "LVLH" );

    #step to elapsed_time
    SC1, sol, eph = step_SC_to_elapsed_time( SC1, initial_epoch, timespan_propduration_3, list_celestial_bodies, flag_fixed_step, 
    flag_write_ephemeris_states, eph );

    list_plots = plot_ephem( eph, Moon.r, true, "Moon" );

    for plot_i in range(1, length( list_plots ) );
        display(list_plots[plot_i]);
    end


end

SimpleProp();