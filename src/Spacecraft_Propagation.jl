using DifferentialEquations;
include("Ephemeris.jl");

function ode_prop_f_1B!(du, u, p, t)

    #=
    ode_prop_f_1B! function
    -----------------------------------------------------------------------------------

    Inputs
    -----------------------------------------------------------------------------------
    du:     Array of derivates
    u:      Current state vector array
    p:      Parameters used in ODE function
    t:      Elapsed time

    Outputs
    -----------------------------------------------------------------------------------
    du:     Array of derivates after being updated by function
    t:      Elapsed time after update
    =#

    #current state vector
    x = u[1];
    y = u[2];
    vx = u[3];
    vy = u[4];

    #mu parameter for central body
    μ = p[1];

    #derivative of position components are velocity components
    du[1] = vx;
    du[2] = vy;

    #acceleration terms

    #position magnitude and directions
    r = sqrt( x^2 + y^2 )

    #x and y directions
    x_hat = x / r;
    y_hat = y / r;

    #update x and y velocity derivatives
    du[3] = - μ * x_hat / r^2;
    du[4] = - μ * y_hat / r^2;

    
end

function ode_prop_f!(du, u, p, t)

    #=
    ode_prop_f! function
    -----------------------------------------------------------------------------------

    Inputs
    -----------------------------------------------------------------------------------
    du:     Array of derivates
    u:      Current state vector array
    p:      Parameters used in ODE function
    t:      Elapsed time

    Outputs
    -----------------------------------------------------------------------------------
    du:     Array of derivates after being updated by function
    t:      Elapsed time after update
    =#

    #get list of celestial bodies in parameter list
    list_CBs = p[1];

    #current state vector
    x = u[1];
    y = u[2];
    vx = u[3];
    vy = u[4];

    arr_a_net = [ 0.0, 0.0 ];

    flag_collision = false;


    #go through list of central bodies and add up accelerations
    for cb in list_CBs

        #gravitational parameter
        μ = cb.mu;

        #inertial position of central body
        cb_x = cb.position[1];
        cb_y = cb.position[2];

        #relative position of the spacecraft with respect to the cb
        x_rel = cb_x - x;
        y_rel = cb_y - y;

        #position magnitude
        r_rel = sqrt( x_rel^2 + y_rel^2 );

        if ( r_rel < cb.r )
            flag_collision = true;
        end

        #position direction
        x_hat = x_rel / r_rel;
        y_hat = y_rel / r_rel;

        x_dd_cb = - μ * x_hat / r_rel^2;
        y_dd_cb = - μ * y_hat / r_rel^2;

        arr_a_net[1] = arr_a_net[1] + x_dd_cb;
        arr_a_net[2] = arr_a_net[2] + y_dd_cb;

    end


    if ( flag_collision == true )

        #derivative of position components are velocity components
        du[1] = 0.0;
        du[2] = 0.0;

        #acceleration terms
        du[3] = 0.0;
        du[4] = 0.0;

    else

        #derivative of position components are velocity components
        du[1] = vx;
        du[2] = vy;

        #acceleration terms
        du[3] = - arr_a_net[1];
        du[4] = - arr_a_net[2];

    end

    
end


function step_SC_to_elapsed_time( SC::Spacecraft, t0::Float64, elapsed_time::Float64, list_celestial_bodies::Vector{Celestial_Body}, flag_fixed_step::Bool,
    flag_write_states_to_ephem::Bool, eph = nothing );

    #error handling
    if ( eph === nothing && flag_write_states_to_ephem == true)
        error("An input ephemeris must be provided to write states");
    end

    #ode timespan
    t_span = ( t0, elapsed_time );

    #parameters for ODE propagator
    params = [ list_celestial_bodies ];

    #define empty solution data struct
    sol = [];

    #initial state vec
    u0 = [ SC.position[1], SC.position[2], SC.velocity[1], SC.velocity[2] ];

    #ode problem container
    prob = ODEProblem( ode_prop_f!, u0, t_span, params );

    if ( flag_fixed_step == false )

        #Solve ODE equation
        sol = solve( prob, Tsit5(), reltol = 1e-13, abstol = 1e-13 );

        if ( flag_write_states_to_ephem == true )
            #add ODE data to ephem
            add_ODE_sol_to_ephem!( eph, sol, false );
        end

        num_vectors = length(sol);

        final_vector = sol[num_vectors];

        SC.position = [ final_vector[1], final_vector[2] ];
        SC.velocity = [ final_vector[3], final_vector[4] ];

    else

        et          = 0.0;
        current_t   = t0;
        next_t      = current_t + SC.step_size;

        while ( et < elapsed_time)

            t_span = ( current_t, next_t );

            et = et + SC.step_size;

            u0 = [ SC.position[1], SC.position[2], SC.velocity[1], SC.velocity[2] ];

            #ode problem container
            prob = ODEProblem( ode_prop_f!, u0, t_span, params );

            #Solve ODE equation
            sol = solve( prob, Tsit5(), reltol = 1e-13, abstol = 1e-13 );

            num_vectors = length(sol);

            final_vector = sol[num_vectors];
    
            SC.position = [ final_vector[1], final_vector[2] ];
            SC.velocity = [ final_vector[3], final_vector[4] ];

            if ( flag_write_states_to_ephem == true )
                eph = add_data( eph, et, SC.position[1], SC.position[2], SC.velocity[1], SC.velocity[2] );
            end
            
        end

    end

    if (flag_write_states_to_ephem == true);
        return SC, sol, eph;
    else
        return SC, sol;
    end
    
end


