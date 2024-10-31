mutable struct Spacecraft

    name::String;               #Name of the body
    epoch::Float64;             #Current epoch of the spacecraft
    position::Array{Float64};   #position vector in km
    velocity::Array{Float64};   #velocity vector in km/s
    mass::Float64;              #mass of the Spacecraft in kg
    fuel_mass::Float64;         #mass of the available propellant in kg
    fuel_ISP::Float64;          #specific impulse of the fuel used in s
    net_force::Array{Float64};  #net force on SC in kN
    step_size::Float64;         #propagator step size in seconds

end

function deepcopy( SC::Spacecraft )

    return Spacecraft( SC.name, SC.epoch, SC.position, SC.velocity, SC.mass, SC.fuel_mass, SC.fuel_ISP, SC.net_force, SC.step_size );
    
end

function Calculate_Net_Gravitational_Force( position::Array{Float64}, list_celestial_bodies::Array{Celestial_Body}, mass::Float64 )

    num_celestial_bodies = length( list_celestial_bodies );

    if( num_celestial_bodies < 1 )
        error("Include a central body");
    end

    #determine the number of dimensions (2 or 3 dims)
    dimensions = length( list_celestial_bodies[1].position );

    #create a net force vector array
    arr_a_Net = zeros( Float64, dimensions );

    for i in 1:num_celestial_bodies

        #determine relative position with respect to central body
        arr_relative_position = position - list_celestial_bodies[i].position;

        #find distance to central body
        r = 0.0;
        for j in 1:dimensions
            r = r + arr_relative_position[j]^2;
        end
        r = sqrt( r );

        #find the acceleration due to gravity in km/s^2
        arr_a = - list_celestial_bodies[i].mu * arr_relative_position / ( r^3 );

        #add to net acceleration
        arr_a_Net = arr_a_Net + arr_a;

    end

    #net force returned is in kN
    arr_F_Net = mass * arr_a_Net;

    return arr_F_Net;
    
end


function RK4_Propagate( SC::Spacecraft, list_celestial_bodies::Array{Celestial_Body} );

    nan_i = findall( isnan, SC.position );
    if ( isempty(nan_i) == false )
        println( "SC r: " * string( SC.position ) );
        println( "SC v: " * string( SC.velocity ) );
        error("NaN detected in propagation");
    end

    nan_i = findall( isnan, SC.velocity );
    if ( isempty(nan_i) == false )
        println( "SC r: " * string( SC.position ) );
        println( "SC v: " * string( SC.velocity ) );
        error("NaN detected in propagation");
    end

    #display( SC.position );
    #display( SC.velocity );

    dt = SC.step_size;
    net_a_0 = Calculate_Net_Gravitational_Force( SC.position, list_celestial_bodies, SC.mass ) / SC.mass;

    #vector k1 terms
    k1_vp1 = net_a_0;
    k1_rp1 = SC.velocity;

    #vector k2 terms
    r_temp = SC.position + k1_rp1 * dt / 2;
    k2_vp1 = Calculate_Net_Gravitational_Force( r_temp, list_celestial_bodies, SC.mass ) / SC.mass;
    k2_rp1 = SC.velocity .* k1_vp1 * dt / 2;

    #vector k3 terms
    r_temp = SC.position + k2_rp1 * dt / 2;
    k3_vp1 = Calculate_Net_Gravitational_Force( r_temp, list_celestial_bodies, SC.mass ) / SC.mass;
    k3_rp1 = SC.velocity .* k2_vp1 * dt / 2;

    #vector k4 terms
    r_temp = SC.position + k3_rp1 * dt;
    k4_vp1 = Calculate_Net_Gravitational_Force( r_temp, list_celestial_bodies, SC.mass ) / SC.mass;
    k4_rp1 = SC.velocity .* k3_vp1 * dt;

    #propagation step
    position_update::Array{Float64} = SC.position + dt/6*( k1_rp1 + 2*k2_rp1 + 2*k3_rp1 + k4_rp1 );
    velocity_update::Array{Float64} = SC.velocity + dt/6*( k1_vp1 + 2*k2_vp1 + 2*k3_vp1 + k4_vp1 );

    #position_update::Array{Float64} = SC.position + SC.velocity*dt;
    #velocity_update::Array{Float64} = SC.velocity + net_a_0*dt;

    #display( position_update );
    #display( velocity_update );


    return position_update::Array{Float64}, velocity_update::Array{Float64};
    
end


function Calculate_Planar_OE( SC::Spacecraft, CB::Celestial_Body )

    X::Float64 = SC.position[1];
    Y::Float64 = SC.position[2];
    R::Float64 = norm( SC.position );

    VX::Float64 = SC.velocity[1];
    VY::Float64 = SC.velocity[2];
    V::Float64 = norm( SC.velocity );

    SC_pos::Array{Float64} = [ X, Y, 0.0 ];
    SC_vel::Array{Float64} = [ VX, VY, 0.0 ];
    Z_hat::Array{Float64} = [ 1.0, 0.0, 0.0 ];

    #angular momentum vector
    h_vec = cross( SC_pos, SC_vel );
    h = norm( h_vec );
    h_hat = h_vec / h;

    N = cross( Z_hat, h_hat );
    N_hat = N / norm(N);

    r_hat = SC_pos / R;

    #specific energy
    eps::Float64 = V^2 / 2 + CB.mu / R;

    #eccentricity vector
    e_vec = cross( SC_vel, h_vec ) / CB.mu - SC_pos / R;
    e = norm( e_vec );
    e_hat = e_vec/e;

    #semi-major axis
    rp::Float64 = h^2 / CB.mu / ( 1 + e * cos(0) );
    ra::Float64 = h^2 / CB.mu / ( 1 + e * cos(π) );
    a::Float64  = 1/2 * ( ra + rp );

    #argument of periapsis
    if ( e_vec[3] >= 0 )
        ω = acos( dot(N_hat,e_hat) );
    else
        ω = 2 * π - acos( dot(N_hat,e_hat) );
    end
  
    #true anomaly
    if ( dot(SC_pos, SC_vel) >= 0 )

        dotp = dot( e_hat, r_hat );
        if ( dotp < -1 ) dotp = -1; end

        if ( abs( dot( e_hat, r_hat ) ) < 1.0  )
            θ = acos( dot( e_hat, r_hat ) );
        elseif ( dot( e_hat, r_hat ) < -1.0 )
            θ = π
        else
            θ = 0
        end


    else

        dotp = dot( e_hat, r_hat );
        if ( dotp < -1 ) dotp = -1; end

        try 
            θ = 2 * π - acos( dot( e_hat, r_hat ) );
        catch e
            
            if isa(e, DomainError)
                θ = π;
            else
                rethrow(e);
            end

        end

    end



    return a, e, ω, θ;
    
end


function reward( SC::Spacecraft, CB::Celestial_Body, flag_impact::Bool, flag_escape::Bool, desired_range_limit::Float64 )

    flag_terminal::Bool = false;

    if ( flag_impact == true )

        r_a = 0.0;
        a = 0.0;
        e = 0.0;
        r = -100;
        flag_terminal = true;
        
    else

        a, e, ω, θ = Calculate_Planar_OE( SC, CB );

        #determine apoapsis radius
        r_a = a * ( 1 + e );

        if ( r_a < desired_range_limit && a > 0 )
            T = 2 * π * sqrt( a^3 / CB.mu ) / 86400;
            r = T;
        elseif ( r_a > desired_range_limit && a > 0 )
            T = 2 * π * sqrt( a^3 / CB.mu ) / 86400;
            r = T * ( desired_range_limit/r_a )^4;
        else
            r = 0;
        end

        if ( a < 0 )
            r = 0;
            #flag_terminal = true;
        end

    end


    return r, flag_terminal;
    
end


function StepForward( SC::Spacecraft, list_celestial_bodies::Array{Celestial_Body}, et_total::Float64, time_to_step::Float64, eph::Ephemeris, A::Array{Float64},
    flag_save_plot_data::Bool = false );
    
    flag_impact::Bool = false;
    flag_escape::Bool = false;
    flag_terminal::Bool = false;
    flag_maneuver::Bool = false;
    elapsed_time::Float64 = 0.0;

    #maneuver the spacecraft
    SC, flag_maneuver = ManeuverSpacecraft( SC, A, list_celestial_bodies[1] );

    while ( elapsed_time < time_to_step )

        #if the save plotting data flag is active then save plotting data
        if ( flag_save_plot_data == true )
            eph = add_data( eph, et_total, SC.position[1], SC.position[2], SC.velocity[1], SC.velocity[2] );
        end

        #check for a collision with any central body
        for central_body in list_celestial_bodies

            r_vec_rel::Array{Float64} = SC.position - central_body.position;
            r_current::Float64 = norm( r_vec_rel );

            if ( r_current < central_body.r )
                #println("Impacted central body");
                flag_impact = true
            end

        end

        if ( flag_impact == true )

            elapsed_time = elapsed_time + SC.step_size;

        else

            arr_pos_update::Array{Float64}, arr_vel_update::Array{Float64} = RK4_Propagate( SC, list_celestial_bodies );
            SC.position = arr_pos_update;
            SC.velocity = arr_vel_update;

            elapsed_time = elapsed_time + SC.step_size;
            et_total = et_total + SC.step_size;

        end

    end

    max_dist = 4 * list_celestial_bodies[1].r;
    CB = list_celestial_bodies[1];

    #determine reward
    r, flag_terminal = reward( SC, CB, flag_impact, flag_escape, max_dist );

    return SC, eph, et_total, flag_terminal, r;
    
end

function GetMaxDVMag( SC::Spacecraft )

    g_0::Float64 = 9.80665; #m/s^2
    m_0::Float64 = SC.mass;
    m_f_max_fuel::Float64 = SC.mass - SC.fuel_mass;
    max_dV_mag::Float64 = SC.fuel_ISP * g_0 * log( m_0 / m_f_max_fuel );
    max_dV_mag = max_dV_mag / 1000; #convert to km/s

    return max_dV_mag;
    
end


function ManeuverSpacecraft( SC::Spacecraft, dV::Array{Float64}, CB::Celestial_Body, str_mnvr_ref_frame::String )

    #=
    ManeuverSpacecraft function
    -----------------------------------------------------------------------------------
    This function maneuvers a spacecraft object given an input delta-V vector. 

    Inputs
    -----------------------------------------------------------------------------------
    SC:                     Spacecraft Object
    dV:                     Delta-V Component Vector
    CB:                     Central Body Object
    str_mnvr_ref_frame:     Reference Frame of the Maneuver

    Outputs
    -----------------------------------------------------------------------------------
    SC:                     The resultant spacecraft object
    flag_maneuver:          Flag indicating whether or not a maneuver successfully occured
    str_maneuver_report:    A string containing the maneuver specifics
    =#

    t = SC.epoch;

    flag_maneuver::Bool = false
    g_0::Float64 = 9.80665; #m/s^2

    if ( str_mnvr_ref_frame == "LVLH" )

        r_SC = [ SC.position[1] - CB.position[1], SC.position[2] - CB.position[2], 0.0 ];
        v_SC = [ SC.velocity[1] - CB.velocity[1], SC.velocity[2] - CB.velocity[2], 0.0 ];
        r_hat = r_SC / norm( r_SC );
        h_hat = cross( r_SC, v_SC ) / norm( cross( r_SC, v_SC ) );
        y_hat = cross( h_hat, r_hat ) / norm( cross( h_hat, r_hat ) );

        DCM_I_LVLH = vcat( r_hat', y_hat', h_hat' );
        nan_indices = findall(isnan, DCM_I_LVLH);

        #check for NaN 
        if ( isempty(nan_indices) == false )
            println( "r_SC: " * string(r_SC) );
            println( "v_SC: " * string(v_SC) );
            display(DCM_I_LVLH);
        end

        #convert delta-V to LVLH
        DCM_LVLH_I = inv( DCM_I_LVLH );
        dV_vector_LVLH = reshape( [dV[1], dV[2], 0.0 ], 3, 1 );
        dV_vector_I = DCM_LVLH_I * dV_vector_LVLH;

    elseif ( str_mnvr_ref_frame == "Inertial" )

        dV_vector_I = dV;
    
    else

        error("Unrecongnized ref frame: " * str_mnvr_ref_frame )

    end

    #reset dV to inertial frame
    dV = [ dV_vector_I[1], dV_vector_I[2] ];
    input_dv_mag::Float64 = norm(dV);
    input_dv_hat::Array{Float64} = dV / input_dv_mag;

    #=
    println("\n\n\n\n\n");
    println("Input DV col vec in LVLH");
    display( dV_vector_LVLH );
    println("I_LVLH DCM");
    display( DCM_I_LVLH );
    println("LVLH_I DCM");
    display( DCM_LVLH_I );
    println("Input DV col vec in I");
    display( dV_vector_I );
    println("Final dV inertial vector");
    display(dV);
    =#

    #If there is no fuel remaining or if the input dV vector is zero, then no maneuver is performed
    if ( SC.fuel_mass == 0.0 )
        return SC, flag_maneuver, "";
    elseif ( norm(dV) == 0.0 )
        return SC, flag_maneuver, "";
    elseif ( SC.fuel_mass < 0.0 )
        error("Error: negative fuel mass");
    end

    #determine the maximum maneuver size based off of remaining fuel.
    m_0 = SC.mass;
    m_f_max_fuel = SC.mass - SC.fuel_mass;
    max_dV_mag = SC.fuel_ISP * g_0 * log( m_0 / m_f_max_fuel );
    max_dV_mag = max_dV_mag / 1000; #convert to km/s

    #if the input maneuver size exceeds the fuel mass available then we perform the largest maneuver possible
    if ( input_dv_mag > max_dV_mag )

        #maneuver direction is unchanged but magnitude is modified
        dV = max_dV_mag * input_dv_hat;

        #update the fuel available on spacecraft and overall mass
        m_f = m_f_max_fuel;
        SC.mass = m_f;
        SC.fuel_mass = 0.0;
        m_fuel_used = m_0 - m_f;

        #update SC velocity
        SC.velocity = SC.velocity + dV;

        flag_maneuver = true;

    #else perform the maneuver
    else

        v_e = SC.fuel_ISP * g_0 / 1000;
        m_f = m_0 / ( exp( input_dv_mag/v_e ) );
        m_fuel_used = m_0 - m_f;
        SC.mass = m_f;
        SC.fuel_mass = SC.fuel_mass - m_fuel_used;

        #update SC velocity
        SC.velocity = SC.velocity + dV;

        flag_maneuver = true;

    end

    #=
    println("r_hat:" * string(r_hat) );
    println("h_hat:" * string(h_hat) );
    println("Initial mass: " * string(m_0) );
    println("Final mass: " * string(m_f) );
    println("Fuel Used: " * string(m_fuel_used) );
    println("Max dV mag: " * string(max_dV_mag) );
    println("Input dV mag: " * string(input_dv_mag) );
    #error("STOP");
    =#
    
    #store data in maneuver report
    str_maneuver_report = string(t) * "," * string(dV[1]) * "," * string(dV[2]);

    return SC, flag_maneuver, str_maneuver_report;

    
end


function StepToPeriapsis_old( SC::Spacecraft, list_celestial_bodies::Array{Celestial_Body}, et_total::Float64, time_limit::Float64, eph::Ephemeris, A::Array{Float64},
    flag_save_plot_data::Bool = false, maneuver_history::Array{String} = Array{String}(undef, 0) );
    
    flag_impact::Bool = false;
    flag_terminal::Bool = false;
    flag_maneuver::Bool = false;
    elapsed_time::Float64 = 0.0;
    flag_stop::Bool = false;
    flag_escape::Bool = false;
    ang_dist_traveled::Float64 = 0.0;
    flag_start_ang_count::Bool = false;

    #error handling
    nan_indices = findall(isnan, SC.position );
    if ( isempty( nan_indices ) == false )
        println("Total elapsed time: " * string(et_total) );
        println("A: " * string(A) );
    end

    #maneuver the spacecraft
    SC, flag_maneuver, str_man_sum = ManeuverSpacecraft( SC, A, list_celestial_bodies[1], et_total );

    if ( flag_save_plot_data == true && flag_maneuver == true )
        push!(maneuver_history, str_man_sum );
    end

    #perform an initial check of the orbital elements
    a, e, ω, θ = Calculate_Planar_OE( SC, list_celestial_bodies[1] );

    #determine orbital period if there is one
    if ( a > 0 )
        T = 2*π*sqrt( a^3 / list_celestial_bodies[1].mu );
    else
        T = Inf;
    end

    #println( "Step to periapsis: T: " * string(T) );

    θ_prev = θ;

    while ( flag_stop == false )

        #println("Elapsed t: " * string(elapsed_time) * "   " * string( θ*180/π ) * "   ADT: " * string(ang_dist_traveled) );

        a, e, ω, θ = Calculate_Planar_OE( SC, list_celestial_bodies[1] );

        d_theta = ( θ - θ_prev ) * 180/π ;

        if ( flag_start_ang_count == false && θ_prev != 0.0 )
            flag_start_ang_count = true;
        end

        if ( flag_start_ang_count == true )
            if ( d_theta < 0 )
                ang_dist_traveled = ang_dist_traveled + d_theta + 360;
            else
                ang_dist_traveled = ang_dist_traveled + d_theta;
            end
        end

        #if the save plotting data flag is active then save plotting data
        if ( flag_save_plot_data == true )
            eph = add_data( eph, et_total, SC.position[1], SC.position[2], SC.velocity[1], SC.velocity[2] );
        end


        #check for a collision with any central body
        for central_body in list_celestial_bodies

            r_vec_rel::Array{Float64} = SC.position - central_body.position;
            r_current::Float64 = norm( r_vec_rel );

            if ( r_current < central_body.r )
                #println("Impacted central body");
                flag_impact = true
            end

        end

        if ( flag_impact == true )

            flag_stop = true;

        else

            arr_pos_update::Array{Float64}, arr_vel_update::Array{Float64} = RK4_Propagate( SC, list_celestial_bodies );
            SC.position = arr_pos_update;
            SC.velocity = arr_vel_update;

            elapsed_time = elapsed_time + SC.step_size;

        end

        elapsed_time = elapsed_time + SC.step_size;
        et_total = et_total + SC.step_size;

        #check stop conditions
        if ( elapsed_time > time_limit )
            flag_stop = true;
            flag_escape = true;
        elseif ( a < 0 )
            #flag_stop = true;
            #flag_escape = true;
        end


        if ( d_theta < 0 && elapsed_time > T/10 )
            flag_stop = true;
        end;

        θ_prev = θ;
        
    end

    #determine reward
    CB = list_celestial_bodies[1];
    max_dist = 4 * list_celestial_bodies[1].r;
    r, flag_terminal = reward( SC, CB, flag_impact, flag_escape, max_dist );

    #=
    println("Impact?: " * string(flag_impact ) );
    println("Escape?: " * string( flag_escape ) );
    println("Elapsed time: " * string(elapsed_time) );
    println("Time limit: " * string(time_limit ) );
    println("Terminal? " * string(flag_terminal ) );
    println("Theta: " * string( θ*180/pi )  );
    =#
    

    #error("STOP");
    return SC, eph, et_total, flag_terminal, r, maneuver_history;


end


function StepToPeriapsis( SC::Spacecraft, list_celestial_bodies::Array{Celestial_Body}, et_total::Float64, time_limit::Float64, eph::Ephemeris, A::Array{Float64},
    flag_save_plot_data::Bool = false, maneuver_history::Array{String} = Array{String}(undef, 0) );
    
    flag_impact::Bool = false;
    flag_terminal::Bool = false;
    flag_maneuver::Bool = false;
    elapsed_time::Float64 = 0.0;
    flag_stop::Bool = false;
    flag_escape::Bool = false;
    ang_dist_traveled::Float64 = 0.0;
    flag_start_ang_count::Bool = false;

    #error handling
    nan_indices = findall(isnan, SC.position );
    if ( isempty( nan_indices ) == false )
        println("Total elapsed time: " * string(et_total) );
        println("A: " * string(A) );
    end

    #maneuver the spacecraft
    SC, flag_maneuver, str_man_sum = ManeuverSpacecraft( SC, A, list_celestial_bodies[1], et_total );

    if ( flag_save_plot_data == true && flag_maneuver == true )
        push!(maneuver_history, str_man_sum );
    end

    #perform an initial check of the orbital elements
    a, e, ω, θ = Calculate_Planar_OE( SC, list_celestial_bodies[1] );

    #determine orbital period if there is one
    if ( a > 0 )
        T = 2*π*sqrt( a^3 / list_celestial_bodies[1].mu );
    else
        T = Inf;
    end

    #println( "Step to periapsis: T: " * string(T) );

    θ_prev = θ;

    while ( flag_stop == false )

        #println("Elapsed t: " * string(elapsed_time) * "   " * string( θ*180/π ) * "   ADT: " * string(ang_dist_traveled) );

        a, e, ω, θ = Calculate_Planar_OE( SC, list_celestial_bodies[1] );

        d_theta = ( θ - θ_prev ) * 180/π ;

        if ( flag_start_ang_count == false && θ_prev != 0.0 )
            flag_start_ang_count = true;
        end

        if ( flag_start_ang_count == true )
            if ( d_theta < 0 )
                ang_dist_traveled = ang_dist_traveled + d_theta + 360;
            else
                ang_dist_traveled = ang_dist_traveled + d_theta;
            end
        end

        #if the save plotting data flag is active then save plotting data
        if ( flag_save_plot_data == true )
            eph = add_data( eph, et_total, SC.position[1], SC.position[2], SC.velocity[1], SC.velocity[2] );
        end


        #check for a collision with any central body
        for central_body in list_celestial_bodies

            r_vec_rel::Array{Float64} = SC.position - central_body.position;
            r_current::Float64 = norm( r_vec_rel );

            if ( r_current < central_body.r )
                #println("Impacted central body");
                flag_impact = true
            end

        end

        if ( flag_impact == true )

            flag_stop = true;

        else

            arr_pos_update::Array{Float64}, arr_vel_update::Array{Float64} = RK4_Propagate( SC, list_celestial_bodies );
            SC.position = arr_pos_update;
            SC.velocity = arr_vel_update;

            elapsed_time = elapsed_time + SC.step_size;

        end

        elapsed_time = elapsed_time + SC.step_size;
        et_total = et_total + SC.step_size;

        #check stop conditions
        if ( elapsed_time > time_limit )
            flag_stop = true;
            flag_escape = true;
        elseif ( a < 0 )
            #flag_stop = true;
            #flag_escape = true;
        end


        if ( d_theta < 0 && elapsed_time > T/10 )
            flag_stop = true;
        end;

        θ_prev = θ;
        
    end

    #determine reward
    CB = list_celestial_bodies[1];
    max_dist = 4 * list_celestial_bodies[1].r;
    r, flag_terminal = reward( SC, CB, flag_impact, flag_escape, max_dist );

    #=
    println("Impact?: " * string(flag_impact ) );
    println("Escape?: " * string( flag_escape ) );
    println("Elapsed time: " * string(elapsed_time) );
    println("Time limit: " * string(time_limit ) );
    println("Terminal? " * string(flag_terminal ) );
    println("Theta: " * string( θ*180/pi )  );
    =#
    

    #error("STOP");
    return SC, eph, et_total, flag_terminal, r, maneuver_history;


end