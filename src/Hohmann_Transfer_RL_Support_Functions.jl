function HT_RL_step( S::Vector{Float64}, A::Vector{Float64}, γ::Float64, SC1::Spacecraft, list_CB::Vector{Celestial_Body}, propagator_settings::Dict,
    eph = nothing, extra_prop = nothing );

    #extract state vector components
    position_sc_0 = [ S[1], S[2] ];
    velocity_sc_0 = [ S[3], S[4] ];
    desired_sma   = S[6];

    #extract action vector components
    wait_time_days      = A[1];
    dV1                 = A[2];
    transfer_time_days  = A[3];
    dV2                 = A[4];

    #assign initial state vector to spacecraft
    SC1.position = position_sc_0;
    SC1.velocity = velocity_sc_0;

    #convert transfer and wait times to seconds
    wait_time_sec = wait_time_days * 86400;
    transfer_time_sec = transfer_time_days * 86400;

    initial_epoch = SC1.epoch;
    ref_mass = SC1.mass;

    flag_impact::Bool = false;
    flag_terminal::Bool = false;

    if ( propagator_settings["flag_write_ephemeris_states"] == false )

        #step to elapsed_time
        SC1, sol = step_SC_to_elapsed_time( SC1, initial_epoch, wait_time_sec, list_CB, propagator_settings["flag_fixed_step"], 
        propagator_settings["flag_write_ephemeris_states"] );

        #maneuver the spacecraft
        arr_V_hat               = SC1.velocity / norm( SC1.velocity );      #find current velocity direction
        arr_dV1                 = dV1 * arr_V_hat;                          #apply maneuever in current direction
        SC1, flag_mnvr, str_mnvr = ManeuverSpacecraft( SC1, arr_dV1, list_CB[1], "Inertial" ); #maneuver function

        #step to elapsed_time
        SC1, sol = step_SC_to_elapsed_time( SC1, initial_epoch, transfer_time_sec, list_CB, propagator_settings["flag_fixed_step"], 
        propagator_settings["flag_write_ephemeris_states"] );

        #maneuver the spacecraft
        arr_V_hat               = SC1.velocity / norm( SC1.velocity );      #find current velocity direction
        arr_dV2                 = dV2 * arr_V_hat;                          #apply maneuever in current direction
        SC1, flag_mnvr, str_mnvr = ManeuverSpacecraft( SC1, arr_dV2, list_CB[1], "Inertial" );

    else

        #step to elapsed_time
        SC1, sol, eph = step_SC_to_elapsed_time( SC1, initial_epoch, wait_time_sec, list_CB, propagator_settings["flag_fixed_step"], 
        propagator_settings["flag_write_ephemeris_states"], eph );

        #maneuver the spacecraft
        arr_V_hat               = SC1.velocity / norm( SC1.velocity );      #find current velocity direction
        arr_dV1                 = dV1 * arr_V_hat;                          #apply maneuever in current direction
        SC1, flag_mnvr, str_mnvr = ManeuverSpacecraft( SC1, arr_dV1, list_CB[1], "Inertial" );

        #step to elapsed_time
        SC1, sol, eph = step_SC_to_elapsed_time( SC1, initial_epoch, transfer_time_sec, list_CB, propagator_settings["flag_fixed_step"], 
        propagator_settings["flag_write_ephemeris_states"], eph );

        #maneuver the spacecraft
        arr_V_hat               = SC1.velocity / norm( SC1.velocity );      #find current velocity direction
        arr_dV2                 = dV2 * arr_V_hat;                          #apply maneuever in current direction
        SC1, flag_mnvr, str_mnvr = ManeuverSpacecraft( SC1, arr_dV2, list_CB[1], "Inertial" );

    end

    #spacecraft final state
    S_out = [ SC1.position[1], SC1.position[2], SC1.velocity[1], SC1.velocity[2], S[5], S[6] ];

    #general reward function based on final spacecraft state
    r, flag_terminal = HT_reward( SC1, list_CB[1], flag_impact, desired_sma, ref_mass );

    if ( extra_prop !== nothing )

        extra_prop_sec = extra_prop * 86400;

        #step to elapsed_time
        SC1, sol, eph = step_SC_to_elapsed_time( SC1, initial_epoch, extra_prop_sec, list_CB, propagator_settings["flag_fixed_step"], 
        propagator_settings["flag_write_ephemeris_states"], eph );
    end

    

    if ( propagator_settings["flag_write_ephemeris_states"] == true  )
        return S_out, r,flag_terminal, eph;
    else
        return S_out, r, flag_terminal;
    end

    
end


function HT_reward( SC::Spacecraft, CB::Celestial_Body, flag_impact::Bool, desired_sma::Float64, ref_mass )

    flag_terminal::Bool = false;

    pos_mag = norm( SC.position );

    a, e, ω, θ = Calculate_Planar_OE( SC, CB );
    r_p        = a * ( 1 - e );

    if ( pos_mag < CB.r || r_p < CB.r )
        flag_impact = true;
        flag_terminal = true;
    end

    #determine percent diff in A
    sma_diff            = a - desired_sma;
    reward_sma          = exp( - sma_diff^2 / (17000)^2 );

    #=
    println("   Position mag: " * string(pos_mag) );
    println("   Flag impact: " * string(flag_impact) );
    println("   Current sma: " * string(a) );
    println("   Current e: " * string(e) );
    println("   Current r_p: " * string(r_p) );
    println("   desired_sma: " * string(desired_sma) );
    println("   Pct diff: " * string(sma_diff/desired_sma) );
    =#

    #determine percent of mass delivered
    mass_frac = SC.mass / ref_mass;

    r = reward_sma;

    if ( flag_impact == true || a < 0 )
        r = r - 1;
        flag_terminal   = true;
    end

    return r, flag_terminal;

end

function HT_RL_rand_action( list_ranges::Vector{Array{Float64}} )

    num_ranges = length( list_ranges );
    A_rand::Vector{Float64} = zeros(num_ranges);

    for i in range(1,num_ranges)
        val_range = list_ranges[i];
        mean = ( val_range[2] + val_range[1] ) / 2;
        width = abs( val_range[2] - val_range[1] );
        r_a::Float64 = rand() * width + val_range[1];
        #println("i: " * string(i) * "   mean: " * string(mean) * "   width: " * string(width) * "   r_a: " * string(r_a) );
        A_rand[i] = r_a;
    end

    return A_rand;
    
end


function HT_RL_apply_actor_constraints( A::Vector{Float64}, list_ranges::Vector{Array{Float64}} )
    
    num_A = length(A);
    A_out = A;

    for i in range(1,num_A)

        action_i = A_out[i];

        lower_bound = list_ranges[i][1];
        upper_bound = list_ranges[i][2];

        if ( action_i < lower_bound )
            action_i = lower_bound;
            A_out[i] = action_i;
        elseif ( action_i > upper_bound )
            action_i = upper_bound;
            A_out[i] = action_i;
        end

    end

    return A_out::Vector{Float64};

end
