mutable struct Policy_Params

    dV::Array{Float64};
    p_dV::Array{Float64};

end

function deepcopy( Θ_in::Policy_Params )

    return Policy_Params( Θ_in.dV, Θ_in.p_dV );
    
end

function Policy( S::Tuple, Θ_params::Policy_Params )

    #parse state tuple
    r_rel_CB1 = S[1];
    r_rel_CB2 = S[2];
    v_rel_CB1 = S[3];
    v_rel_CB2 = S[4];
    mu_CB1 = S[5];
    mu_CB2 = S[6];
    SC_mass = S[7];
    SC_fuel_mass = S[8];

    FMR = SC_fuel_mass/ SC_mass;

    #=
    display( r_rel_CB1 / mu_CB1 );
    display( r_rel_CB2 / mu_CB2 );
    display( v_rel_CB1 / mu_CB1 );
    display( v_rel_CB2 / mu_CB2 );
    =#

    A::Array{Float64} = Θ_params.dV;


    return A;

end;



function ValueRollout( Θ_params::Policy_Params, S_input::Tuple, γ::Float64, prop_settings1, SC_input::Spacecraft, list_celestial_bodies_input::Array{Celestial_Body} )

    max_steps::Int = prop_settings1.max_steps_rollout;
    time_per_step::Float64 = prop_settings1.time_per_step;
    time_limit_periapsis_step::Float64 = prop_settings1.time_limit_periapsis_step;
    num_max_steps_between_mans::Float64 = prop_settings1.num_max_steps_between_mans;

    count::Int = 0;
    flag_terminal::Bool = false;
    flag_maneuver::Bool = false;
    R::Float64 = 0.0;
    r::Float64 = 0.0;
    eph = Ephemeris( [], [], [], [], [], 0 );
    elapsed_time::Float64 = 0.0;
    num_steps_between_mans::Int = 0;

    #make copies to prevent undesired changes
    SC = deepcopy( SC_input );
    list_celestial_bodies = copy( list_celestial_bodies_input );


    while ( count < max_steps && flag_terminal == false )

        r_rel_1 = SC.position - list_celestial_bodies[1].position;
        r_rel_2 = SC.position - list_celestial_bodies[2].position;
        v_rel_1 = SC.velocity - list_celestial_bodies[1].velocity;
        v_rel_2 = SC.velocity - list_celestial_bodies[2].velocity;

        #construct state tuple
        S = ( r_rel_1, v_rel_1, r_rel_2, v_rel_2, list_celestial_bodies[1].mu, list_celestial_bodies[2].mu, SC.mass, SC.fuel_mass, flag_terminal );
        
        #resultant delta V vector is output of our policy function
        A_in = Policy( S, Θ_params );

        

        #step forward 
        while ( num_steps_between_mans < num_max_steps_between_mans && flag_terminal == false )

            #only maneuver on 0
            if ( num_steps_between_mans != 0 )
                A = zeros( length(A_in) );
            else
                A = A_in;
                num_steps_between_mans = 0;
            end

            SC, eph, elapsed_time, flag_terminal, r = StepToPeriapsis( SC, list_celestial_bodies, elapsed_time, time_limit_periapsis_step, eph, A, false );

            num_steps_between_mans  = num_steps_between_mans + 1;
            count                   = count + 1;

            #tally reward
            R = R + r * γ^(count);

        end

        a, e, ω, θ = Calculate_Planar_OE( SC, list_celestial_bodies[1] );

        println("Count:   " * string( count ) * "   r: " * string(r) * "   R: " * string( R ), "   a: " * string(a) );

    end

    #println("Input A: " * string( Policy(S_input, Θ_params ) ) * "   Steps completed: " * string(count) * "   R: " * string(R) );

    return R::Float64;
    
end




function policy_gradient_ascent( Θ_IN::Policy_Params, SC_input::Spacecraft, list_celestial_bodies_input::Array{Celestial_Body}, prop_settings1, S::Tuple, γ::Float64, α::Float64, ϵ::Float64,
    δ_R_frac::Float64  )

    Θ_OUT::Policy_Params = deepcopy( Θ_IN );

    #paramerers/perturbations to search over
    params::Array{Float64}          = copy( Θ_IN.dV );
    perturbs::Array{Float64}        = copy( Θ_IN.p_dV );

    flag_stop_line_search::Bool     = false;
    flag_stop_gradient_calc::Bool   = false;
    flag_stop_successful_calc::Bool = false;
    max_line_search_steps::Int      = 10;
    count_line_search_steps::Int    = 0;
    num_random_hops::Int            = 0;
    max_random_hops::Int            = 0;

    #make copies to prevent undesired changes
    SC::Spacecraft = deepcopy( SC_input );
    list_celestial_bodies::Array{Celestial_Body} = copy( list_celestial_bodies_input );

    R0::Float64 = ValueRollout( Θ_OUT, S, γ, prop_settings1, SC, list_celestial_bodies );

    #initialize empty array
    ∇Θ::Array{Float64} = [];

    #ADD LOGIC HERE FOR RANDOM RESTARTS IF THE GRAD IS ZERO
    while ( flag_stop_gradient_calc == false )

        #clear gradient vector
        ∇Θ = [];
        
        #step through the parameters and perturb
        for i in range(1, length(params) )

            Θ_p::Policy_Params = deepcopy( Θ_IN );

            p_perturbed::Float64 = copy(params[i]) + copy(perturbs[i]);

            params_temp = copy(params);
            params_temp[i] = p_perturbed;

            Θ_p.dV = params_temp;

            R_p::Float64 = ValueRollout( Θ_p, S, γ, prop_settings1, SC, list_celestial_bodies );

            #find change in trajectory reward
            ΔR = R_p - R0;

            #find partial derivative wrt parameter
            δR_δp = ΔR / perturbs[i];

            #add partial to gradient vector
            push!( ∇Θ, δR_δp );

        end

        #we can succesfully exit if the gradient is non-zero
        if ( norm( ∇Θ ) != 0 )
            flag_stop_gradient_calc = true;
            flag_stop_successful_calc = true;
        else 

            if ( num_random_hops > max_random_hops ) 

                flag_stop_gradient_calc = true;
                flag_stop_successful_calc = false;

            else

                num_random_hops = num_random_hops + 1;

                old_params = params;
                v_mag_old = norm( params );

                #find a new random maneuver direction
                dx_rand = rand();
                dy_rand = rand();

                v_hat_rand = [ dx_rand, dy_rand ];
                v_hat_rand = v_hat_rand / norm( v_hat_rand );
                v_max = GetMaxDVMag( SC );
                v_mag_new = rand() * v_max;
                v_vec = v_mag_new * v_hat_rand;

                params = v_vec;

                println("Random hop #: " * string( num_random_hops ) * "   Old dV: " * string(old_params) * "   New dV: " * string(params) );

            end



        end

    end

    if ( flag_stop_successful_calc == false )
        println("Gradient computation failed, returning input dV: " * string(Θ_OUT.dV) )
        return Θ_OUT, ϵ;
    end

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #vanilla gradient ascent (method 1)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #params_updated = params + α * ∇Θ;

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #restricted gradient ascent (method 2)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #u_hat = ∇Θ / norm( ∇Θ );
    #params_updated = params + sqrt( 2 * ϵ ) * u_hat;

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #scale factor line search (method 3)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    u_hat = ∇Θ / norm( ∇Θ );
    count_line_search_steps = 0;
    params_updated = params + sqrt( 2 * ϵ ) * u_hat;


    while ( flag_stop_line_search == false && count_line_search_steps < max_line_search_steps )

        println("Line search: " * string(count_line_search_steps) * "   ϵ: " * string(ϵ) * "   p: " * string(params_updated) );
        
        #check the reward rollout
        params_updated = params + sqrt( 2 * ϵ ) * u_hat;
        Θ_OUT.dV = params_updated;
        R_test::Float64 = ValueRollout( Θ_OUT, S, γ, prop_settings1, SC, list_celestial_bodies );

        R_p_diff = abs( (R_test - R0) / R0 );

        #print("Line search: " * string(count_line_search_steps) * "   ϵ: " * string(ϵ) * "   p: " * string(params_updated) * "   R_p_diff: " * string(R_p_diff) );

        #check if we have exceeded the allowable reward percent diff, if so reduce eps and perturbations and check again
        if ( R_p_diff > δ_R_frac )
            ϵ = ϵ / 2;
            perturbs = perturbs / 2;
            Θ_OUT.p_dV = perturbs;
        else
            flag_stop_line_search = true;
        end

        count_line_search_steps = count_line_search_steps + 1;

    end

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    Θ_OUT.dV = params_updated;

    R_test = ValueRollout( Θ_OUT, S, γ, prop_settings1, SC, list_celestial_bodies );

    println("   GA:   Input: " * string( Θ_IN ) * "   Output: " * string( Θ_OUT ) * "   R_final: " * string(R_test) );

    return Θ_OUT, ϵ;
    
end

