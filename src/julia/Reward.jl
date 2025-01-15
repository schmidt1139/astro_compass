function reward_SC( SC::Spacecraft, CB::Celestial_Body, flag_impact::Bool, desired_range_limit::Float64 )

    flag_terminal::Bool = false;

    if ( flag_impact == true )

        r = -100;
        flag_terminal = true;
        
    else

        a, e, ω, θ = Calculate_Planar_OE( SC, CB );

        #determine apoapsis radius
        r_a = a * ( 1 + e );

        if ( r_a < desired_range_limit && a > 0 )
            r = 1;
        else
            r = 0;
        end

    end
    
end