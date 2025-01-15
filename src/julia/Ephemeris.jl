mutable struct Ephemeris
    arr_elapsed_time::Array{Float64};
    arr_X::Array{Float64};
    arr_Y::Array{Float64};
    arr_VX::Array{Float64};
    arr_VY::Array{Float64};
    num_vectors::Int;
end

function add_data( Eph::Ephemeris, elapsed_time::Float64, X::Float64, Y::Float64, VX::Float64, VY::Float64 );

    #add data to Ephemeris
    push!( Eph.arr_elapsed_time, elapsed_time );
    push!( Eph.arr_X, X );
    push!( Eph.arr_Y, Y );
    push!( Eph.arr_VX, VX );
    push!( Eph.arr_VY, VY );
    Eph.num_vectors = Eph.num_vectors + 1;

    return Eph;

end


function plot_ephem( Eph::Ephemeris, xy_scale::Float64, flag_plot_CB_rad::Bool, flag_CB_Name::String  )

    list_plots = [];
    arr_CB_rad_x = [];
    arr_CB_rad_y = [];

    p1 = plot( Eph.arr_X/xy_scale, Eph.arr_Y/xy_scale, ratio=:equal, label="Spacecraft" );


    if ( flag_plot_CB_rad == true )
    
        for i in 1:360
    
            x = xy_scale * cosd( i );
            y = xy_scale * sind( i );
    
            push!( arr_CB_rad_x, x );
            push!( arr_CB_rad_y, y );
    
        end

    end

    plot!( p1, arr_CB_rad_x/xy_scale, arr_CB_rad_y/xy_scale, ratio=:equal, label=flag_CB_Name );

    push!( list_plots, p1 );

    return list_plots;
    
end

function add_ODE_sol_to_ephem!( eph::Ephemeris, sol, flag_only_final_step::Bool )

    num_vectors = length(sol);

    #println("Solution vector length: " * string( num_vectors ) );

    if ( flag_only_final_step == false )

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
    
    else

        vec_i = num_vectors;
        vector = sol[vec_i];

        t_eph = sol.t[vec_i];
        x_eph = vector[1];
        y_eph = vector[2];
        vx_eph = vector[3];
        vy_eph = vector[4];

        eph = add_data( eph, t_eph, x_eph, y_eph, vx_eph, vy_eph );

    end

end