using ReinforcementLearning;
using Random;
using Plots;
using LinearAlgebra;
using IntervalSets;

include("Ephemeris.jl");
include("Celestial_Body.jl");
include("Spacecraft.jl");
include("Spacecraft_Propagation.jl");

struct Hohmann_Transfer_Env_Params{T}
    mu_cb::T
    radius_cb::T
    max_steps::Int
    position_extrema::T 
    velocity_extrema::T
end

mutable struct Hohmann_Transfer_Env{T,ACT} <: AbstractEnv
    params::Hohmann_Transfer_Env_Params{T}
    state::Vector{T}
    action::ACT
    done::Bool
    t::Int
    rng::AbstractRNG
end


Base.show(io::IO, params::Hohmann_Transfer_Env_Params) = print(
    io,
    join(["$p=$(getfield(params, p))" for p in fieldnames(Hohmann_Transfer_Env_Params)], ","),
)

#This function establishes a set of parameters that are used to initialize the Hohmann Transfer Env
function Hohmann_Transfer_Env_Params(;
    T = Float64,
    mu_cb = 4903.0,
    radius_cb = 1740.0,
    max_steps = 100,
    position_extrema = 1000000.0,
    velocity_extrema = 100.0
)

    Hohmann_Transfer_Env_Params{T}(
        mu_cb,
        radius_cb,
        max_steps,
        position_extrema,
        velocity_extrema
    )

end

function Hohmann_Transfer_Env(;
    T = Float64,
    rng=Random.default_rng(),
    kwargs... )

    params = Hohmann_Transfer_Env_Params(; T=T, kwargs... );

    env = Hohmann_Transfer_Env( params, zeros(T,6), 0.0, false, 0, rng );
    reset!(env);

    #return env
    env
    
end


function RLBase.reset!(env::Hohmann_Transfer_Env{T}) where {T}
    env.state[1] = 0.0;                             #X
    env.state[2] = 7000.0;                          #Y
    env.state[3] = sqrt(env.params.mu_cb/7000);     #VX
    env.state[4] = 0.0;                             #VY
    env.state[5] = env.params.mu_cb;                #Central Body mu
    env.state[6] = 14 * 1740.0;                     #target_sma
    env.done = false
    env.t = 0
    nothing
end

#use the env 'done' property to check if the environment has been terminated
RLBase.is_terminated( env::Hohmann_Transfer_Env ) = env.done;

#standard state interface using env property
RLBase.state( env::Hohmann_Transfer_Env, ::Observation, ::DefaultPlayer) = env.state;

#need to define state space for Hohmann transfer problem (X,Y,VX,VY,cb_mu,SMA_target)
function RLBase.state_space(env::Hohmann_Transfer_Env)

    
    #=
    (-env.params.position_extrema .. env.params.position_extrema,
    -env.params.position_extrema .. env.params.position_extrema,
    -env.params.velocity_extrema .. env.params.velocity_extrema,
    -env.params.velocity_extrema .. env.params.velocity_extrema,
    0.0 .. 1000000.0,
    0.0 .. 1000000.0)
    =#

    interval_1 = -env.params.position_extrema .. env.params.position_extrema
    interval_2 = -env.params.position_extrema .. env.params.position_extrema
    interval_3 = -env.params.velocity_extrema .. env.params.velocity_extrema
    interval_4 = -env.params.velocity_extrema .. env.params.velocity_extrema
    interval_5 = 0.0 .. 1.0e6
    interval_6 = -1.0e6 .. 1.0e6

    S = ( ( ( ( ( interval_1 × interval_2 ) × interval_3 ) × interval_4 ) × interval_5 ) × interval_6 )


end

#defining action space
#first '<:AbstractFloat' corresponds to the state vector type
#action is of range -1 to 1
RLBase.action_space(::Hohmann_Transfer_Env{<:AbstractFloat,<:AbstractFloat}) = (-1.0,1.0);


function RLBase.reward(env::Hohmann_Transfer_Env{T}) where {T}

    #unpack state vector
    x = env.state[1];
    y = env.state[2];
    vx = env.state[3];
    vy = env.state[4];
    mu = env.state[5];
    target_sma = env.state[6];

    #create spacecraft and central body objects
    central_body = Celestial_Body( "Moon", [0.0,0.0], [0.0,0.0], mu, 1740 );
    spacecraft = Spacecraft( "SC1", 0.0, [x,y], [vx,vy], 1000.0, 999.0, 220.0, [0.0, 0.0], 600.0 );

    #find orbital elements
    a, e, ω, θ = Calculate_Planar_OE( spacecraft, central_body );

    r_p        = a * ( 1 - e );
    pos_mag    = norm(spacecraft.position);           

    reward = 0.0;

    if ( pos_mag < central_body.r || r_p < central_body.r )
        flag_impact = true;
        flag_terminal = true;
        reward = - 100.0;
    end

    #determine percent diff in A
    sma_diff            = a - target_sma;
    reward              = reward + exp( - sma_diff^2 / (17000)^2 );

    #=
    println("Reward test function");
    println("a: " * string(a) );
    println("e: " * string(e) );
    println("w: " * string(ω) );
    println("θ: " * string(θ) );
    println("r_p: " * string(r_p) );
    println("reward: " * string(reward) );
    =#

    return reward;

end

function RLBase.act!(env::Hohmann_Transfer_Env, a::AbstractFloat)
    @assert a in action_space(env)
    env.action = a
    _step!(env, a)
end


function _step!( env::Hohmann_Transfer_Env, dV )

    #increment time counter
    env.t += 1;

    #unpack the state vector
    x = env.state[1];
    y = env.state[2];
    vx = env.state[3];
    vy = env.state[4];
    mu = env.state[5];
    target_sma = env.state[6];

    #create spacecraft and central body objects using state vector
    Moon = Celestial_Body( "Moon", [0.0,0.0], [0.0,0.0], mu, 1740 );
    spacecraft = Spacecraft( "SC1", 0.0, [x,y], [vx,vy], 1000.0, 999.0, 220.0, [0.0, 0.0], 600.0 );
    list_celestial_bodies::Vector{Celestial_Body} = [ Moon ];

    #find velocity direction
    v_vec::Vector = [ vx, vy ] ./ sqrt( vx^2 + vy^2 );

    #apply dV in velocity direction
    dV_vec = v_vec .* dV;

    #adjust spaceraft velocity by dV vector
    spacecraft.velocity = spacecraft.velocity + dV_vec;

    #propagate forward in time
    spacecraft, sol = step_SC_to_elapsed_time( spacecraft, 0.0, spacecraft.step_size, list_celestial_bodies, false, false );

    #record new state in env
    x_p = spacecraft.position[1];
    y_p = spacecraft.position[2];
    vx_p = spacecraft.velocity[1];
    vy_p = spacecraft.velocity[2];
    mu_p = mu;
    target_sma_p = target_sma;

    #clamp states and terminate if we exceed bounds
    x_p = clamp( x_p, -env.params.position_extrema, env.params.position_extrema );
    y_p = clamp( y_p, -env.params.position_extrema, env.params.position_extrema );
    vx_p = clamp( vx_p, -env.params.velocity_extrema, env.params.velocity_extrema );
    vy_p = clamp( vy_p, -env.params.velocity_extrema, env.params.velocity_extrema );

    #set new state vector in env
    env.state[1] = x_p;
    env.state[2] = y_p;
    env.state[3] = vx_p;
    env.state[4] = vy_p;
    env.state[5] = mu_p; #mu unchanged
    env.state[6] = target_sma_p; #target unchanged


    #terminate if max steps exceeded or position/vel exceeds bounds
    if ( env.t >= env.params.max_steps )
        env.done = true;
    elseif ( abs(x_p) >= env.params.position_extrema || abs(y_p >= env.params.position_extrema ) 
        || abs(vx_p) >= env.params.velocity_extrema || abs(vy_p >= env.params.velocity_extrema ) )
        env.done = true;
    end

    #return nothing
    nothing

end


function reward_test()

    #unpack state vector
    x = 0.0;
    y = 7000.0;
    vx = sqrt(4903.0/7000);
    vy = 0.0;
    mu = 4903.0;
    target_sma = 14 * 1740.0;

    central_body = Celestial_Body( "Moon", [0.0,0.0], [0.0,0.0], mu, 1740 );
    spacecraft = Spacecraft( "SC1", 0.0, [x,y], [vx,vy], 1000.0, 999.0, 220.0, [0.0, 0.0], 600.0 );

    a, e, ω, θ = Calculate_Planar_OE( spacecraft, central_body );

    r_p        = a * ( 1 - e );
    pos_mag    = norm(spacecraft.position);           

    reward = 0.0;

    if ( pos_mag < central_body.r || r_p < central_body.r )
        flag_impact = true;
        flag_terminal = true;
        reward = - 100.0;
    end

    #determine percent diff in A
    sma_diff            = a - target_sma;
    reward              = reward + exp( - sma_diff^2 / (17000)^2 );

    println("Reward test function");
    println("a: " * string(a) );
    println("e: " * string(e) );
    println("w: " * string(ω) );
    println("θ: " * string(θ) );
    println("r_p: " * string(r_p) );
    println("reward: " * string(reward) );

    return reward;

end

#reward_test();

#declare Hohmann Transfer Environment object
env = Hohmann_Transfer_Env()

#check if the environment is runnable
RLBase.test_runnable!( env )

#=
s = [7119.325609474477, 6692.095629695555, 1.6608655292679837, -0.13627812104494702, 4903.0, 24360.0]

S_intervals = (-1.0e6 .. 1.0e6, -1.0e6 .. 1.0e6, -100.0 .. 100.0, -100.0 .. 100.0, 0.0 .. 1.0e6, 0.0 .. 1.0e6)

interval_1 = -1.0e6 .. 1.0e6
interval_2 = -1.0e6 .. 1.0e6
interval_3 = -100.0 .. 100.0
interval_4 = -100.0 .. 100.0
interval_5 = 0.0 .. 1.0e6
interval_6 = -1.0e6 .. 1.0e6


#S_product = (-1.0e6 .. 1.0e6) × (-1.0e6 .. 1.0e6)
S_product = ( ( ( ( ( interval_1 × interval_2 ) × interval_3 ) × interval_4 ) × interval_5 ) × interval_6 )

display(S_product)


flag = s in S_intervals
display( flag )
flag = s in S_product
display( flag )
=#