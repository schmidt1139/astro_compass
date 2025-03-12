
import numpy as np;
import gymnasium as gym;
import matplotlib.pyplot as plot;
import matplotlib.ticker as ticker
import sys
import os

from gymnasium import envs
from gymnasium.envs.registration import register
from stable_baselines3 import A2C

# Adding python src code directory
sys.path.append(os.path.abspath("../python"))

from Ephemeris import Ephemeris
from Hamiltonian_Control import Hamiltonian_Controller_TBT
from scipy.integrate import solve_ivp
from Propagation import Hamiltonian_EOM_TBT_nd


#register the environment if it isn't registered
if ( ("TwoBody_Orb2Orb_Transfer_Env-v0" in envs.registry.keys()) == False ):
    
    register(
        id="TwoBody_Orb2Orb_Transfer_Env-v0",
        entry_point="TwoBody_Orb2Orb_Transfer_Env:TwoBody_Orb2Orb_Transfer_Env",
    )
    


#initialize the environment
env = gym.make("TwoBody_Orb2Orb_Transfer_Env-v0");


num_traj = 1;

#The prescribed time of flight for the transfer trajectory [s]
input_TOF = 1.1 * 365.25 * 24 * 60 * 60;
steps_per_traj = np.ceil( input_TOF / env.unwrapped.step_size );

np.set_printoptions(precision=3)  # Limit to 3 decimal places


def test_non_dim_propagation( env, num_trajectories, num_steps_per_traj ):
    
    eph = Ephemeris( );
    init_observation, init_info = env.reset();
    sun_rad = env.unwrapped.planet_radii[0];
    C1 = init_info["max_thrust"]*1000; #max thrust in N
    C2 = init_info["ISP"];
    
    #compute Hamiltonian Solution
    H_controller = Hamiltonian_Controller_TBT(env, init_observation, 
                                              init_info, input_TOF);
    
    mu = env.unwrapped.arr_mu[0] * 1000**3;
    
    #set up parameter array
    params = np.array( [mu, C1, C2 ], dtype=np.float32 );
    
    #initial state vector
    r_0         = init_observation[0]*1000;     #m
    theta_0     = init_observation[1];          #rad
    r_dot_0     = init_observation[2]*1000;     #m/s
    v_theta_0   = init_observation[3]*1000;     #m/s
    m_0         = init_observation[4];          #kg
    r_f         = init_observation[6]*1000;     #m
    r_dot_f     = 0.0;                          #m/s
    v_theta_f   = (mu/r_f) ** 0.5;              #m/s
    
    #constants/non-dim factors
    sma_Earth = 149598023 * 1000;               #m
    sma_Mars  = 2.32495e8 * 1000;               #m
    g0        = 9.80665;                        #m/s^2
    l_star    = sma_Earth;
    m_star    = m_0;
    t_star      = ( sma_Earth**3 / mu ) ** (0.5);
    
    #non-dim state vector
    r_0_nd = r_0 / l_star;
    theta_0_nd = theta_0;
    r_dot_0_nd = r_dot_0 / l_star * t_star;
    v_theta_0_nd = v_theta_0 / l_star * t_star;
    m_0_nd = m_0 / m_star;
    
    #non-dim TOF
    input_TOF_nd = input_TOF / t_star;
    
    #non-dim parameters
    mu_nd = 1; #by definition of t_star
    C1_nd = C1 * ( 1 / m_star ) * ( 1 / l_star ) * t_star**2;
    ve = C2 * g0;
    ve_nd = C2 * g0 * t_star / l_star;
    
    #define time span
    t_span = (0,input_TOF_nd);
    t_eval = np.linspace(*t_span, 1000);
    
    #set up parameter array
    params = np.array( [mu_nd, C1_nd, ve_nd ], dtype=np.float32 );
    
    #initial nd state vector
    arr_y0 =  np.array( [r_0_nd, theta_0_nd, r_dot_0_nd, v_theta_0_nd, m_0_nd ], dtype=np.float32 );
    
    lam_guess = np.array( [0.01, 0.001, 0.001, 0.001, -0.01 ], dtype=np.float32 );
    
    #construct full state vector at t=0
    arr_full_y0 = np.hstack( (arr_y0, lam_guess) );
    
    #integrate forward in time
    sol = solve_ivp(Hamiltonian_EOM_TBT_nd, t_span, arr_full_y0, method='RK45', args=(params,), t_eval=t_eval );
    
    
    if ( sol.status == -1 ):
        print(sol.message);
        raise Exception("Integration failed");
    
    arr_time    = sol.t;
    variables   = sol.y;
    arr_r_nd    = variables[0];
    arr_theta_nd = variables[1];
    arr_r_dot_nd = variables[2];
    arr_v_theta_nd = variables[3];
    arr_m_nd = variables[4];
    arr_lam_r_nd = variables[5];
    arr_lam_theta_nd = variables[6];
    arr_lam_r_dot_nd = variables[7];
    arr_lam_v_theta_nd = variables[8];
    arr_lam_m_nd = variables[9];
    
    #empty plotting structs
    arr_beta = [];
    arr_u = [];
    
    #ephemeris
    eph = Ephemeris();
    
    for index, t in enumerate(arr_time):
        
        r_nd            = variables[0,index];
        theta_nd        = variables[1,index];
        r_dot_nd        = variables[2,index];
        v_theta_nd      = variables[3,index];
        m_nd            = variables[4,index];
        
        lam_r_nd           = variables[5,index];
        lam_theta_nd       = variables[6,index];
        lam_r_dot_nd       = variables[7,index];
        lam_v_theta_nd     = variables[8,index];
        lam_m_nd           = variables[9,index];
        
        beta        = np.atan(lam_r_nd/lam_v_theta_nd);
        u           = - lam_m_nd * C1 / C2;
        u           = u + C1 * ( lam_r_nd * np.sin(beta) + lam_v_theta_nd * np.cos(beta) ) / m_nd;
        u_unclipped = u;
        u           = np.clip(u, 0, 1);
        
        #append data to arrays
        arr_u.append(u);
        arr_beta.append(beta);
        
        #add ephemeris data
        r_d = r_nd * l_star;
        r_dot_d = r_dot_nd * l_star / t_star;
        v_theta_d = v_theta_nd * l_star / t_star;
        m_d = m_nd * m_star;
        
        eph.add_polar_data( t, r_d, theta_nd, r_dot_d, v_theta_d );
        
        
    fig, ax = plot.subplots(figsize=(6, 6));
    ax.plot( arr_time, arr_u );
    ax.set_title("U over time");
    
    fig, ax = plot.subplots(figsize=(6, 6));
    ax.plot( arr_time, arr_beta );
    ax.set_title("Beta over time");    
    
    fig, ax = plot.subplots(figsize=(6, 6));
    ax.plot( arr_time, arr_r_nd );
    ax.set_title("r over time");   
    
    fig, ax = plot.subplots(figsize=(6, 6));
    ax.plot( arr_time, arr_theta_nd );
    ax.set_title("Theta over time");   
    
    fig, ax = plot.subplots(figsize=(6, 6));
    ax.plot( arr_time, arr_lam_r_nd, label="lam r" );
    ax.plot( arr_time, arr_lam_theta_nd, label="lam theta" );
    ax.plot( arr_time, arr_lam_r_dot_nd, label="lam r dot" );
    ax.plot( arr_time, arr_lam_v_theta_nd, label="lam v theta" );
    ax.plot( arr_time, arr_lam_m_nd, label="lam m" );
    ax.set_title("Co-States over time");   
    ax.legend();
    
    fig, ax = plot.subplots(figsize=(6, 6));
    ax.plot( arr_time, arr_m_nd );
    ax.set_title("Mass Fraction over time");  
    
    sun_rad = 6.957e5*1000;
    eph.plot_xy(sun_rad);
    eph.plot_xy_ref_orbit(sma_Earth, "Earth Orbit" );
    eph.plot_xy_ref_orbit(sma_Mars, "Mars Orbit" );
        
    
    print(f"TOF nd: {input_TOF_nd}");
    print(f"R0 nd: {r_0_nd}");
    print(f"Theta0 nd: {theta_0}");
    print(f"r_dot_0 nd: {r_dot_0_nd}");
    print(f"v_theta_0 nd: {v_theta_0_nd}");
    print(f"m_0 nd: {m_0_nd}");
    print("");
    print("Parameters (nd)");
    print(f"C1_nd nd: {C1_nd}");
    print(f"C2: {C2}");
    print(f"v_e: {ve}");
    print(f"ve_nd: {ve_nd}");
    

test_non_dim_propagation(env, num_traj, steps_per_traj);