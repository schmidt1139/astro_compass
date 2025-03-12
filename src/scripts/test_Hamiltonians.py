
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


def test_Hamiltonian_Solution( env, num_trajectories, num_steps_per_traj ):
    
    eph = Ephemeris( );
    init_observation, init_info = env.reset();
    sun_rad = env.unwrapped.planet_radii[0];
    C1 = init_info["max_thrust"]*1000; #max thrust in N
    C2 = init_info["ISP"];
    
    #compute Hamiltonian Solution
    H_controller = Hamiltonian_Controller_TBT(env, init_observation, 
                                              init_info, input_TOF);
    
    
    sol = H_controller.hamiltonian_solution_finder();
    
    l_star = H_controller.sma_Earth;
    t_star = H_controller.t_star;
    m_star = H_controller.m_star;
    
    arr_time    = sol.t;
    variables   = sol.y;
    
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
        
    
    sun_rad = 6.957e5;
    eph.plot_xy(sun_rad);
    eph.plot_xy_ref_orbit(l_star, "Earth Orbit" );
    eph.plot_xy_ref_orbit(2.32495e8, "Mars Orbit" );


test_Hamiltonian_Solution(env, num_traj, steps_per_traj);