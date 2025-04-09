
import gymnasium as gym;
from gymnasium import envs;
from gymnasium.envs.registration import register;
import sys;
import os;
import time;
import csv;
import matplotlib.pyplot as plt


# Adding python src code directory
sys.path.append(os.path.abspath("../python"))


from Ephemeris import Ephemeris
from StateVectorUtilities import *
from Hamiltonian_Control import *


#register the environment if it isn't registered
if ( ("TwoBody_Orb2Orb_Transfer_Env-v0" in envs.registry.keys()) == False ):
    
    register(
        id="TwoBody_Orb2Orb_Transfer_Env-v0",
        entry_point="TwoBody_Orb2Orb_Transfer_Env:TwoBody_Orb2Orb_Transfer_Env",
    )
    
# Adding python src code directory
sys.path.append(os.path.abspath("../python"));

#initialize the environment
env = gym.make("TwoBody_Orb2Orb_Transfer_Env-v0");








def test_random_env_rest( env ):

    #reset the evironment using set seed value
    seed_env = 1;
    count = 1;
    
    while(count<10):
        
        eph = Ephemeris( );
        init_observation, init_info = env.reset(seed=seed_env);
        
        r, theta, v_r, v_theta = init_observation[0:4];
        x, y, vx, vy = polar_to_cartesian( r, theta, v_r, v_theta );
        print( "Traj count: ", count );
        print("r0: ", r );
        print("theta0: ", theta );
        print("v_r0: ", v_r );
        print("v_theta0: ", v_theta );
        
        #ephemeris
        eph = Ephemeris();
        
        #extract some parameters of interest
        sun_rad = env.unwrapped.planet_radii[0];
        C1 = init_info["max_thrust"]*1000; #max thrust in N
        C2 = init_info["ISP"]; #spacecraft specific impulse in seconds
        
        #compute Hamiltonian Solution
        input_TOF = 1.1 * 365.25 * 24 * 60 * 60;
        input_TOF = input_TOF * 1.2;
        H_controller = Hamiltonian_Controller_TBT(env, init_observation, 
                                                  init_info, input_TOF);
        
        
        #compute solution
        flag_solved, h_sol, eps, sol,log = H_controller.hamiltonian_solution_finder();
        
        #write output ephemeris
        eph_out, arr_time, arr_u, arr_rho, arr_alpha_x, arr_alpha_y = H_controller.generate_output_ephemeris(eph);
        
        fig, ax = plot.subplots(figsize=(6, 6));
        ax.plot( arr_time, arr_rho );
        ax.set_title("Switching Function");
        plt.show();
        
        fig, ax = plot.subplots(figsize=(6, 6));
        ax.plot( arr_time, arr_u );
        ax.set_title("Spacecraft Thrust Throttle over Time");
        plt.show();
        
        fig, ax = plot.subplots(figsize=(6, 6));
        ax.plot( arr_time, arr_alpha_x, label="alpha_x" );
        ax.plot( arr_time, arr_alpha_y, label="alpha_y" );
        ax.set_title("Alpha Vector (Maneuver Direction) over Time"); 
        ax.legend();  
        plt.show();
        
        #Ephemeris plotting
        sun_rad = 6.957e8;
        sma_Earth   = 149598023 * 1000;             #m
        sma_Mars    = 2.32495e8 * 1000;             #m
        eph_out.plot_xy(sun_rad);
        eph_out.plot_xy_ref_orbit(sma_Earth, "Earth Orbit" );
        eph_out.plot_xy_ref_orbit(sma_Mars, "Mars Orbit" );
        plt.show();
        
        
        np.set_printoptions(precision=16)
        print("Solution for initial co-states: ", h_sol);
        print("Final smoothing parameter used in solution generation: ", eps);
        np.set_printoptions(precision=3)
        
        count = count + 1;
        seed_env = seed_env + 1;
        
        
test_random_env_rest(env);