
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


np.set_printoptions(precision=4);

#reset the evironment using set seed value
seed_env = 1;
count = 1;
count_solutions = 0;
time_start = time.time();
sa_state_costate = [];

while(count<=10):
    
    
    eph = Ephemeris( );
    init_observation, init_info = env.reset(seed=seed_env);
    
    r, theta, v_r, v_theta = init_observation[0:4];
    x, y, vx, vy = polar_to_cartesian( r, theta, v_r, v_theta );
    
    
    print( "Traj count: ", count );
    
    #ephemeris
    eph = Ephemeris();
    
    #extract some parameters of interest
    sun_rad = env.unwrapped.planet_radii[0];
    C1 = init_info["max_thrust"]*1000; #max thrust in N
    C2 = init_info["ISP"]; #spacecraft specific impulse in seconds
    
    #compute Hamiltonian Solution
    input_TOF = 1.1 * 365.25 * 24 * 60 * 60;
    H_controller = Hamiltonian_Controller_TBT(env, init_observation, 
                                              init_info, input_TOF);
    
    #state vector array
    arr_sv = np.array([float(x/H_controller.l_star*1000), float(y/H_controller.l_star*1000),
              float(vx/H_controller.l_star*1000*H_controller.t_star), 
              float(vy/H_controller.l_star*1000*H_controller.t_star), 1.0]);
    
    #compute solution
    contoller_outputs = H_controller.hamiltonian_solution_finder();
    flag_solved = contoller_outputs[0];
    h_sol       = contoller_outputs[1];
    eps         = contoller_outputs[2];
    sol         = contoller_outputs[3];
    log         = contoller_outputs[4];
    
    if ( flag_solved == False ):
        #print the targeter log to the console
        for info in log:
            print(info);
        print("Warning: solver failed");
    else:
        count_solutions = count_solutions + 1;
    
    if ( flag_solved == True ):
                
        #write output ephemeris
        eph_out, arr_time, arr_u, arr_rho, arr_alpha_x, arr_alpha_y = H_controller.generate_output_ephemeris(eph);
        
        fig, ax = plot.subplots(figsize=(6, 6));
        ax.plot( arr_time, eph_out.arr_m );
        ax.set_title("SC Mass: Traj " + str(count) );
        plt.show();
        
        fig, ax = plot.subplots(figsize=(6, 6));
        ax.plot( arr_time, arr_rho );
        ax.set_title("Switching Function: Traj " + str(count) );
        plt.show();
        
        fig, ax = plot.subplots(figsize=(6, 6));
        ax.plot( arr_time, arr_u );
        ax.set_title("Spacecraft Thrust Throttle over Time: Traj " + str(count) );
        plt.show();
        
        fig, ax = plot.subplots(figsize=(6, 6));
        ax.plot( arr_time, arr_alpha_x, label="alpha_x" );
        ax.plot( arr_time, arr_alpha_y, label="alpha_y" );
        ax.set_title("Alpha Vector (Maneuver Direction) over Time: Traj " + str(count) ); 
        ax.legend();  
        plt.show();
        
        #Ephemeris plotting
        sun_rad = 6.957e8;
        sma_Earth   = 149598023 * 1000;             #m
        sma_Mars    = 2.32495e8 * 1000;             #m
        eph_out.plot_xy(sun_rad);
        eph_out.plot_xy_ref_orbit(sma_Earth, "Earth Orbit" );
        eph_out.plot_xy_ref_orbit(sma_Mars, "Mars Orbit" );
        plt.title("Trajectory: Traj " + str(count)  );
        plt.show();
        
    #the sys time at the end of this iteration
    time_end = time.time();
    elapsed_time = time_end - time_start;
    elapsed_time_per_i = elapsed_time / count;
        
    print("Solution for initial co-states: ", h_sol);
    print("Elapsed time: ", round(elapsed_time) );
    print("Convergence rate: ", count_solutions/count );
    print("");
    
    count = count + 1;
    seed_env = seed_env + 1;
    
    
print("\n\n");
print("Convergence rate: ", count_solutions/(count-1) );
print("Elapsed time per iteration (s): ", elapsed_time_per_i );
