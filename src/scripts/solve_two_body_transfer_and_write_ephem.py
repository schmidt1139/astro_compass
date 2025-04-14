
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

#USER INPUTS ------------------------------------------------------------------

#Initialize dictionary with input parameters to function
args = {
    "TOF": 1.1 * 365.25 * 24 * 60 * 60, #s
    "r0": 2.32495e+08, #Initial state/radius [km]
    "theta0": 0.0, #Initial state/true anomaly [rad]
    "vr0": 0.0, #Initial state/radial velocity [km/s]
    "vtheta0": 23.6464, #Initial state/transpose velocity [km/s]
    "m0": 3366.0, #Initial state/mass [kg]
    "mu": 1.3e+11, #Central body gravitational parameter [km^3/s^2]
    "sma_target": 1.49598e+08, #Target sma of final circular orbit [km]
    "max_thrust": 1.33, #Max thrust of the spacecraft engine [N]
    "ISP": 3872.0 #Specific impulse of the spacecraft engine [s]
}

#Ephemeris filename
dir_ephemeris_out = "..\\..\\data\\training_ephems\\"
filename_ephemeris_out = dir_ephemeris_out + "solution_ephemeris.txt";
#END USER INPUTS --------------------------------------------------------------

#register the environment if it isn't registered
if ( ("TwoBody_Orb2Orb_Transfer_Env-v0" in envs.registry.keys()) == False ):
    
    register(
        id="TwoBody_Orb2Orb_Transfer_Env-v0",
        entry_point="TwoBody_Orb2Orb_Transfer_Env:TwoBody_Orb2Orb_Transfer_Env",
    )
    

def solve_two_body_transfer_and_write_ephem(env, filename_eph, args ):
    
    num_traj = 1;

    #The prescribed time of flight for the transfer trajectory [s]
    input_TOF = args["TOF"];
    steps_per_traj = np.ceil( input_TOF / env.unwrapped.step_size );
    
    np.set_printoptions(precision=3)  # Limit to 3 decimal places
    
    #reset the TBT env
    seed = 42;
    init_observation, init_info = env.reset(seed=seed);
    
    #override init observation to input arguments
    init_observation[0] = args["r0"];
    init_observation[1] = args["theta0"];
    init_observation[2] = args["vr0"];
    init_observation[3] = args["vtheta0"];
    init_observation[4] = args["m0"];
    init_observation[5] = args["mu"];
    init_observation[6] = args["sma_target"];
    init_info["max_thrust"] = args["max_thrust"]/1000;
    init_info["ISP"] = args["ISP"];
    
    #ephemeris
    eph = Ephemeris();
    
    print(init_observation);
    
    #extract some parameters of interest
    sun_rad     = env.unwrapped.planet_radii[0];
    C1          = args["max_thrust"];
    C2          = args["ISP"];

    #compute Hamiltonian Solution
    H_controller = Hamiltonian_Controller_TBT(env, init_observation, 
                                              init_info, input_TOF);
    
    
    #compute solution
    flag_solved, h_sol, eps, sol, log = H_controller.hamiltonian_solution_finder();
    
    #write output ephemeris
    eph_out, arr_time, arr_u, arr_rho, arr_alpha_x, arr_alpha_y = H_controller.generate_output_ephemeris(eph);
    
    #plotting
    fig, ax = plot.subplots(figsize=(6, 6));
    ax.plot( arr_time, arr_rho );
    ax.set_title("Switching Function");
    
    fig, ax = plot.subplots(figsize=(6, 6));
    ax.plot( arr_time, arr_u );
    ax.set_title("Spacecraft Thrust Throttle over Time");
    
    fig, ax = plot.subplots(figsize=(6, 6));
    ax.plot( arr_time, arr_alpha_x, label="alpha_x" );
    ax.plot( arr_time, arr_alpha_y, label="alpha_y" );
    ax.set_title("Alpha Vector (Maneuver Direction) over Time"); 
    ax.legend();  
    
    fig, ax = plot.subplots(figsize=(6, 6));
    ax.plot( arr_time, sol.y[4,:] );
    ax.set_title("Spacecraft Mass Fraction over Time"); 
    
    #Ephemeris plotting
    sun_rad = 6.957e8;
    eph_out.plot_xy(sun_rad);
    eph_out.plot_xy_ref_orbit( args["sma_target"]*1000, "Target Orbit" );
    
    np.set_printoptions(precision=16)
    print("Solution for initial co-states: ", h_sol);
    print("Final smoothing parameter used in solution generation: ", eps);
    
    #write ephemeris file
    eph_out.write_to_file(filename_ephemeris_out, mod_vector_write_frequency=10 );
    print("Ephemeris of trajectory written to: ", filename_ephemeris_out );


#initialize the environment
env = gym.make("TwoBody_Orb2Orb_Transfer_Env-v0");

solve_two_body_transfer_and_write_ephem(env, filename_ephemeris_out, args );