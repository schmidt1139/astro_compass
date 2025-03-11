
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
    
    arr_time    = sol.t;
    variables   = sol.y;


test_Hamiltonian_Solution(env, num_traj, steps_per_traj);