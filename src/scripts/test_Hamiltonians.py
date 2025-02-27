
import numpy as np;
import gymnasium as gym;
import matplotlib.pyplot as plot;
import sys
import os

from gymnasium import envs
from gymnasium.envs.registration import register
from stable_baselines3 import A2C

# Adding python src code directory
sys.path.append(os.path.abspath("../python"))

from Ephemeris import Ephemeris
from Hamiltonian_Control import Hamiltonian_Controller_TBT


#register the environment if it isn't registered
if ( ("TwoBody_Orb2Orb_Transfer_Env-v0" in envs.registry.keys()) == False ):
    
    register(
        id="TwoBody_Orb2Orb_Transfer_Env-v0",
        entry_point="TwoBody_Orb2Orb_Transfer_Env:TwoBody_Orb2Orb_Transfer_Env",
    )
    


#initialize the environment
env = gym.make("TwoBody_Orb2Orb_Transfer_Env-v0");


steps_per_traj = 1000;
num_traj = 1;





def test_Hamiltonian_Solution( env, num_trajectories, num_steps_per_traj ):
    
    count_traj = 0;
    arr_episodes = np.array([]);
    arr_reward_totals = np.array([]);
    total_steps_in_env = 0;

    for count_traj in range(0,num_traj):
        
        #reset the environment
        et = 0.0;
        steps = 0;
        r_tot = 0.0;
    
        eph = Ephemeris( );
        init_observation, init_info = env.reset();
        
        #compute Hamiltonian Solution
        H_controller = Hamiltonian_Controller_TBT(env, init_observation, init_info);
    
    
        while (steps < steps_per_traj):
            
            #Sample randomly from the action space. Since the action is a delta-V
            #magnitude in km/s, and the action space is unbounded (-inf to inf) the
            #test maneuver that is returned will be sampled from a Gaussian normal
            #distribution with a mean of 0 and a standard deviation of 1. We 
            #devide by 1000 in this test case to give relatively small maneuvers.
            action = env.action_space.sample();
            
            observation, reward, terminated, truncated, info = env.step(action);
    
            r_tot = r_tot + reward;
    
            elapsed_time = info['Elapsed time'];
            a = info['a'];
            e = info['e'];
            
            if ( terminated == True ):
                break;
            
            eph.add_polar_data(elapsed_time, observation[0], observation[1], observation[2], observation[3] );
            
            #print( elapsed_time, a, e, reward );
            steps = steps + 1;
            
                
        arr_episodes = np.append(arr_episodes, count_traj);
        arr_reward_totals = np.append(arr_reward_totals, r_tot);
        
        total_steps_in_env = total_steps_in_env + steps;
            
        print("Episode count: " + str(count_traj+1) + " of " + str(num_traj) + "   Total steps: " + str(steps) );
        print("Total steps in environment: " + str(total_steps_in_env) );
        
        if (count_traj == num_traj-1):
            print("Plotting last trajectory...");
            fig = eph.plot_xy(info["planet_radii"]);
            plot.show(fig);
            
     
    fig_reward, ax = plot.subplots(figsize=(6, 6));
    
    ax.plot( arr_episodes, arr_reward_totals, label="Total Reward" );
    
    # Customize the figure
    ax.set_title("Total Reward Per Episode");
    ax.set_xlabel("Episode Count");
    ax.set_ylabel("Total R");
    ax.legend();
    ax.grid(False);
    plot.show(fig_reward);
    
            
    print("Test successful");


test_Hamiltonian_Solution(env, num_traj, steps_per_traj);