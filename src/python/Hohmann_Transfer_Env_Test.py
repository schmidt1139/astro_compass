
import numpy as np;
import gymnasium as gym;
import matplotlib.pyplot as plot;

from gymnasium import envs
from gymnasium.envs.registration import register
from Ephemeris import Ephemeris
from stable_baselines3 import A2C

#register the environment if it isn't registered
if ( ("HohmannTransferEnv-v0" in envs.registry.keys()) == False ):
    
    register(
        id="HohmannTransferEnv-v0",
        entry_point="Hohmann_Transfer_Env:HohmannTransferEnv",
    )
    
    #end


#initialize the environment
env = gym.make("HohmannTransferEnv-v0");


steps_per_traj = 10000;
num_traj = 100;





def test_runnable_env( env, num_trajectories, num_steps_per_traj ):
    
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
        observation, info = env.reset();
    
    
        while (steps < steps_per_traj):
            
            #Sample randomly from the action space. Since the action is a delta-V
            #magnitude in km/s, and the action space is unbounded (-inf to inf) the
            #test maneuver that is returned will be sampled from a Gaussian normal
            #distribution with a mean of 0 and a standard deviation of 1. We 
            #devide by 1000 in this test case to give relatively small maneuvers.
            action = env.action_space.sample() / 100;
            
            observation, reward, terminated, truncated, info = env.step(action);
    
            r_tot = r_tot + reward;
    
            elapsed_time = info['Elapsed time'];
            a = info['a'];
            e = info['e'];
            
            if ( terminated == True ):
                break;
            
            eph.add_data(elapsed_time, observation[0], observation[1], observation[2], observation[3] );
            
            #print( elapsed_time, a, e, reward );
            steps = steps + 1;
            
            #end while (steps < steps_per_traj):
                
        arr_episodes = np.append(arr_episodes, count_traj);
        arr_reward_totals = np.append(arr_reward_totals, r_tot);
        
        total_steps_in_env = total_steps_in_env + steps;
            
        print("Episode count: " + str(count_traj+1) + " of " + str(num_traj) + "   Total steps: " + str(steps) );
        print("Total steps in environment: " + str(total_steps_in_env) );
        
        if (count_traj == num_traj-1):
            print("Plotting last trajectory...");
            fig = eph.plot_xy(info["planet_radii"]);
            plot.show(fig);
            
        
        
        #end for count_traj in range(1,num_traj):
     
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
    #end def test_runnable_env( env, num_trajectories, num_steps_per_traj ):


test_runnable_env(env, num_traj, steps_per_traj);