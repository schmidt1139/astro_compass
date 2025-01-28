
import numpy as np;
import gymnasium as gym;
import matplotlib.pyplot as plot;

from gymnasium import envs
from gymnasium.envs.registration import register
from Ephemeris import Ephemeris

#register the environment if it isn't registered
if ( ("HohmannTransferEnv-v0" in envs.registry.keys()) == False ):
    
    register(
        id="HohmannTransferEnv-v0",
        entry_point="Hohmann_Transfer_Env:HohmannTransferEnv",
    )
    
    #end




#initialize the environment
env = gym.make("HohmannTransferEnv-v0");

observation, info = env.reset();
steps = 0;
et = 0.0;

eph = Ephemeris( info["planet_radii"] );

while (steps < 1000):
    
    action = 0.01;
    
    observation, reward, terminated, truncated, info = env.step(action);
    elapsed_time = info['Elapsed time'];
    a = info['a'];
    e = info['e'];
    
    eph.add_data(elapsed_time, observation[0], observation[1], observation[2], observation[3] );
    
    #print( elapsed_time, a, e, reward );
    steps = steps + 1;
    

fig = eph.plot_xy();

plot.show(fig);

