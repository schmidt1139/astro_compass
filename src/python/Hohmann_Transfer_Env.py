
import numpy as np
import gymnasium as gym 

from gymnasium import spaces

class HohmannTransferEnv(gym.Env):
    
    def __init__(self):
        
        #define limits of the state parameters
        low_array = np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]);
        high_array = np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]);
        
        #define the state space (in this case the observation is the state)
        self.observation_space = gym.spaces.Box( low = low_array, high = high_array );
        
        
        
        #end def __init__(self):
    
    #end class HohmannTransferEnv(gym.Env):