
import numpy as np
import gymnasium as gym 

from typing import Optional
from scipy.integrate import solve_ivp

from Spacecraft import Spacecraft
        

class HohmannTransferEnv(gym.Env):
    
    def __init__(self):
        
        #define limits of the state parameters
        low_array = np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf], dtype = np.float32 )
        high_array = np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf], dtype = np.float32 )
        
        #define the state space (in this case the observation is the state)
        self.observation_space = gym.spaces.Box( low = low_array, high = high_array )
        
        #internal state of the environment
        self._state = np.array([0,0,0,0,0,0], dtype = np.float32 )
        
        #spacecraft object
        self._spacecraft = Spacecraft()
        
        #current keplerian elements
        self._keplerian_elements = np.array([0,0,0,0,0,0], dtype = np.float32 )
        
        # list of environment parameters
        self.arr_mu = np.array([4903.0])
        self.planet_radii = np.array([1740.0])
        self.elapsed_t = 0.0
        self.step_size = 60.0
        
        
        #define the action space
        low_array_action = np.array([-np.inf], dtype = np.float32 )
        high_array_action = np.array([np.inf], dtype = np.float32 )
        self.action_space = gym.spaces.Box( low = low_array_action, high = high_array_action )
        
        #end def __init__(self):
            
    def _get_info(self, ode_solution, delta_r ):
        
        #to-do: add orbital elements as optional and append to output dictionary
        
        return {
            "Elapsed time":self.elapsed_t,
            "ODE Solution":ode_solution,
            "delta_state":delta_r,
            "planet_radii":self.planet_radii,
            "a":self._keplerian_elements[0],
            "e":self._keplerian_elements[1],
            "w":np.rad2deg( self._keplerian_elements[2] ),
            "theta":np.rad2deg( self._keplerian_elements[3] ),
            }
            
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        #set the initial parameters
        x = 0.0
        y = 7000.0
        vx = (4903/y) ** 0.5
        vy = 0.0
        mu = 4903.0
        mass = 1000.0
        sma_target = 14*1740
        
        #set the location of the central body
        x_cb = 0.0
        y_cb = 0.0
        vx_cb = 0.0
        vy_cb = 0.0
        
        #set the initial state of the environment
        self._state = np.array( [x,y,vx,vy,mu,sma_target], dtype = np.float32 )
        
        #set the location of the central body
        self._arr_cb = np.array( [x_cb, y_cb, vx_cb, vy_cb], dtype = np.float32 )
        
        #Initialize a spacecraft object with the state of the environment
        sc = Spacecraft( x, y, vx, vy, mass )
        
        #Update the spacecraft in the environment
        self._spacecraft = sc
        
        #calculate orbital elements
        a, e, w, theta = sc.calc_Planar_OE( x_cb, y_cb, vx_cb, vy_cb, mu )
        
        self._keplerian_elements[0] = a
        self._keplerian_elements[1] = e
        self._keplerian_elements[2] = w
        self._keplerian_elements[3] = theta
        
        observation = self._state
        info = self._get_info(None,None,)
        
        return observation, info
      
    def calc_reward(self):
        
        #determine reward based on state input, also check if state is terminal
        
        #unpack state vector
        x = self._state[0]
        y = self._state[1]
        sma_target = self._state[5]
        
        a = self._keplerian_elements[0]

        #central body parameters
        cb_rad = self.planet_radii[0]
        
        r = ( x**2 + y**2 )**0.5
        
        terminated = False
        
        #Check if a collision has taken place, terminate with negative reward 
        #Otherwise, compute a reward based on distance from target SMA
        if ( r < cb_rad ):
            reward = -100
            terminated = True
            #end if ( r < cb_rad ):
        else:
            #exponential decaying reward based on the difference between target
            #SMA and current SMA 
            sma_diff = a - sma_target
            reward = np.exp( -sma_diff**2 / (17000)**2 )
            
        
        return reward, terminated
            
    def _apply_dV_in_VNB_frame(self, dV, X_i, Y_i, VX_i, VY_i):
        
        #determine the vel magnitude
        v_norm = np.sqrt( VX_i**2 + VY_i**2 )
        
        #calculate the current dV vector
        v_vec = np.array( [ VX_i, VY_i ] ) / v_norm
        
        #Multiply the delta-V magnitude by the velocity unit vector
        dV_vec = dV * v_vec
        
        return dV_vec
    
    def step(self, action):
        
        #unpack the current state vector
        x = self._state[0]
        y = self._state[1]
        vx = self._state[2]
        vy = self._state[3]
        mu = self._state[4]
        
        #central body location
        x_cb = self._arr_cb[0]
        y_cb = self._arr_cb[1]
        vx_cb = self._arr_cb[2]
        vy_cb = self._arr_cb[3]
        
        #get the current spacecraft object container
        sc = self._spacecraft
        
        #action is defined to be delta-V in vel direction
        arr_dV_in_track = self._apply_dV_in_VNB_frame( action, x, y, vx, vy )
        
        vx = vx + arr_dV_in_track[0]
        vy = vy + arr_dV_in_track[1]
        
        #step the spacecraft forward
        t_span = (0.0,self.step_size)
        y0 = np.array( [x, y, vx, vy] )
        params = np.array( [self.arr_mu[0], self.planet_radii[0], x_cb, y_cb], dtype=np.float32 )
        
        #solve ODE
        solution = solve_ivp( sc.spacecraft_EOM_f_2D_2B, t_span, y0, method='RK45', args=(params,) )
        
        #extract the final state vector from ODE solution (last column in y)
        y_final = (solution.y[:,-1]).astype(np.float32)
        
        #change in state vector
        delta_r = y_final - y0
        
        #state vector components
        x = y_final[0]
        y = y_final[1]
        vx = y_final[2]
        vy = y_final[3]
        
        #update the state and elapsed time of the environment
        self.elapsed_t = self.elapsed_t + self.step_size
        self._state[0] = x
        self._state[1] = y
        self._state[2] = vx
        self._state[3] = vy
        #self._state[4]  and self._state[5] are constant
        
        #update the spacecraft object
        sc.x = self._state[0]
        sc.y = self._state[1]
        sc.vx = self._state[2]
        sc.vy = self._state[3]
        
        #update the environment spacecraft object
        self._spacecraft = sc
        
        #calculate the new orbital elements
        a, e, w, theta = sc.calc_Planar_OE( x_cb, y_cb, vx_cb, vy_cb, mu )
        
        self._keplerian_elements[0] = a
        self._keplerian_elements[1] = e
        self._keplerian_elements[2] = w
        self._keplerian_elements[3] = theta
        
        #determine reward and terminated status
        reward, terminated = self.calc_reward()
        
        #the observation is just the state vector
        observation = self._state
        
        #extract other environment information
        info = self._get_info(solution, delta_r)
        
        #set truncated permanently to false since this is handled externally
        truncated = False
        
        return observation, reward, terminated, truncated, info
        
        
        
        
        
        
        
        
        