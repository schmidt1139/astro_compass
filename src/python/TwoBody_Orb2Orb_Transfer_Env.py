
import numpy as np
import gymnasium as gym 


from gymnasium import spaces
from typing import Optional
from scipy.integrate import solve_ivp

from Spacecraft import Spacecraft
from Propagation import *

        

class TwoBody_Orb2Orb_Transfer_Env(gym.Env):
    
    def __init__(self):
        
        #define limits of the state parameters
        low_array = np.full(7, -np.inf, dtype=np.float32); #lower bounds for state space
        high_array = np.full(7, np.inf, dtype=np.float32); #upper-bounds for state space
        
        #define the state space (in this case the observation is the state) 5x5
        self.observation_space = gym.spaces.Box( low = low_array, high = high_array );
        
        self._state = np.full(7, 0.0, dtype=np.float32); #initialize state vector
        
        self._keplerian_elements = np.array([0,0,0,0,0,0], dtype = np.float32 );
        
        # list of environment parameters (Sun is the central body)
        self.arr_mu = np.array([1.3e11]);           #solar mu [km^3/s^2]
        self.planet_radii = np.array([6.957e5]);    #solar radius [km]
        self.elapsed_t = 0.0;
        self.step_size = 3600.0;
        
        
        #define the action space
        #The action space consists of two variables:
        #    1) a control throlle input (scaled from 0 to 1)
        #    2) a thrust vector (0 to 2pi)
        low_array_action = np.array( [0.0,0.0], dtype = np.float32 ); 
        high_array_action = np.array( [1.0, 2*np.pi], dtype = np.float32 );
        self.action_space = gym.spaces.Box( low = low_array_action, high = high_array_action );
        

            
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
            "max_thrust":self._spacecraft.max_thrust,
            "ISP": self._spacecraft.specific_impulse
            }
        

            
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        #extract central body gravitational parameter
        mu          = self.arr_mu[0];
        
        #set the initial parameters
        r           = 2.32495e8         #Mars distance
        theta       = 0.0;              #Default initial theta
        r_dot       = 0.0;              #Initial radial velocity
        v_theta     = (mu/r)**0.5;      #Tangential velocity
        mass        = 3366.0;           #Assumed spacecraft total mass
        sma_target  = 149598023;        #Earth SMA
        C1          = 1.33/1000;        #Spacecraft max thrust (in kN)
        C2          = 3872.0;           #Spacecraft specific impulse (s)
        
        #randomly select theta
        theta = self.np_random.uniform(low=0.0, high=1.0) * 2 *np.pi;

        
        #set the location of the central body
        x_cb = 0.0;
        y_cb = 0.0;
        vx_cb = 0.0;
        vy_cb = 0.0;
        
        #set the initial state of the environment
        self._state = np.array( [r,theta,r_dot,v_theta,mass,mu,sma_target], dtype = np.float32 );
        
        #set the location of the central body
        self._arr_cb = np.array( [x_cb, y_cb, vx_cb, vy_cb], dtype = np.float32 );
        
        #Initialize a spacecraft object with the state of the environment
        sc = Spacecraft( r, theta, r_dot, v_theta, mass, C1, C2 );
        
        #Update the spacecraft in the environment
        self._spacecraft = sc;
        
        #calculate orbital elements
        a, e, w, theta = sc.calc_Planar_OE( x_cb, y_cb, vx_cb, vy_cb, mu );
        
        self._keplerian_elements[0] = a;
        self._keplerian_elements[1] = e;
        self._keplerian_elements[2] = w;
        self._keplerian_elements[3] = theta;
        
        observation = self._state;
        info = self._get_info(None,None,);
        
        return observation, info
        

      
    def calc_reward(self):
        
        #determine reward based on state input, also check if state is terminal
        
        #unpack state vector
        x = self._state[0];
        y = self._state[1];
        sma_target = self._state[5];
        
        a = self._keplerian_elements[0];

        
        #central body parameters
        cb_rad = self.planet_radii[0];
        
        r = ( x**2 + y**2 )**0.5;
        
        terminated = False;
        
        #Check if a collision has taken place, terminate with negative reward 
        #Otherwise, compute a reward based on distance from target SMA
        if ( r < cb_rad ):
            reward = -100;
            terminated = True;
            #end if ( r < cb_rad ):
        else:
            #exponential decaying reward based on the difference between target
            #SMA and current SMA 
            sma_diff = a - sma_target;
            reward = np.exp( -sma_diff**2 / (17000)**2 );
            
        
        return reward, terminated;
        

            
    def _apply_dV_in_VNB_frame(self, dV, X_i, Y_i, VX_i, VY_i):
        
        #determine the vel magnitude
        v_norm = np.sqrt( VX_i**2 + VY_i**2 );
        
        #calculate the current dV vector
        v_vec = np.array( [ VX_i, VY_i ] ) / v_norm;
        
        #Multiply the delta-V magnitude by the velocity unit vector
        dV_vec = dV * v_vec;
        
        return dV_vec;
        

    
    def step(self, action):
        
        #unpack the state vector
        r = self._state[0];
        theta = self._state[1];
        r_dot = self._state[2];
        v_theta = self._state[3];
        mass = self._state[4];
        mu = self._state[5];
        sma_target = self._state[6];
        
        #central body location
        x_cb = self._arr_cb[0];
        y_cb = self._arr_cb[1];
        vx_cb = self._arr_cb[2];
        vy_cb = self._arr_cb[3];
        
        #get the current spacecraft object container
        sc = self._spacecraft;
        
        #unpack the action vector
        u = action[0];      #throttle control
        beta = action[1];   #spacecraft attitude
        
        
        #step the spacecraft forward
        t_span = (0.0,self.step_size);
        y0 = np.array( [r, theta, r_dot, v_theta, mass] );
        params = np.array( [self.arr_mu[0], self.planet_radii[0], 
                            sc.max_thrust, sc.specific_impulse, u, beta], dtype=np.float32 );
        
        
        #solve ODE
        solution = solve_ivp( spacecraft_EOM_radial_2D_EB, t_span, y0, method='RK45', args=(params,) );
        
        #extract the final state vector from ODE solution (last column in y)
        y_final = (solution.y[:,-1]).astype(np.float32);
        
        #change in state vector
        delta_r = y_final - y0;
        
        #update the state and elapsed time
        self.elapsed_t  = self.elapsed_t + self.step_size;
        self._state     = np.append(y_final, [mu,sma_target]);
        #self._state[4]  and self._state[5] are constant
        
        #update the spacecraft object
        sc.update_state(*y_final);
        
        #update the environment spacecraft object
        self._spacecraft = sc;
        
        #calculate the new orbital elements
        a, e, w, theta = sc.calc_Planar_OE( x_cb, y_cb, vx_cb, vy_cb, mu );
        
        self._keplerian_elements[0] = a;
        self._keplerian_elements[1] = e;
        self._keplerian_elements[2] = w;
        self._keplerian_elements[3] = theta;
        
        #determine reward and terminated status
        reward, terminated = self.calc_reward();
        
        #the observation is just the state vector
        observation = self._state;
        
        #extract other environment information
        info = self._get_info(solution, delta_r);
        
        #set truncated permanently to false since this is handled externally
        truncated = False;
        
        return observation, reward, terminated, truncated, info;
        

        
        
        
        
        
        
        
        