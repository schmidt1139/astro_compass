
import numpy as np
import gymnasium as gym 


from gymnasium import spaces
from typing import Optional
from scipy.integrate import solve_ivp


def spacecraft_EOM_f_2D_2B( t,y,params ):
    
    '''
    ode propagation function
    -----------------------------------------------------------------------------------
    This is a special case force model function for use in propagating a
    spacecraft. This function assumes that there are only two dimensions
    (x and y) and that there are only two bodies: the spacecraft and the 
    central body.
    
    Inputs
    -----------------------------------------------------------------------------------
    t:      Elapsed time
    y:      Input state vector
    arr_mu: The list of parameters
    
    Outputs
    -----------------------------------------------------------------------------------
    '''
    
    dy = np.zeros(4, dtype = np.float32 );
    params = params.astype(np.float32);
    
    num_params = len(params);
    
    if ( num_params == 4 ):
        mu_cb = params[0];
        radius_cb = params[1];
        x_cb = params[2];
        y_cb = params[3];
    else:
        raise Exception('Invalid number of parameters');
        #end if ( num_params == 3 ):
        
    #unpack the state vector
    x_sc = y[0];
    y_sc = y[1];
    vx_sc = y[2];
    vy_sc = y[3];
    
    #determine relative position
    x_rel = x_sc - x_cb;
    y_rel = y_sc - y_cb;
    
    #relative position magnitude
    r_rel = ( x_rel**2 + y_rel**2 )**(0.5);
    
    #define a flag to track if the spacecraft has collided with the central body
    flag_collision = False;
    
    if ( r_rel < radius_cb ):
        
        flag_collision = True;
        
        dy[0] = 0.0;
        dy[1] = 0.0;
        dy[2] = 0.0;
        dy[3] = 0.0;
        
        return dy.astype(np.float32);
    
    else:
        
        x_hat_rel = x_rel / r_rel;
        y_hat_rel = y_rel / r_rel;
        
        ddx = - mu_cb * x_hat_rel / r_rel**2;
        ddy = - mu_cb * y_hat_rel / r_rel**2;
        
        dy[0] = vx_sc;
        dy[1] = vy_sc;
        dy[2] = ddx;
        dy[3] = ddy;
                
        return dy.astype(np.float32);
        
        #end if ( r_rel < radius_cb ):
        
    
    #end def spacecraft_EOM( x, y, vx, vy, mu ):
        
        
def Calc_Planar_OE(x,y,vx,vy,mu_cb):
    
    #position and velocity magnitudes
    r = ( x**2 + y**2 )**0.5;
    v = ( vx**2 + vy**2 )**0.5;
    
    #spacecraft position, vel, and z vectors
    sc_pos = np.array([ x, y, 0.0 ]);
    sc_vel = np.array([ vx, vy, 0.0 ]);
    z_hat = np.array([ 1.0, 0.0, 0.0 ]);
    r_hat = sc_pos / r;
    
    #angular momentum
    h_vec = np.cross( sc_pos, sc_vel );
    h = np.linalg.norm(h_vec);
    h_hat = h_vec / h;
    
    #node line
    N = np.cross( z_hat, h_hat );
    N_hat = N / np.linalg.norm(N);
    
    #specific energy
    eps = v**2 / 2 + mu_cb/r;
    
    #eccentricity vector
    e_vec = np.cross(sc_vel,h_vec) / mu_cb - sc_pos/r;
    e = np.linalg.norm(e_vec);
    e_hat = e_vec/e;
    
    #semi major axis
    rp = h**2 / mu_cb / ( 1 + e * np.cos(0) );
    ra = h**2 / mu_cb / ( 1 + e * np.cos( np.pi ) );
    a = 1/2 * ( rp + ra );
    
    #argument of periapsis
    if ( e_vec[2] >= 0.0 ):
        w = np.acos( np.dot( N_hat, e_hat ) );
    else:
        w = 2 * np.pi - np.acos( np.dot( N_hat, e_hat ) );
    
    w_deg = np.rad2deg(w);
    
    #true anomaly - extra error handling included,
    #mainly needed for hyperbolic instances
    if ( np.dot( sc_pos, sc_vel ) >= 0.0 ):
        
        dotp = np.dot(e_hat,r_hat);
        
        if (dotp < -1 ):
            dotp = -1;
            
        if ( abs( np.dot(e_hat,r_hat) ) < 1.0 ):
            theta = np.acos( np.dot( e_hat,r_hat ) );
        elif ( np.dot(e_hat,r_hat) < -1.0 ):
            theta = np.pi;
        else:
            theta = 0.0;         
            
    else:
        
        dotp = np.dot(e_hat,r_hat);
        if (dotp < -1 ):
            dotp = -1;
            
        theta = 2 * np.pi - np.acos( np.dot( e_hat, r_hat ) );
        
        #end if ( np.dot( sc_pos, sc_vel ) >= 0.0 ):
    
    theta_deg = np.rad2deg(theta);
    
    print(a)
    print(e)
    print(w_deg)
    print(theta_deg)
    
    return a, e, w, theta;
    
    #end def Calc_Planar_OE():
        

class HohmannTransferEnv(gym.Env):
    
    def __init__(self):
        
        #define limits of the state parameters
        low_array = np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf], dtype = np.float32 );
        high_array = np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf], dtype = np.float32 );
        
        #define the state space (in this case the observation is the state)
        self.observation_space = gym.spaces.Box( low = low_array, high = high_array );
        
        self._state = np.array([0,0,0,0,0,0], dtype = np.float32 );
        
        # list of environment parameters
        self.arr_mu = np.array([4903.0]);
        self.planet_radii = np.array([1740.0]);
        self.elapsed_t = 0.0;
        self.step_size = 60.0;
        
        #define the action space
        low_array_action = np.array([-np.inf], dtype = np.float32 );
        high_array_action = np.array([np.inf], dtype = np.float32 );
        self.action_space = gym.spaces.Box( low = low_array_action, high = high_array_action );
        
        #end def __init__(self):
            
    def _get_info(self, ode_solution, delta_r ):
        
        #to-do: add orbital elements as optional and append to output dictionary
        
        return {
            "Elapsed time":self.elapsed_t,
            "ODE Solution":ode_solution,
            "delta_state":delta_r,
            "planet_radii":self.planet_radii
            }
        
        #end def _get_info(self):
            
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        #set the initial parameters
        x = 0.0;
        y = 7000.0;
        vx = (4903/y) ** 0.5;
        vy = 0.0;
        mu = 4903.0;
        sma_target = 14*1740;
        
        self._state = np.array( [x,y,vx,vy,mu,sma_target], dtype = np.float32 );
        
        observation = self._state;
        info = self._get_info(None,None);
        
        return observation, info
        
        #end def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
      
    def calc_reward(self):
        
        #determine reward based on state input, also check if state is terminal
        
        terminated = False;
        reward = 1.0;
        
        return reward, terminated;
        
        #end def calc_reward(self):
            
    def _apply_dV_in_VNB_frame(self, dV, X_i, Y_i, VX_i, VY_i):
        
        #determine the vel magnitude
        v_norm = np.sqrt( VX_i**2 + VY_i**2 );
        
        #calculate the current dV vector
        v_vec = np.array( [ VX_i, VY_i ] ) / v_norm;
        
        #Multiply the delta-V magnitude by the velocity unit vector
        dV_vec = dV * v_vec;
        
        return dV_vec;
        
        #end def apply_dV_in_VNB_frame(action, X, Y, VX, VY):
    
    def step(self, action):
        
        x = self._state[0];
        y = self._state[1];
        vx = self._state[2];
        vy = self._state[3];
        
        #action is defined to be delta-V in vel direction
        arr_dV_in_track = self._apply_dV_in_VNB_frame( action, x, y, vx, vy );
        
        vx = vx + arr_dV_in_track[0];
        vy = vy + arr_dV_in_track[1];
        
        #step the spacecraft forward
        t_span = (0.0,self.step_size);
        y0 = np.array( [x, y, vx, vy] );
        params = np.array( [self.arr_mu[0], self.planet_radii[0], 0.0, 0.0] );
        
        #solve ODE
        solution = solve_ivp( spacecraft_EOM_f_2D_2B, t_span, y0, method='RK45', args=(params,) );
        
        #extract the final state vector from ODE solution (last column in y)
        y_final = solution.y[:,-1];
        
        #change in state vector
        delta_r = y_final - y0;
        
        #update the state and elapsed time
        self.elapsed_t = self.elapsed_t + self.step_size;
        self._state = y_final;
        
        #state vector
        x = self._state[0];
        y = self._state[1];
        vx = self._state[2];
        vy = self._state[3];
        
        #calculate orbital elements
        a, e, w, theta = Calc_Planar_OE( x, y, vx, vy, self.arr_mu[0] );
        
        #return observation, reward, terminated, truncated, info
        
        #the observation is just the state vector
        observation = self._state;
        
        #extract other environment information
        info = self._get_info(solution, delta_r);
        
        #determine reward and terminated status
        reward, terminated = self.calc_reward();
        
        #set truncated permanently to false since this is handled externally
        truncated = False;
        
        return observation, reward, terminated, truncated, info;
        
        #end def step(self, action):
    
    #end class HohmannTransferEnv(gym.Env):
        
        
        
        
        
        
        
        
        