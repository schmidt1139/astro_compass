from TwoBody_Orb2Orb_Transfer_Env import *
from Propagation import Hamiltonian_EOM_TBT
from scipy.optimize import root
from scipy.optimize import fsolve
import numpy as np;

class Hamiltonian_Controller_TBT:
    
    def extract_env_boundary_conditions(self):
        
        #extract initial conditions
        r_0         = self.init_observation[0];
        theta_0     = self.init_observation[1];
        r_dot_0     = self.init_observation[2];
        v_theta_0   = self.init_observation[3];
        m_0         = self.init_observation[4];
        mu          = self.init_observation[5];
        r_f         = self.init_observation[6];
        r_dot_f     = 0.0;
        v_theta_f   = (mu/r_f) ** 0.5;
        
        #constants
        self.sma_Earth      = 149598023; #Earth SMA in km
        self.mu             = mu; #gravitational parameter in km^3/s^2
        self.t_star         = ( self.sma_Earth**3 / self.mu ) ** 0.5; #non-dimensional time in s
        self.m_star         = m_0; #kg
        g0                  = 9.80665 / 1000; #km/s^2
        
        #establish non-dimensionalization factors
        self.position_nd_f  = 1 / self.sma_Earth;
        self.velocity_nd_f  = self.t_star / self.sma_Earth;
        self.mass_nd_f      = 1 / self.m_star;
        self.time_nd_f      = 1 / self.t_star;
        
        #Non-dimensionalize initial states
        self.r_0_nd         = r_0 * self.position_nd_f;
        self.theta_0_nd     = theta_0;
        self.r_dot_0_nd     = r_dot_0 * self.velocity_nd_f;
        self.v_theta_0_nd   = v_theta_0 * self.velocity_nd_f;
        self.m_0_nd         = m_0 * self.mass_nd_f;
        
        #Non-dimensionalize final state constraints
        self.r_f_nd         = r_f * self.position_nd_f;
        self.r_dot_f_nd     = r_dot_f * self.velocity_nd_f;
        self.v_theta_f_nd   = v_theta_f * self.velocity_nd_f;
        
        #Non-Dim TOF
        self.input_TOF_nd   = self.input_TOF * self.time_nd_f;
        
        #Parameters
        C1 = self.init_info["max_thrust"] * 1000; #max thrust in N
        C2 = self.init_info["ISP"]; #specific impulse of thruster in seconds
        
        #Non-Dimensionalize Parameters
        self.C1_nd = C1 * ( 1 / self.m_star ) * ( 1 / self.sma_Earth ) * self.t_star**2;
        self.C2_nd = C2 * g0 * self.velocity_nd_f;
        self.mu_nd = 1.0;
        
        #spacecraft initial state vectors (raw and scaled)
        self.arr_y0         = np.array([r_0, theta_0, r_dot_0, v_theta_0, m_0]);
        self.arr_y0_nd      = np.array([self.r_0_nd, self.theta_0_nd, self.r_dot_0_nd, self.v_theta_0_nd, self.m_0_nd]);
        
        #supply heuristic initial guess for the shooting method for the co-states
        lam_r_0         = -10**(-4);
        lam_theta_0     = 0;
        lam_r_dot_0     = 10**(-2);
        lam_v_theta_0   = 1;
        lam_m_0         = -10**(-3);
        
        #initial co-state vector
        self.arr_lam_0 = np.array([lam_r_0, lam_theta_0, lam_r_dot_0, lam_v_theta_0, lam_m_0]);
        
        #set scale factors
        self.scale_factors = [1, 1, 1, 1, 1];
        
        print("Boundary Conditions");
        print(f"R0 nd: {self.r_0_nd}");
        print(f"theta_0 nd: {self.theta_0_nd}");
        print(f"r_dot_0 nd: {self.r_dot_0_nd}");
        print(f"v_theta_0 nd: {self.v_theta_0_nd}");
        print(f"m_0 nd: {self.m_0_nd}");
        print(f"t_star: {self.t_star}");
        print("");
        print(f"r_f nd: {self.r_f_nd}");
        print(f"r_dot_f nd: {self.r_dot_f_nd}");
        print(f"v_theta_f nd: {self.v_theta_f_nd}");
        print("");
        print("C1 nd: ", self.C1_nd );
        print("C2 nd: ", self.C2_nd );
        print("");
        print("Initial co-state vector guess");
        print(self.arr_lam_0);
        print("\n\n\n");
    
    def __init__(self, env: TwoBody_Orb2Orb_Transfer_Env, init_observation, 
                 init_info, input_TOF ):
        
        self.env = env;                             #The Two body transfer gym environment
        self.init_observation = init_observation;   #The initial state of the env
        self.init_info = init_info;                 #Initial env info dict
        self.input_TOF = input_TOF;                 #User input time of flight [s]
        print("Hamiltonian targeter created\n");
        
        #extract the state vector boundary conditions from the problem
        self.extract_env_boundary_conditions();
        
        
    def shooting_iteration(self, lam_guess):
        
        #scale lambdas as necessary
        lam_guess_scaled = lam_guess*self.scale_factors;
        
        #construct full state vector at t=0
        arr_full_y0 = np.hstack( (self.arr_y0_nd, lam_guess_scaled) );
        
        #define time span
        t_span = (0,self.input_TOF);
        t_eval = np.linspace(*t_span, 1000);
        
        #prescribed boundary conditions for lambda_m and lambda_theta
        lam_m_f = -1;
        lam_theta_f = lam_guess[1]; #lambda theta isn't changing, so value should be init guess
        
        C1 = self.init_info["max_thrust"];
        C2 = self.init_info["ISP"];
        
        #set up parameter array
        params = np.array( [self.mu, C1, C2 ], dtype=np.float32 );
        
        #integrate forward in time
        sol = solve_ivp(Hamiltonian_EOM_TBT, t_span, arr_full_y0, method='RK45', args=(params,), t_eval=t_eval);
        
        #extract final state
        r_f_p, theta_f_p, r_dot_f_p, v_theta_f_p, m_f_p = sol.y[:5,-1];
        
        #extract final co-state
        lam_r_f_p, lam_theta_f_p, lam_r_dot_f_p, lam_v_theta_f_p, lam_m_f_p = sol.y[5:10,-1];
        
        #pack final state into an array
        y_f = [r_f_p, theta_f_p, r_dot_f_p, v_theta_f_p, m_f_p];
        
        residuals = np.array([
        r_f_p - self.r_f,                       # Final radius constraint
        r_dot_f_p - self.r_dot_f,               # Final radial velocity constraint
        v_theta_f_p - self.v_theta_f,           # Final tangential velocity constraint
        lam_theta_f_p - lam_theta_f,            # Co-state for theta shouldn't change
        lam_m_f_p - lam_m_f                     # Final mass co-state should be -1
        ])
        
        return residuals;
    
    def hamiltonian_root_finder(self):
        
        lam_guess_0 = self.arr_lam_0;
        
        lam_sol = root( self.shooting_iteration, lam_guess_0, method='lm' );
        
        # Check if the solution was successful
        if lam_sol.success:
            lam_solution = lam_sol.x
        else:
            raise Exception("Solver failed:", lam_sol.message)
        
        return lam_solution;
            
        
        
        
        
        