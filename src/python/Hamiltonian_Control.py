from TwoBody_Orb2Orb_Transfer_Env import *
from Propagation import Hamiltonian_EOM_TBT
from scipy.optimize import root
from scipy.optimize import fsolve
from StateVectorUtilities import *
import numpy as np;

class Hamiltonian_Controller_TBT:
    
    def extract_env_boundary_conditions(self):
        
        #extract initial conditions
        r_0         = self.init_observation[0] * 1000;    #radius in m
        theta_0     = self.init_observation[1];           #theta in rad
        r_dot_0     = self.init_observation[2] * 1000;    #radial vel in m/s
        v_theta_0   = self.init_observation[3] * 1000;    #tangential vel in m/s
        m_0         = self.init_observation[4];           #mass in kg
        mu          = self.init_observation[5] * 1000**3; #gravitational param    
        r_f         = self.init_observation[6] * 1000;    #final r in m
        r_dot_f     = 0.0; #final radial r in m/s
        v_theta_f   = (mu/r_f) ** 0.5; #final tangential vel in m/s
        
        
        #constants
        self.l_star         = 149598023 * 1000; #Earth SMA in m
        self.mu             = mu; #gravitational parameter in m^3/s^2
        self.t_star         = ( self.l_star**3 / self.mu ) ** 0.5; #non-dimensional time in s
        self.m_star         = m_0; #kg
        g0                  = 9.80665; #m/s^2
        
        #Parameters
        T_max = self.init_info["max_thrust"]*1000; #max thrust in N
        ISP = self.init_info["ISP"]; #specific impulse of thruster in seconds
        
        #supply heuristic initial guess for the shooting method for the co-states
        lam_x0 = 1.0;
        lam_y0 = 1.0;
        lam_vx0 = -1.0;
        lam_vy0 = 1.0;
        lam_m0 = 1.0;
        
        #convert initial state to cartesian
        x0, y0, vx0, vy0 = polar_to_cartesian(r_0, theta_0, r_dot_0, v_theta_0 );
        
        #Create initial state array
        arr_y0 = np.array([x0, y0, vx0, vy0, m_0]);
        
        
        #set scale factors
        self.scale_factors = [1, 1, 1, 1, 1];
        #Non-Dimensionalize State Vector and Parameters
        nd_outputs = non_dimensionalize( arr_y0, g0, mu, T_max, ISP, 
                                        self.input_TOF, self.l_star, 
                                        self.m_star, self.t_star );
        
        #Unpack state vector
        arr_y0_nd, g0_nd, mu_nd, T_max_nd, ISP_nd, input_TOF_nd = nd_outputs;
        self.arr_y0_nd = arr_y0_nd;
        self.g0_nd = g0_nd;
        self.mu_nd = mu_nd;
        self.T_max_nd = T_max_nd;
        self.ISP_nd = ISP_nd;
        self.input_TOF_nd = input_TOF_nd;
        
        #Pack initial co-state vector
        self.arr_lam_0 = np.array([lam_x0, lam_y0, lam_vx0, lam_vy0, lam_m0]);
        
        #Non-dimensionalize final boundary states
        self.r_f_nd         = r_f / self.l_star;
        self.r_dot_f_nd     = r_dot_f / self.l_star * self.t_star;
        self.v_theta_f_nd   = v_theta_f / self.l_star * self.t_star;
        
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
        t_span = (0,self.input_TOF_nd);
        t_eval = np.linspace(*t_span, 1000);
        
        #prescribed boundary conditions for lambda_m and lambda_theta
        lam_m_f = 0.0;
        lam_theta_f = lam_guess[1]; #lambda theta isn't changing, so value should be init guess
        
        #set up parameter array
        params = np.array( [self.mu_nd, self.C1_nd, self.C2_nd ], dtype=np.float32 );
        
        #integrate forward in time
        sol = solve_ivp(Hamiltonian_EOM_TBT_nd, t_span, arr_full_y0, method='RK45', args=(params,), t_eval=t_eval );
        
        if ( sol.status == -1 ):
            print(sol.message);
            raise Exception("Integration failed");
        
        #extract final state
        r_f_p_nd, theta_f_p, r_dot_f_p_nd, v_theta_f_p_nd, m_f_p_nd = sol.y[:5,-1];
        
        #extract final co-state
        lam_r_f_p, lam_theta_f_p, lam_r_dot_f_p, lam_theta_f_p_nd, lam_m_f_p_nd = sol.y[5:10,-1];
        
        #pack final state into an array
        y_f = [r_f_p_nd, theta_f_p, r_dot_f_p_nd, v_theta_f_p_nd, m_f_p_nd];
        
        #scale final co-state for mass
        lam_m_f_p_scaled = lam_m_f_p_nd;
        
        residuals = np.array([
        r_f_p_nd - self.r_f_nd,                 # Final radius constraint
        r_dot_f_p_nd - self.r_dot_f_nd,         # Final radial velocity constraint
        v_theta_f_p_nd - self.v_theta_f_nd,     # Final tangential velocity constraint
        lam_theta_f_p_nd - lam_theta_f,            # Co-state for theta shouldn't change
        lam_m_f_p_scaled - lam_m_f              # Final mass co-state should be 0
        ])
        
        # print("r_f_p_nd: ", r_f_p_nd);
        # print("theta_f_p_nd", theta_f_p);
        # print("r_dot_f_p_nd: ", r_dot_f_p_nd);
        # print("v_theta_f_p_nd: ", v_theta_f_p_nd);
        # print("m_f_p_nd: ", m_f_p_nd);
        # print("");
        # print("lam_r_f_p: ", lam_r_f_p);
        # print("lam_theta_f_p: ", lam_theta_f_p);
        # print("lam_r_dot_f_p: ", lam_r_dot_f_p);
        # print("lam_v_theta_f_p: ", lam_v_theta_f_p);
        # print("lam_m_f_p: ", lam_m_f_p);
        # print("");
        # print("lam_theta_f: ", lam_theta_f);
        # print("lam_m_f: ", lam_m_f );
        # print("")
        #print("Res norm: ", np.linalg.norm(residuals));
        #print("");
        # print("Lam guess", lam_guess );
        # print("Residual lambda m: ", residuals[4] );
        # print("");
        print(lam_guess);
        print(residuals);
        print("");
        # print("Res norm: ", np.linalg.norm(residuals));
        # print("\n\n\n");
        
        return residuals;
    
    def hamiltonian_root_finder(self):
        
        lam_guess_0 = self.arr_lam_0;
        
        #lam_sol, info, ier, msg = fsolve(self.shooting_iteration, lam_guess_0, full_output=1);
        #lam_sol = root(self.shooting_iteration, lam_guess_0, method='lm');
        lam_sol = root(self.shooting_iteration, lam_guess_0 );
        fjac = lam_sol.fjac;
        cn = np.linalg.cond(fjac);
        
        # Check if the solution was successful
        if (lam_sol.success):
            print(lam_sol);
            print(fjac);
            print(cn);
            lam_solution = lam_sol.x;
        else:
            print(lam_sol);
            print(fjac);
            print(cn);
            raise Exception("Solver failed:");
        
        return lam_solution;
            
    def hamiltonian_solution_finder(self):
        
       
        #determine initial states for co-states
        self.arr_lam_sol = self.hamiltonian_root_finder();
        
        print("Initial co-state values found...");
        
        #scale lambdas as necessary
        lam_sol_scaled = self.arr_lam_sol*self.scale_factors;
        
        #construct full state vector at t=0
        arr_full_y0 = np.hstack( (self.arr_y0_nd, lam_sol_scaled) );
        
        #define time span
        t_span          = (0,self.input_TOF_nd);
        t_eval = np.linspace(*t_span, 1000);
        
        #set up parameter array
        params = np.array( [self.mu_nd, self.C1_nd, self.C2_nd ], dtype=np.float32 );
        
        #integrate forward in time
        sol = solve_ivp(Hamiltonian_EOM_TBT_nd, t_span, arr_full_y0, method='RK45', args=(params,), t_eval=t_eval );
        
        if ( sol.status == -1 ):
            print(sol.message);
            raise Exception("Integration failed");
        
        
        return sol;