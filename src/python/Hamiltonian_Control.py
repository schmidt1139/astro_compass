from TwoBody_Orb2Orb_Transfer_Env import *
from Propagation import Hamiltonian_EOM_TBT
from scipy.optimize import root
from scipy.optimize import fsolve
from StateVectorUtilities import *
from Ephemeris import *
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
        
        #convert initial state to cartesian
        x0, y0, vx0, vy0 = polar_to_cartesian(r_0, theta_0, r_dot_0, v_theta_0 );
        
        #Create initial state array
        arr_y0 = np.array([x0, y0, vx0, vy0, m_0]);
        
        
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
        
        #initial co-state guess
        lam_x0 = 0.1;
        lam_y0 = 0.1;
        lam_vx0 = 0.1;
        lam_vy0 = 0.1;
        lam_m0 = 0.1;
        
        #Pack initial co-state vector
        self.arr_lam_0 = np.array([lam_x0, lam_y0, lam_vx0, lam_vy0, lam_m0]);
        
        #Non-dimensionalize final boundary states
        self.r_f_nd         = r_f / self.l_star;
        self.r_dot_f_nd     = r_dot_f / self.l_star * self.t_star;
        self.v_theta_f_nd   = v_theta_f / self.l_star * self.t_star;
        
        #solution found flag (default to false)
        self.flag_solved = False;
        
        self._log_controller_info("Boundary Conditions");
        self._log_controller_info(f"arr_y nd: {arr_y0_nd}");
        self._log_controller_info(f"t_star: {self.t_star}");
        self._log_controller_info("");
        self._log_controller_info(f"r_f_nd: {self.r_f_nd}");
        self._log_controller_info("r_dot_f_nd: " + str( self.r_dot_f_nd ) );
        self._log_controller_info("v_theta_f_nd: " + str( self.v_theta_f_nd ) );
        self._log_controller_info("Initial co-state vector guess " + str(self.arr_lam_0) );
        
        
    
    def __init__(self, env: TwoBody_Orb2Orb_Transfer_Env, init_observation, 
                 init_info, input_TOF ):
        
        #Targeter log string array
        self.log = [];
        
        self.env = env;                             #The Two body transfer gym environment
        self.init_observation = init_observation;   #The initial state of the env
        self.init_info = init_info;                 #Initial env info dict
        self.input_TOF = input_TOF;                 #User input time of flight [s]
        self._log_controller_info("Hamiltonian Targeter Initialized");
        
        #extract the state vector boundary conditions from the problem
        self.extract_env_boundary_conditions();
        
        #Smoothing parameters
        #eps_threshold: The min value of smoothing parameter needed to reach a solution
        #gamma: The value to multiply eps by to gradually decrease it to eps_threshold
        #eps_0: The value of epsilon to start at
        #eps: The current value of epsilon
        #max_k: The maximum number of smoothing iterations to perform (ensures exit)
        #root_tol: The root finder function zero tolerance (increased if solver struggles)
        #root_tol_max: The max root tolerance, if this value is reached and there 
        #is still no convergence the targeting procedure fails.
        self.gamma  = 1 - (1/2)**(6);
        self.eps_threshold = 0.0025;
        self.eps_0  = 0.5;
        self.eps    = self.eps_0;
        self.max_k  = 640;
        self.root_tol = 0.5e-8;
        self.root_tol_max = 0.005;
        self.flag_constrain_u = True;
        self.root_method = "hybr"; #Choose from "hybr", "lm", "broyden1"
        self.root_max_iters = 1000;
        self.smoothing_method = 0; #Choose from 0 (tanh), 1 (homotopic)
        self.flag_stop_targeting = False;
        self.ivp_solve_rtol = 10**(-3);
        self.ivp_solve_atol = 10**(-6);
        
        
    def shooting_iteration(self, lam_guess_shooting, eps ):
        
        #construct full state vector at t=0
        arr_full_y0 = np.hstack( (self.arr_y0_nd, lam_guess_shooting) );
        
        #define time span
        t_span = (0,self.input_TOF_nd);
        t_eval = np.linspace(*t_span, 1000);
        
        #prescribed boundary conditions for lambda_m and lambda_theta
        lam_m_f = 0.0;
        
        #set up parameter array
        params = np.array( [self.mu_nd, self.T_max_nd, self.ISP_nd, 
                            self.l_star, self.m_star, self.t_star, self.g0_nd,
                            eps, self.flag_constrain_u, 
                            self.smoothing_method ] );
        
        #integrate forward in time
        sol = solve_ivp(Hamiltonian_EOM_TBT_v2, t_span, arr_full_y0, 
                        method='RK45', args=(params,), t_eval=t_eval, 
                        rtol = self.ivp_solve_rtol);
        
        if ( sol.status == -1 ):
            print(sol.message);
            raise Exception("Integration failed");
        
        #extract final cartesian state
        x_f_nd_p, y_f_nd_p, vx_f_nd_p, vy_f_nd_p, m_f_nd_p = sol.y[:5,-1];
        
        #extract final co-state
        lam_x_f_nd_p, lam_y_f_nd_p, lam_vx_f_nd_p, lam_vy_f_nd_p, lam_m_f_nd_p = sol.y[5:10,-1];
        
        #convert state to polar coordinates
        r_f_nd_p, theta_f_nd_p, vr_f_nd_p, vtheta_f_nd_p = cartesian_to_polar( 
                                                                        x_f_nd_p,
                                                                        y_f_nd_p,
                                                                        vx_f_nd_p,
                                                                        vy_f_nd_p );
        
        residuals = np.array([
        r_f_nd_p - self.r_f_nd,             # Final radius constraint
        vr_f_nd_p - self.r_dot_f_nd,        # Final radial velocity constraint
        vtheta_f_nd_p - self.v_theta_f_nd,  # Final tangential velocity constraint
        0.0,                                # Co-state for theta shouldn't change
        lam_m_f_nd_p - lam_m_f              # Final mass co-state should be 0
        ])
        
        return residuals;
    
    def hamiltonian_root_finder(self, eps, lam_guess ):
        
        try_max         = 10;
        try_count       = 1;
        flag_continue   = True;
        
        while( flag_continue ):

            #Call root finder method with the current eps (smoothing parameter)
            #and the current root finder tolerance value. If the root finder
            #function fails to reach a solution, the process is repeated with
            #a relaxed tolerance value. This process is repeated until the 
            #a maximum try count is reached or if the 
            
            if ( self.root_method != "hybr"):
                lam_sol = root(self.shooting_iteration, lam_guess, eps, 
                               tol=self.root_tol, method=self.root_method,
                               options={'maxiter': self.root_max_iters} );
            else:
                lam_sol = root(self.shooting_iteration, lam_guess, eps, 
                               tol=self.root_tol, method=self.root_method );
            
            #check if targeter is actually within tolerance if there is an early
            #exit
            if ( np.linalg.norm(lam_sol.fun) < self.root_tol ):
                lam_sol.success = True;
            
            if ( self.root_method != "broyden1"):
                
                fjac    = lam_sol.fjac;
                cn      = np.linalg.cond(fjac);
            
            if (lam_sol.success):
                
                flag_continue = False;
                
            elif ( (try_count < try_max ) and self.root_tol < self.root_tol_max ):
                
                self.root_tol   = self.root_tol * 10;
                try_count       = try_count + 1;
                self._log_controller_info(f"Increasing root tolerance value: {self.root_tol:.4e}");
                
            else:
                
                self._log_controller_info("Maximum attempts reached for root finding method");
                self._log_controller_info("self.root_tol: " + str( self.root_tol ) );
                flag_continue = False;
         
        # The throttle should be constained after the first iteration. The cap
        # on the throttle is lifted to get an initial solution
        if ( self. flag_constrain_u == False ):
            self. flag_constrain_u == True;
        
        # Check if the solution was successful
        if (lam_sol.success):
            
            lam_solution = lam_sol.x;
            
        else:
            
            lam_solution = lam_sol.x;
            self._log_controller_info("Lambda solution: " + str( lam_sol ) );
            
            if ( self.root_method != "broyden1"):
                self._log_controller_info( str( fjac ) );
                self._log_controller_info( "Jacobian condition number: " + str(cn) );
                self._log_controller_info( "Root tolerance reached: " + str( self.root_tol ) );
            
            
            self._log_controller_info("Try count: " + str(try_count));
            self.flag_stop_targeting = True;
            
        
        return lam_solution;
            
    def hamiltonian_solution_finder(self):
    
        #Initial smoothing parameters for the solution finder. The number of 
        #smoothing iterations that has been performed is tracked with the k
        #counter. The initial epsilon value is taken from the object property
        #self.eps_0.
        k   = 1;
        eps = self.eps_0;
        
        #The first step is to check the initial co-state guess, if it does not
        #lie sufficiently close to the real solution and the root finder fails, 
        #we re-try until a solution is achieved.
        arr_lam_sol_0 = self.check_initial_costate_guess();
        
        #provide initial co-state guess
        arr_lam_sol_0 = self.arr_lam_0;
        arr_lam_sol_k = arr_lam_sol_0;
        
        while ( (k <= self.max_k ) and (eps > self.eps_threshold) ):
            
            #update/decrease epsilon by gamma factor if it is not the first
            #iteration
            if( k != 1 ):
                eps = eps * self.gamma;
            
            #determine initial boundary values for co-states
            arr_lam_sol_k = self.hamiltonian_root_finder(eps, arr_lam_sol_0 );
            
            #next initial guess is the previous solution
            arr_lam_sol_0 = arr_lam_sol_k;
            
            #update k counter
            k = k + 1;
            
            if (self.flag_stop_targeting == True ):
                break;
        
        #assign co-state solution to Hamiltonian object after smoothing
        #iteration is complete.
        self.eps = eps;
        self.arr_lam_sol = arr_lam_sol_k;
        
        #construct full state vector at t=0
        arr_full_y0 = np.hstack( (self.arr_y0_nd, self.arr_lam_sol) );
        
        #define time span
        t_span          = (0,self.input_TOF_nd);
        t_eval = np.linspace(*t_span, 1000);
        
        #set up parameter array
        params = np.array( [self.mu_nd, self.T_max_nd, self.ISP_nd, 
                            self.l_star, self.m_star, self.t_star, self.g0_nd,
                            self.eps, self.flag_constrain_u,
                            self.smoothing_method] );
        
        #integrate forward in time
        sol = solve_ivp(Hamiltonian_EOM_TBT_v2, t_span, arr_full_y0, 
                        method='RK45', args=(params,), t_eval=t_eval,
                        rtol=self.ivp_solve_rtol );
        
        #assign solution to controller object and set solution flag to true
        self.final_sol      = sol;
        
        if ( sol.status == -1 or self.flag_stop_targeting == True ):
            self._log_controller_info(sol.message);
            self.flag_solved    = False;
            self._log_controller_info("Targeter failed to converge");
            self._log_controller_info("Epsilon reached: " + str(self.eps));
        else:
            self.flag_solved    = True;
            self._log_controller_info("Targeter converged");
               
        return self.flag_solved, self.arr_lam_sol, self.eps, sol, self.log;
                          
    
    def generate_output_ephemeris(self, ephemeris):
        
        #only write ephemeris if the controller has found a solution
        if ( self.flag_solved == False ):
            raise Exception("Controller has not solved, cannot write ephemeris");
            
        #extract time and state variables from solution
        arr_time    = self.final_sol.t;
        arr_u       = [];
        arr_rho     = [];
        alpha_vec_x = [];
        alpha_vec_y = [];
        
        #step through states and add to ephem object
        for index, t in enumerate(arr_time):
            
            #states
            t_i = t * self.t_star;
            x_i = self.final_sol.y[0,index] * self.l_star;
            y_i = self.final_sol.y[1,index] * self.l_star;
            vx_i = self.final_sol.y[2,index] * self.l_star / self.t_star;
            vy_i = self.final_sol.y[3,index] * self.l_star / self.t_star;
            m_i_nd = self.final_sol.y[4,index];
            m_i = m_i_nd * self.m_star;
            
            #co-states
            lam_x_i = self.final_sol.y[5,index];
            lam_y_i = self.final_sol.y[6,index];
            lam_vx_i = self.final_sol.y[7,index];
            lam_vy_i = self.final_sol.y[8,index];
            lam_m_i = self.final_sol.y[9,index];
            
            r_i = np.linalg.norm([x_i, y_i]);
            r_vec = np.array([x_i, y_i, 0]);
            v_vec = np.array([vx_i, vy_i, 0]);
            lam_v_vec = np.array([lam_vx_i, lam_vy_i]);
            lam_v_mag = np.linalg.norm(lam_v_vec);
            
            #Find alpha vector
            alpha_vec = - lam_v_vec / lam_v_mag;
            
            #Switching function
            rho = lam_m_i + self.ISP_nd * self.g0_nd * lam_v_mag / m_i_nd - 1;
            
            if ( self.eps == 0.0 ):  
                if ( rho >= 0 ):
                    u = 1.0;
                else:
                    u = 0.0;       
            else:
                #check the smoothing method
                if( self.smoothing_method == 0 ):
                    u = smoothing_function_tanh( rho, self.eps );
                elif( self.smoothing_method == 1 ):
                    u = smoothing_function_homotopic(rho, self.eps, 
                                                     self.flag_constrain_u);
            
            r_i = np.linalg.norm([x_i, y_i]);
            r_vec = np.array([x_i, y_i, 0]);
            v_vec = np.array([vx_i, vy_i, 0]);
            lam_v_vec = np.array([lam_vx_i, lam_vy_i]);
            lam_v_mag = np.linalg.norm(lam_v_vec);
            
            ephemeris.add_data( t_i, x_i, y_i, vx_i, vy_i, m_i, alpha_vec[0], 
                               alpha_vec[1], u );
            
            arr_u.append(u);
            arr_rho.append(rho);
            alpha_vec_x.append(alpha_vec[0]);
            alpha_vec_y.append(alpha_vec[1]);
            
        return ephemeris, arr_time, arr_u, arr_rho, alpha_vec_x, alpha_vec_y;
    
    def check_initial_costate_guess(self):
        
        self._log_controller_info("Checking initial co-state guess");
        
        flag_good_first_guess   = False;
        counter_first_guess     = 0;
        max_iters               = 100;
        mean_co_state_guess     = 0.0;
        std_co_state_guess      = 0.01;
        len_co_state_guess      = len(self.arr_lam_0);
        bias_co_states          = np.array([ 0.0, 0.0, 0.0, 0.0, 0.1]);
        lam_guess               = self.arr_lam_0;
        
        while( flag_good_first_guess == False ):
            
            counter_first_guess = counter_first_guess + 1;
            
            if ( counter_first_guess > max_iters ):
                raise Exception("Cannot find good initial co-state guess");
            
            #randomize the first guess if the first guess is no good
            if ( counter_first_guess > 1 ):
                lam_guess = np.random.normal(loc=mean_co_state_guess, 
                                             scale=std_co_state_guess,
                                             size=len_co_state_guess);
                
                #add bias array
                lam_guess = lam_guess + bias_co_states;
            
            lam_sol     = root(self.shooting_iteration, lam_guess, self.eps_0, tol=self.root_tol );
            
            if (self.root_method != "broyden1"):
                fjac        = lam_sol.fjac;
                cn          = np.linalg.cond(fjac);
                
            success     = lam_sol.success;
            
            if ( success ):
                self._log_controller_info("Attempt " + str( counter_first_guess ) + "   Lambda: " + str( lam_guess ) + " passed" );
                return lam_guess;
            else:
                self._log_controller_info("Lambda: " + str( counter_first_guess ) + "   lam_guess " + str( lam_guess ) + " failed" );
        
    def _log_controller_info( self, info ):
        
        self.log.append(info);