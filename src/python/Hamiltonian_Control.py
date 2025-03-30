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
        
        #Smoothing parameters
        #eps_threshold: The min value of smoothing parameter needed to reach a solution
        #gamma: The value to multiply eps by to gradually decrease it to eps_threshold
        #eps_0: The value of epsilon to start at
        #eps: The current value of epsilon
        #max_k: The maximum number of smoothing iterations to perform (ensures exit)
        #root_tol: The root finder function zero tolerance (increased if solver struggles)
        #root_tol_max: The max root tolerance, if this value is reached and there 
        #              is still no convergence the targeting procedure fails.
        self.eps_threshold = 10**(-3);
        self.gamma  = 0.97;
        self.eps_0  = 0.6;
        self.eps    = self.eps_0;
        self.max_k  = 640;
        self.root_tol = 1e-8;
        self.root_tol_max = 1e-3;
        
        #supply heuristic initial guess for the shooting method for the co-states
        lam_x0 = 0.286298956079894;
        lam_y0 = -0.0214548070543362;
        lam_vx0 = -0.0689585667746195;
        lam_vy0 = 0.6266476511221035;
        lam_m0 = 0.14579433945759;
        
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
        
        #Pack initial co-state vector
        self.arr_lam_0 = np.array([lam_x0, lam_y0, lam_vx0, lam_vy0, lam_m0]);
        
        #Non-dimensionalize final boundary states
        self.r_f_nd         = r_f / self.l_star;
        self.r_dot_f_nd     = r_dot_f / self.l_star * self.t_star;
        self.v_theta_f_nd   = v_theta_f / self.l_star * self.t_star;
        
        #solution found flag (default to false)
        self.flag_solved = False;
        
        print("Boundary Conditions");
        print(f"arr_y nd: {arr_y0_nd}");
        print(f"t_star: {self.t_star}");
        print("");
        print(f"r_f_nd: {self.r_f_nd}");
        print("r_dot_f_nd: ", self.r_dot_f_nd );
        print("v_theta_f_nd: ", self.v_theta_f_nd );
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
        sol = solve_ivp(Hamiltonian_EOM_TBT_v2, t_span, arr_full_y0, method='RK45', args=(params,), t_eval=t_eval );
        
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
            lam_sol = root(self.shooting_iteration, lam_guess, eps, tol=self.root_tol );
            fjac    = lam_sol.fjac;
            cn      = np.linalg.cond(fjac);
            
            if (lam_sol.success):
                
                flag_continue = False;
                
            elif ( (try_count < try_max) and self.root_tol <= self.root_tol_max ):
                
                self.root_tol   = self.root_tol * 10;
                try_count       = try_count + 1;
                print(f"Increasing root tolerance value: {self.root_tol:.4e}");
                
            else:
                
                print("Maximum attempts reached for root finding method");
                print("self.root_tol: ", self.root_tol );
                flag_continue = False;
                
        
        # Check if the solution was successful
        if (lam_sol.success):
            
            lam_solution = lam_sol.x;
            
        else:
            print("Lambda solution: ", lam_sol);
            print(fjac);
            print("Jacobian condition number: ", cn);
            print("Root tolerance reached: ", self.root_tol);
            print("Try count: ", try_count);
            raise Exception("Solver failed:");
        
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
        
        while ( (k <= self.max_k) and (eps > self.eps_threshold) ):
            
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
        sol = solve_ivp(Hamiltonian_EOM_TBT_v2, t_span, arr_full_y0, method='RK45', args=(params,), t_eval=t_eval );
        
        #assign solution to controller object and set solution flag to true
        self.final_sol      = sol;
        self.flag_solved    = True;
        
        if ( sol.status == -1 ):
            print(sol.message);
            raise Exception("Integration failed");
               
        return self.flag_solved, self.arr_lam_sol, self.eps, sol;
                          
    
    def generate_output_ephemeris(self, ephemeris):
        
        #only write ephemeris if the controller has found a solution
        if ( self.flag_solved == False ):
            raise Exception("Controller has not solved, cannot write ephemeris");
            
        #extract time and state variables from solution
        arr_time    = self.final_sol.t;
        variables   = self.final_sol.y;    
        arr_u       = [];
        arr_rho     = [];
        alpha_vec_x = [];
        alpha_vec_y = [];
        
        #step through states and add to ephem object
        for index, t in enumerate(arr_time):
            
            #states
            x_i = variables[0,index] * self.l_star;
            y_i = variables[1,index] * self.l_star;
            vx_i = variables[2,index] * self.l_star / self.t_star;
            vy_i = variables[3,index] * self.l_star / self.t_star;
            m_i_nd = variables[4,index];
            m_i = m_i_nd * self.m_star;
            
            #co-states
            lam_x_i = variables[5,index];
            lam_y_i = variables[6,index];
            lam_vx_i = variables[7,index];
            lam_vy_i = variables[8,index];
            lam_m_i = variables[9,index];
            
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
            
            ephemeris.add_data( t, x_i, y_i, vx_i, vy_i, m_i );
            
            arr_u.append(u);
            arr_rho.append(rho);
            alpha_vec_x.append(alpha_vec[0]);
            alpha_vec_y.append(alpha_vec[1]);
            
        return ephemeris, arr_time, arr_u, arr_rho, alpha_vec_x, alpha_vec_y;
    
    def check_initial_costate_guess(self):
        
        print("Checking initial co-state guess");
        
        flag_good_first_guess   = False;
        counter_first_guess     = 0;
        max_iters               = 100;
        mean_co_state_guess     = 0.0;
        std_co_state_guess      = 0.01;
        len_co_state_guess      = len(self.arr_lam_0);
        bias_co_states          = np.array([ 0.0, 0.0, 0.0, 0.0, 0.0]);
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
            fjac        = lam_sol.fjac;
            cn          = np.linalg.cond(fjac);
            success     = lam_sol.success;
            
            if ( abs(max(lam_sol.x)) > 1 ):
                success = False;
            
            if ( success ):
                print("Attempt ", counter_first_guess, "   Lambda: ", lam_guess, " passed" );
                return lam_guess;
            else:
                print("Lambda: ", counter_first_guess, "   lam_guess ", lam_guess, " failed" );
        
        
        