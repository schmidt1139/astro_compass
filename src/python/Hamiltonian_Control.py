from TwoBody_Orb2Orb_Transfer_Env import *


class Hamiltonian_Controller_TBT:
    
    def extract_env_boundary_conditions(self):
        
        r_0         = self.init_observation[0];
        theta_0     = self.init_observation[1];
        r_dot_0     = self.init_observation[2];
        v_theta_0   = self.init_observation[3];
        m_0         = self.init_observation[4];
        mu          = self.init_observation[5];
        r_f         = self.init_observation[6];
        r_dot_f     = 0.0;
        v_theta_f   = (mu/r_f) ** 0.5;
        
        self.r_0            = r_0;
        self.theta_0        = theta_0;
        self.r_dot_0        = r_dot_0;
        self.v_theta_0      = v_theta_0;
        self.m_0            = m_0;
        self.r_f            = r_f;
        self.r_dot_f        = r_dot_f;
        self.v_theta_f      = v_theta_f;
        
        
        
        print("Boundary Conditions");
        print(f"R0: {r_0}");
        print(f"theta_0: {theta_0}");
        print(f"r_dot_0: {r_dot_0}");
        print(f"v_theta_0: {v_theta_0}");
        print(f"m_0: {m_0}");
        print(f"r_f: {r_f}");
        print(f"r_dot_f: {r_dot_f}");
        print(f"v_theta_f: {v_theta_f}");
    
    def __init__(self, env: TwoBody_Orb2Orb_Transfer_Env, init_observation, 
                 init_info, input_TOF ):
        
        self.env = env;                             #The Two body transfer gym environment
        self.init_observation = init_observation;   #The initial state of the env
        self.init_info = init_info;                 #Initial env info dict
        self.input_TOF = input_TOF;                 #User input time of flight [s]
        print("Hamiltonian targeter created\n");
        
        #extract the state vector boundary conditions from the problem
        self.extract_env_boundary_conditions();
        
        #supply heuristic initial guess for the shooting method for the co-states
        lam_r_0         = -10^(-4);
        lam_theta_0     = 0;
        lam_r_dot_0     = 10^(-2);
        lam_v_theta_0   = 1;
        lam_m_0         = -10^(-3);
        
        self.arr_lam_0 = np.array([lam_r_0, lam_theta_0, lam_r_dot_0, lam_v_theta_0, 
                              lam_m_0]);
        
        
    