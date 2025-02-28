from TwoBody_Orb2Orb_Transfer_Env import *


class Hamiltonian_Controller_TBT:
    
    def extract_env_initial_conditions(self):
        r_0         = self.init_observation[0];
        theta_0     = self.init_observation[1];
        r_dot_0     = self.init_observation[2];
        v_theta_0   = self.init_observation[3];
        m_0         = self.init_observation[4];
        
        self.r_0 = r_0;
        self.theta_0 = theta_0;
        self.r_dot_0 = r_dot_0;
        self.v_theta_0 = v_theta_0;
        self.m_0 = m_0;
        
        print("Initial Conditions");
        print(f"R0: {r_0}");
        print(f"theta_0: {theta_0}");
        print(f"r_dot_0: {r_dot_0}");
        print(f"v_theta_0: {v_theta_0}");
        print(f"m_0: {m_0}");
    
    def __init__(self, env: TwoBody_Orb2Orb_Transfer_Env, init_observation, 
                 init_info, input_TOF ):
        
        self.env = env;                             #The Two body transfer gym environment
        self.init_observation = init_observation;   #The initial state of the env
        self.init_info = init_info;                 #Initial env info dict
        self.input_TOF = input_TOF;                 #User input time of flight [s]
        print("Hamiltonian targeter created\n");
        
        self.extract_env_initial_conditions();
        
        
    