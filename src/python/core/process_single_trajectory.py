import os
import time
import matplotlib.pyplot as plot
from envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env
from core.gen_Hamiltonian_trajectory import gen_Hamiltonian_trajectory

def process_single_trajectory(traj_info, params):
    """
    Process a single trajectory. This function will be called in parallel.
    
    Args:
        traj_info: Tuple of (traj_num, seed_traj, tof_scale)
        params: Dictionary of parameters
        
    Returns:
        Tuple of (success, ephem_path, seed_traj, str_gen_time)
    """
    traj_num, seed_traj, tof_scale = traj_info
    
    # Create environment for this process
    env = TwoBodyRendezvous_Env(
        mu=params["mu"],
        max_T=params["max_T"],
        ISP=params["ISP"],
        l_star=params["l_star"],
        m_star=params["m_star"],
        t_star=params["t_star"],
        g0=params["g0"],
        step_size=params["env_step_size"],
        a_min_env_nd=params["a_min_env_nd"],
        a_max_env_nd=params["a_max_env_nd"],
        e_min_env=params["e_min_env"],
        e_max_env=params["e_max_env"],
        w_min_env_deg=params["w_min_env_deg"],
        w_max_env_deg=params["w_max_env_deg"],
    )
    
    # Generate filename
    tof_scale_str = str(tof_scale).replace('.', 'p')
    ephem_filename = f"test_TBR_ephem_traj_seed_{seed_traj}_tof_{tof_scale_str}"
    
    # Generate trajectory
    test_log = []
    flag_solved, test_log, eph_output = gen_Hamiltonian_trajectory(
        env, seed_traj, tof_scale, params, ephem_filename
    )
    
    str_gen_time = time.strftime("%b %d %Y %H:%M:%S")
    
    # Save plots if solved
    if flag_solved and params["flag_plot_traj"] and eph_output is not None:
        eph_output.save_plots(params["data_path"], ephem_filename, params, env)
    
    ephem_path = os.path.join(params["data_path"], ephem_filename + ".txt") if flag_solved else None
    
    # Close any plots to free memory
    plot.close('all')
    
    return (flag_solved, ephem_path, seed_traj, str_gen_time)