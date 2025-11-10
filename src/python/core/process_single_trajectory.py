import os
import time
import matplotlib.pyplot as plot
from envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env
from core.gen_Hamiltonian_trajectory import gen_Hamiltonian_trajectory
from utils.log_utils import write_log_to_file, log, log_parameters

def process_single_trajectory(params):
    """
    Process a single trajectory. This function will be called in parallel.
    
    Args:
        params: Dictionary of parameters containing all trajectory-specific settings
                including 'traj_num', 'seed_traj', and 'tof_scale'
        
    Returns:
        Tuple of (success, ephem_path, seed_traj, str_gen_time)
    """
    traj_num = params["traj_num"]
    seed_traj = params["seed_traj"]
    tof_scale = params["tof_scale"]
    flag_report_live = params.get("flag_report_live", False)
    
    #print(f"[DEBUG] traj_num={traj_num}, seed={seed_traj}, flag_report_live={flag_report_live}, in params: {'flag_report_live' in params}")
    
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
        a_min_init_env_nd=params["a_min_init_env_nd"],
        a_max_init_env_nd=params["a_max_init_env_nd"],
        e_min_init_env=params["e_min_init_env"],
        e_max_init_env=params["e_max_init_env"],
        w_min_init_env_deg=params["w_min_init_env_deg"],
        w_max_init_env_deg=params["w_max_init_env_deg"],
        a_min_final_env_nd=params["a_min_final_env_nd"],
        a_max_final_env_nd=params["a_max_final_env_nd"],
        e_min_final_env=params["e_min_final_env"],
        e_max_final_env=params["e_max_final_env"],
        w_min_final_env_deg=params["w_min_final_env_deg"],
        w_max_final_env_deg=params["w_max_final_env_deg"]
    )
    
    # Generate filename
    tof_scale_str = str(tof_scale).replace('.', 'p')
    ephem_filename = f"test_TBR_ephem_traj_seed_{seed_traj}_tof_{tof_scale_str}"
    
    # Record when this worker actually starts processing
    worker_start_time = time.time()
    
    # Generate trajectory
    test_log = []
    flag_solved, test_log, eph_output = gen_Hamiltonian_trajectory(
        env, seed_traj, tof_scale, params, ephem_filename, test_log, 
        flag_report_live = False
    )
    
    str_gen_time = time.strftime("%b %d %Y %H:%M:%S")
    
    # Save plots if solved
    if flag_solved and params["flag_plot_traj"] and eph_output is not None:
        eph_output.save_plots(params["data_path"], ephem_filename, params, env)
    
    ephem_path = os.path.join(params["data_path"], ephem_filename + ".txt") if flag_solved else None
    
    # write log to file
    if (flag_report_live):
        
        # Add parameters to the log after trajectory generation
        test_log = log_parameters(params, test_log, False)

        # flag solved string
        flag_solved_str = "Solved" if flag_solved else "Failed"

        #make log directory if it doesn't exist
        # Convert to absolute path to ensure it works in multiprocessing workers
        data_path_abs = os.path.abspath(params["data_path"])
        log_dir = os.path.join(data_path_abs, "logs")
        
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, f"process_single_trajectory_{seed_traj}_{flag_solved_str}.log")
            write_log_to_file(log_file_path, test_log)
            #print(f"[DEBUG] Log written to: {log_file_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write log: {type(e).__name__}: {e}")
            print(f"[ERROR] Attempted path: {log_dir}")
            print(f"[ERROR] data_path from params: {params['data_path']}")
            print(f"[ERROR] data_path_abs: {data_path_abs}")

    # Close any plots to free memory
    plot.close('all')
    
    # Calculate actual processing time
    processing_time = time.time() - worker_start_time
    
    # Check if we exceeded timeout
    timeout_per_trajectory = params.get("timeout_per_trajectory", 300)
    timed_out = processing_time > timeout_per_trajectory
    
    return (flag_solved, ephem_path, seed_traj, str_gen_time, timed_out, processing_time, params)