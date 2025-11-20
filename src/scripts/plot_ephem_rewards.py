from core.training_data_generation import read_ephems_from_dir
from envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env
from utils.h_rl_fusion import calc_rewards_from_H_ephem
import matplotlib.pyplot as plt
from utils.log_utils import read_config_file
import os

def plot_H_ephem_rewards():

    print("Plotting H ephemeris rewards")

    num_ephems = 20
    dir_ephems = "C:\\Users\\micha\\MSI_Data\\Masters_Thesis\\z_script_output\\training_TBR_circular_20251110\\training_TBR_circular\\ephems"
    print("Reading ephemerides from directory: ", dir_ephems)
    set_ephems = read_ephems_from_dir(dir_ephems, num_ephems, version=2.0)
    print(f"Read {len(set_ephems)} ephemerides")
    # config path
    path_config = os.path.join("data", "config", "SAC_training_TBR_config.txt")

    # define normalization parameters (for NN)
    params = read_config_file(path_config)


    # Plot rewards
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    ax1.set_xlabel('Elapsed Time (days)')
    ax1.set_ylabel('Reward')
    ax1.set_title(f'Ephemeris Rewards Over Time')
    ax1.legend()
    ax2.set_xlabel('Elapsed Time (days)')
    ax2.set_ylabel('Cumulative Reward')
    plt.grid()


    for i in range(len(set_ephems)):
        ephem_H = set_ephems[i]
        print(f"Processing ephemeris {i+1}/{len(set_ephems)} with {ephem_H.num_vectors} vectors")

        # Create environment

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
            w_max_final_env_deg=params["w_max_final_env_deg"],
            pos_r_weight=params.get("pos_r_weight", 1.0),
            vel_r_weight=params.get("vel_r_weight", 1.0),
            mass_r_weight=params.get("mass_r_weight", 1.0),
            tof_scale=params.get("tof_scale", 1.0),
            r_dist_weight=params.get("r_dist_weight", 1.0),
            v_dist_weight=params.get("v_dist_weight", 1.0),
            success_threshold_pos=params.get("success_threshold_pos", 0.01),
            success_threshold_vel=params.get("success_threshold_vel", 0.01),
            terminal_bonus=params.get("terminal_bonus", 100.0),
            precision_mult=params.get("precision_mult", 10.0),
            tof_weight=params.get("tof_weight", 1.0),
            time_dist_weight=params.get("time_dist_weight", 1.0)
        )

        # Calculate rewards from ephemeris
        [
            arr_elapsed_time,
            arr_rewards,
            arr_pos_r_components,
            arr_vel_r_components,
            arr_mass_r_components,
            arr_r_tot
        ] = calc_rewards_from_H_ephem(ephem_H, env, params)

        # Plot total rewards
        ax1.plot(arr_elapsed_time, arr_rewards )
        ax2.plot(arr_elapsed_time, arr_r_tot )

    plt.show()


plot_H_ephem_rewards()
