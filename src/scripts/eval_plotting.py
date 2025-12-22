import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import scipy

def misc_plotting(path_to_data=None):
    """Generate and save plots for pre-training losses.

    Args:
        arr_iters (list): List of iteration numbers.
        arr_actor_loss_pt (list): List of actor loss values during pre-training.
        arr_critic_loss_pt (list): List of critic loss values during pre-training.
        path_plots (str): Directory path to save the plots.
    """
    import matplotlib.pyplot as plt
    import os

    print("Misc Plotting Script")
    print(f"Path to plots: {path_to_data}")

    plt.style.use("data/support_files/light_paper.mplstyle")

    #subdirs
    path_vanilla_r = os.path.join(path_to_data, "0_pt", "plots","mc_episode_rewards.npy")
    path_vanilla_10k_r = os.path.join(path_to_data, "10k_pt", "plots","mc_episode_rewards.npy")
    path_vanilla_20k_r = os.path.join(path_to_data, "20k_pt", "plots","mc_episode_rewards.npy")
    path_vanilla_310k_r = os.path.join(path_to_data, "310k_pt", "plots","mc_episode_rewards.npy")
    path_vanilla_760k_r = os.path.join(path_to_data, "760k_pt", "plots","mc_episode_rewards.npy")

    # Define all five runs
    runs = [
        ("0_pt", "pt0"),
        ("760k_pt", "pt760k"),
    ]
    
    # Load data for all runs
    all_data = {}
    for dir_name, label in runs:
        data_dict = {}
        
        # Load rewards
        path_r = os.path.join(path_to_data, dir_name, "plots", "mc_episode_rewards.npy")
        data_dict['r'] = np.load(path_r)
        data_dict['r_sorted'] = np.sort(data_dict['r'])
        data_dict['cdf_r'] = np.arange(1, len(data_dict['r_sorted']) + 1) / len(data_dict['r_sorted'])
        
        # Load final mass
        path_m = os.path.join(path_to_data, dir_name, "plots", "mc_episode_final_mass.npy")
        data_dict['m'] = np.load(path_m)
        data_dict['m_sorted'] = np.sort(data_dict['m'])
        data_dict['cdf_m'] = np.arange(1, len(data_dict['m_sorted']) + 1) / len(data_dict['m_sorted'])
        
        # Load position residual
        path_p = os.path.join(path_to_data, dir_name, "plots", "mc_episode_position_res.npy")
        data_dict['p'] = np.load(path_p)
        data_dict['p_sorted'] = np.sort(data_dict['p'])
        data_dict['cdf_p'] = np.arange(1, len(data_dict['p_sorted']) + 1) / len(data_dict['p_sorted'])
        
        # Load velocity residual
        path_v = os.path.join(path_to_data, dir_name, "plots", "mc_episode_velocity_res.npy")
        data_dict['v'] = np.load(path_v)
        data_dict['v_sorted'] = np.sort(data_dict['v'])
        data_dict['cdf_v'] = np.arange(1, len(data_dict['v_sorted']) + 1) / len(data_dict['v_sorted'])
        
        all_data[label] = data_dict
    
    # Legacy variable names for the old vanilla SAC
    r_data_vanilla = all_data['pt0']['r']
    r_data_vanilla_sorted = all_data['pt0']['r_sorted']
    cdf_vanilla = all_data['pt0']['cdf_r']
    m_data_vanilla = all_data['pt0']['m']
    m_data_vanilla_sorted = all_data['pt0']['m_sorted']
    cdf_vanilla_m = all_data['pt0']['cdf_m']
    p_data_vanilla = all_data['pt0']['p']
    p_data_vanilla_sorted = all_data['pt0']['p_sorted']
    cdf_vanilla_p = all_data['pt0']['cdf_p']
    v_data_vanilla = all_data['pt0']['v']
    v_data_vanilla_sorted = all_data['pt0']['v_sorted']
    cdf_vanilla_v = all_data['pt0']['cdf_v']
    
    # Legacy variable names for pre-trained SAC (use pt760k as reference)
    r_data_pt = all_data['pt760k']['r']
    r_data_pt_sorted = all_data['pt760k']['r_sorted']
    cdf_pt = all_data['pt760k']['cdf_r']
    m_data_pt = all_data['pt760k']['m']
    m_data_pt_sorted = all_data['pt760k']['m_sorted']
    cdf_pt_m = all_data['pt760k']['cdf_m']
    p_data_pt = all_data['pt760k']['p']
    p_data_pt_sorted = all_data['pt760k']['p_sorted']
    cdf_pt_p = all_data['pt760k']['cdf_p']
    v_data_pt = all_data['pt760k']['v']
    v_data_pt_sorted = all_data['pt760k']['v_sorted']
    cdf_pt_v = all_data['pt760k']['cdf_v']

    plt.figure(figsize=(14, 10))
    
    # Rewards CDF
    ax1 = plt.subplot(2, 2, 1)
    colors = ['C0', 'C1', 'C2']
    for (_, label), color in zip(runs, colors):
        data = all_data[label]
        ax1.plot(data['r_sorted'], data['cdf_r'], linewidth=1, label=f'{label} (avg: {np.mean(data["r"]):.3f})', color=color)
    ax1.set_xlabel('Reward', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)
    
    # Final Mass CDF
    ax2 = plt.subplot(2, 2, 2)
    for (_, label), color in zip(runs, colors):
        data = all_data[label]
        ax2.plot(data['m_sorted'], data['cdf_m'], linewidth=1, label=f'{label} (avg: {np.mean(data["m"]):.3f})', color=color)
    ax2.set_xlabel('Final Mass (nd)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)
    
    # Position Residual CDF
    ax3 = plt.subplot(2, 2, 3)
    for (_, label), color in zip(runs, colors):
        data = all_data[label]
        ax3.plot(data['p_sorted'], data['cdf_p'], linewidth=1, label=f'{label} (avg: {np.mean(data["p"]):.3f})', color=color)
    ax3.set_xlabel('Terminal Position Residual (nd)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=11)
    
    # Velocity Residual CDF
    ax4 = plt.subplot(2, 2, 4)
    for (_, label), color in zip(runs, colors):
        data = all_data[label]
        ax4.plot(data['v_sorted'], data['cdf_v'], linewidth=1, label=f'{label} (avg: {np.mean(data["v"]):.3f})', color=color)
    ax4.set_xlabel('Terminal Velocity Residual (nd)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=13)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_data, "mc_cdf.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # PDF plots
    plt.figure(figsize=(14, 10))
    
    # Rewards PDF
    ax1 = plt.subplot(2, 2, 1)
    for (_, label), color in zip(runs, colors):
        data = all_data[label]
        ax1.hist(data['r'], bins=20, linewidth=1, alpha=0.6, label=f'{label} (avg: {np.mean(data["r"]):.3f})', color=color)
    ax1.set_xlabel('Reward', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)
    
    # Final Mass PDF
    ax2 = plt.subplot(2, 2, 2)
    for (_, label), color in zip(runs, colors):
        data = all_data[label]
        ax2.hist(data['m'], bins=20, linewidth=1, alpha=0.6, label=f'{label} (avg: {np.mean(data["m"]):.3f})', color=color)
    ax2.set_xlabel('Final Mass (nd)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)
    
    # Position Residual PDF
    ax3 = plt.subplot(2, 2, 3)
    for (_, label), color in zip(runs, colors):
        data = all_data[label]
        ax3.hist(data['p'], bins=20, linewidth=1, alpha=0.6, label=f'{label} (avg: {np.mean(data["p"]):.3f})', color=color)
    ax3.set_xlabel('Terminal Position Residual (nd)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=11)
    
    # Velocity Residual PDF
    ax4 = plt.subplot(2, 2, 4)
    for (_, label), color in zip(runs, colors):
        data = all_data[label]
        ax4.hist(data['v'], bins=20, linewidth=1, alpha=0.6, label=f'{label} (avg: {np.mean(data["v"]):.3f})', color=color)
    ax4.set_xlabel('Terminal Velocity Residual (nd)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=13)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_data, "mc_pdf.png"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    path_to_data = "C:\\Users\\micha\\MSI_Data\\Masters_Thesis\\z_script_output\\runs_for_record\\Dec16\\agent_evals"
    path_to_data = os.path.abspath(path_to_data)
    misc_plotting(path_to_data)