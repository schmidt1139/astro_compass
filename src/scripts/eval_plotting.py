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

    #vanilla SAC
    path_vanilla_r = os.path.join(path_to_data, "vanilla", "plots","mc_episode_rewards.npy")
    r_data_vanilla = np.load(path_vanilla_r)
    r_data_vanilla_sorted = np.sort(r_data_vanilla)
    cdf_vanilla = np.arange(1, len(r_data_vanilla_sorted) + 1) / len(r_data_vanilla_sorted)

    path_vanilla_m = os.path.join(path_to_data, "vanilla", "plots","mc_episode_final_mass.npy")
    m_data_vanilla = np.load(path_vanilla_m)
    m_data_vanilla_sorted = np.sort(m_data_vanilla)
    cdf_vanilla_m = np.arange(1, len(m_data_vanilla_sorted) + 1) / len(m_data_vanilla_sorted)

    path_vanilla_p = os.path.join(path_to_data, "vanilla", "plots","mc_episode_position_res.npy")
    p_data_vanilla = np.load(path_vanilla_p)
    p_data_vanilla_sorted = np.sort(p_data_vanilla)
    cdf_vanilla_p = np.arange(1, len(p_data_vanilla_sorted) + 1) / len(p_data_vanilla_sorted)

    path_vanilla_v = os.path.join(path_to_data, "vanilla", "plots","mc_episode_velocity_res.npy")
    v_data_vanilla = np.load(path_vanilla_v)
    v_data_vanilla_sorted = np.sort(v_data_vanilla)
    cdf_vanilla_v = np.arange(1, len(v_data_vanilla_sorted) + 1) / len(v_data_vanilla_sorted)

    #pre-trained SAC
    path_pt_r = os.path.join(path_to_data, "pt", "plots","mc_episode_rewards.npy")
    r_data_pt = np.load(path_pt_r)
    r_data_pt_sorted = np.sort(r_data_pt)
    cdf_pt = np.arange(1, len(r_data_pt_sorted) + 1) / len(r_data_pt_sorted)

    path_pt_m = os.path.join(path_to_data, "pt", "plots","mc_episode_final_mass.npy")
    m_data_pt = np.load(path_pt_m)
    m_data_pt_sorted = np.sort(m_data_pt)
    cdf_pt_m = np.arange(1, len(m_data_pt_sorted) + 1) / len(m_data_pt_sorted)

    path_pt_p = os.path.join(path_to_data, "pt", "plots","mc_episode_position_res.npy")
    p_data_pt = np.load(path_pt_p)
    p_data_pt_sorted = np.sort(p_data_pt)
    cdf_pt_p = np.arange(1, len(p_data_pt_sorted) + 1) / len(p_data_pt_sorted)

    path_pt_v = os.path.join(path_to_data, "pt", "plots","mc_episode_velocity_res.npy")
    v_data_pt = np.load(path_pt_v)
    v_data_pt_sorted = np.sort(v_data_pt)
    cdf_pt_v = np.arange(1, len(v_data_pt_sorted) + 1) / len(v_data_pt_sorted)

    plt.figure(figsize=(14, 10))
    
    # Rewards CDF
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(r_data_vanilla_sorted, cdf_vanilla, linewidth=2.5, label=f'Vanilla SAC (avg: {np.mean(r_data_vanilla):.2f})')
    ax1.plot(r_data_pt_sorted, cdf_pt, linewidth=2.5, label=f'Pre-trained SAC (avg: {np.mean(r_data_pt):.2f})')
    ax1.set_xlabel('Reward', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)
    
    # Final Mass CDF
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(m_data_vanilla_sorted, cdf_vanilla_m, linewidth=2.5, label=f'Vanilla SAC (avg: {np.mean(m_data_vanilla):.2f})')
    ax2.plot(m_data_pt_sorted, cdf_pt_m, linewidth=2.5, label=f'Pre-trained SAC (avg: {np.mean(m_data_pt):.2f})')
    ax2.set_xlabel('Final Mass (nd)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)
    
    # Position Residual CDF
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(p_data_vanilla_sorted, cdf_vanilla_p, linewidth=2.5, label=f'Vanilla SAC (avg: {np.mean(p_data_vanilla):.2f})')
    ax3.plot(p_data_pt_sorted, cdf_pt_p, linewidth=2.5, label=f'Pre-trained SAC (avg: {np.mean(p_data_pt):.2f})')
    ax3.set_xlabel('Terminal Position Residual (nd)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=11)
    
    # Velocity Residual CDF
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(v_data_vanilla_sorted, cdf_vanilla_v, linewidth=2.5, label=f'Vanilla SAC (avg: {np.mean(v_data_vanilla):.2f})')
    ax4.plot(v_data_pt_sorted, cdf_pt_v, linewidth=2.5, label=f'Pre-trained SAC (avg: {np.mean(v_data_pt):.2f})')
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
    ax1.hist(r_data_vanilla, bins=30, linewidth=1.5, alpha=0.7, label=f'Vanilla SAC (avg: {np.mean(r_data_vanilla):.2f})', density=False)
    ax1.hist(r_data_pt, bins=30, linewidth=1.5, alpha=0.7, label=f'Pre-trained SAC (avg: {np.mean(r_data_pt):.2f})', density=False)
    ax1.set_xlabel('Reward', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)
    
    # Final Mass PDF
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(m_data_vanilla, bins=30, linewidth=1.5, alpha=0.7, label=f'Vanilla SAC (avg: {np.mean(m_data_vanilla):.2f})', density=False)
    ax2.hist(m_data_pt, bins=30, linewidth=1.5, alpha=0.7, label=f'Pre-trained SAC (avg: {np.mean(m_data_pt):.2f})', density=False)
    ax2.set_xlabel('Final Mass (nd)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)
    
    # Position Residual PDF
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(p_data_vanilla, bins=30, linewidth=1.5, alpha=0.7, label=f'Vanilla SAC (avg: {np.mean(p_data_vanilla):.2f})', density=False)
    ax3.hist(p_data_pt, bins=30, linewidth=1.5, alpha=0.7, label=f'Pre-trained SAC (avg: {np.mean(p_data_pt):.2f})', density=False)
    ax3.set_xlabel('Terminal Position Residual (nd)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=11)
    
    # Velocity Residual PDF
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(v_data_vanilla, bins=30, linewidth=1.5, alpha=0.7, label=f'Vanilla SAC (avg: {np.mean(v_data_vanilla):.2f})', density=False)
    ax4.hist(v_data_pt, bins=30, linewidth=1.5, alpha=0.7, label=f'Pre-trained SAC (avg: {np.mean(v_data_pt):.2f})', density=False)
    ax4.set_xlabel('Terminal Velocity Residual (nd)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=13)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_data, "mc_pdf.png"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    path_to_data = "C:\\Users\\micha\\MSI_Data\\Masters_Thesis\\z_script_output\\runs_for_record\\agent_evals"
    path_to_data = os.path.abspath(path_to_data)
    misc_plotting(path_to_data)