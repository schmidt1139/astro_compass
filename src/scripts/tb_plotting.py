import matplotlib.pyplot as plt
import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from scipy.ndimage import uniform_filter1d


def smooth_data(data, window_size=10):
    """Smooth data using uniform filter (moving average)."""
    return uniform_filter1d(data, size=window_size, mode='nearest')


def misc_plotting():
    """Generate and save plots for pre-training losses and rewards.

    Args:
        arr_iters (list): List of iteration numbers.
        arr_actor_loss_pt (list): List of actor loss values during pre-training.
        arr_critic_loss_pt (list): List of critic loss values during pre-training.
        path_plots (str): Directory path to save the plots.
    """
    import matplotlib.pyplot as plt
    import os

    print("Misc Plotting Script")

    plt.style.use("data/support_files/light_paper.mplstyle")

    path_root = "C:\\Users\\micha\\MSI_Data\\Masters_Thesis\\z_script_output\\runs_for_record\\Dec16\\tbplots"
    path_pre_train_actor = "C:\\Users\\micha\\MSI_Data\\Masters_Thesis\\z_script_output\\Dec08\\curiosity\\Dec08\\pre_train\\SAC_training_TBR_polar20251208_192218"
    path_root = os.path.abspath(path_root)
    path_vanilla_pt0 = os.path.join(path_root, "pt0" );
    path_vanilla_pt10k = os.path.join(path_root, "pt10k" );
    path_vanilla_pt20k = os.path.join(path_root, "pt20k" );
    path_vanilla_pt310k = os.path.join(path_root, "pt310k" );
    path_vanilla_pt760k = os.path.join(path_root, "pt760k" );

    print(f"Path to tb data: {path_root}")

    # no pt
    ea_pt0 = event_accumulator.EventAccumulator(
        path_vanilla_pt0,
        size_guidance={
            event_accumulator.SCALARS: 0,   # 0 = load all
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
        },
    )
    ea_pt0.Reload()  # actually loads the data from disk

    # 10k pt
    ea_pt10k = event_accumulator.EventAccumulator(
        path_vanilla_pt10k,
        size_guidance={
            event_accumulator.SCALARS: 0,   # 0 = load all
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
        },
    )
    ea_pt10k.Reload()  # actually loads the data from disk

    # 20k pt
    ea_pt20k = event_accumulator.EventAccumulator(
        path_vanilla_pt20k,
        size_guidance={
            event_accumulator.SCALARS: 0,   # 0 = load all
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
        },
    )
    ea_pt20k.Reload()  # actually loads the data from disk

    # 310k pt
    ea_pt310k = event_accumulator.EventAccumulator(
        path_vanilla_pt310k,
        size_guidance={
            event_accumulator.SCALARS: 0,   # 0 = load all
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
        },
    )
    ea_pt310k.Reload()  # actually loads the data from disk

    # 760k pt
    ea_pt760k = event_accumulator.EventAccumulator(
        path_vanilla_pt760k,
        size_guidance={
            event_accumulator.SCALARS: 0,   # 0 = load all
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
        },
    )
    ea_pt760k.Reload()  # actually loads the data from disk




    print("Loaded data from TB")
    
    # Print available scalars to see what data we have

    fig, ax = plt.subplots()

    # Define runs with labels and colors
    runs = [
        (ea_pt0, "pt0", "C0"),
        (ea_pt10k, "pt10k", "C1"),
        (ea_pt20k, "pt20k", "C2"),
        (ea_pt310k, "pt310k", "C3"),
        (ea_pt760k, "pt760k", "C4"),
    ]
    
    # Extract and plot reward data from all runs
    print("\n=== Reward Crossing -1 Report ===")
    for ea, label, color in runs:
        if "rollout/ep_rew_mean" in ea.Tags()["scalars"]:
            reward_data = ea.Scalars("rollout/ep_rew_mean")
            steps = [event.step for event in reward_data]
            rewards = [event.value for event in reward_data]
            rewards = smooth_data(rewards, window_size=100)
            
            # Find when reward crosses -0.5
            crossing_step = None
            for i in range(len(rewards)):
                if rewards[i] > -1:
                    crossing_step = steps[i]
                    break
            
            if crossing_step is not None:
                print(f"{label}: crossed -1 at step {crossing_step}")
            else:
                print(f"{label}: never crossed -1 (max reward: {max(rewards):.4f})")
            
            # Plot reward time history
            ax.plot(steps, rewards, linewidth=1, label=label, color=color)

        
    # Save figure
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Reward per Episode")
    ax.set_ylim([-3, 0])
    ax.grid(True, alpha=0.3)
    ax.legend()
    path_to_save = path_root
    fig.savefig(os.path.join(path_to_save, "reward_time_history.png"), dpi=300, bbox_inches="tight")
    print(f"\nReward plot saved to {os.path.join(path_to_save, 'reward_time_history.png')}")

    #----------------------------------------------------------------------------------------------------

    #Actor losses
    df_actor_loss = pd.read_csv( os.path.join(path_pre_train_actor, "actor_losses.csv") )
    arr_actor_loss_pt = df_actor_loss["actor_loss"].tolist()
    arr_iters = list( range(1, len(arr_actor_loss_pt)+1) )
    arr_iters = [x * 1_000 for x in arr_iters]  # assuming batch size of 1_000 for pre-training

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # Define runs with labels and colors
    runs = [
        (ea_pt0, "pt0", "C0"),
        (ea_pt10k, "pt10k", "C1"),
        (ea_pt20k, "pt20k", "C2"),
        (ea_pt310k, "pt310k", "C3"),
        (ea_pt760k, "pt760k", "C4"),
    ]
    
    # Extract and plot actor loss data from all runs
    for ea, label, color in runs:
        if "train/actor_loss" in ea.Tags()["scalars"]:
            actor_loss_data = ea.Scalars("train/actor_loss")
            steps = [event.step for event in actor_loss_data]
            actor_losses = [event.value for event in actor_loss_data]
            actor_losses = smooth_data(actor_losses, window_size=100)
            ax1.plot(steps, actor_losses, linewidth=1, label=label, color=color)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Actor Network Loss")
    ax1.set_ylim([0, 1.25])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    #----------------------------------------------------------------------------------------------------

    #Critic losses
    df_critic_loss = pd.read_csv( os.path.join(path_pre_train_actor, "critic_losses.csv") )
    arr_critic_loss_pt = df_critic_loss["critic_loss"].tolist()
    arr_iters = list( range(1, len(arr_critic_loss_pt)+1) )
    arr_iters = [x * 1_000 for x in arr_iters]  # assuming batch size of 1_000 for pre-training

    # Extract and plot critic loss data from all runs
    for ea, label, color in runs:
        if "train/critic_loss" in ea.Tags()["scalars"]:
            critic_loss_data = ea.Scalars("train/critic_loss")
            steps = [event.step for event in critic_loss_data]
            critic_losses = [event.value for event in critic_loss_data]
            critic_losses = smooth_data(critic_losses, window_size=100)
            ax2.semilogy(steps, critic_losses, linewidth=1, label=label, color=color)

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Critic Network Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    path_to_save = path_root
    fig.savefig(os.path.join(path_to_save, "actor_critic_loss_time_history.png"), dpi=300, bbox_inches="tight")
    print(f"\nCritic loss plot saved to {os.path.join(path_to_save, 'actor_critic_loss_time_history.png')}")





if __name__ == "__main__":
    misc_plotting()