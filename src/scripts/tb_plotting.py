import matplotlib.pyplot as plt
import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


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

    path_root = "C:\\Users\\micha\\MSI_Data\\Masters_Thesis\\z_script_output\\runs_for_record\\tb_plots"
    path_pre_train_actor = "C:\\Users\\micha\\MSI_Data\\Masters_Thesis\\z_script_output\\Dec08\\curiosity\\Dec08\\pre_train\\SAC_training_TBR_polar20251208_192218"
    path_root = os.path.abspath(path_root)
    path_vanilla_100k = os.path.join(path_root, "100k")
    path_vanilla_1mil = os.path.join(path_root, "1mil")
    path_pt_1mil = os.path.join(path_root, "1mil_rb")

    print(f"Path to tb data: {path_vanilla_100k}")

    # 1mil run
    ea_1mil = event_accumulator.EventAccumulator(
        path_vanilla_1mil,
        size_guidance={
            event_accumulator.SCALARS: 0,   # 0 = load all
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
        },
    )
    ea_1mil.Reload()  # actually loads the data from disk

    ea_1mil_rb = event_accumulator.EventAccumulator(
        path_pt_1mil,
        size_guidance={
            event_accumulator.SCALARS: 0,   # 0 = load all
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
        },
    )
    ea_1mil_rb.Reload()  # actually loads the data from disk


    print("Loaded data from TB")
    
    # Print available scalars to see what data we have

    fig, ax = plt.subplots()

    # Extract reward data
    if "rollout/ep_rew_mean" in ea_1mil.Tags()["scalars"]:
        reward_data = ea_1mil.Scalars("rollout/ep_rew_mean")
        steps = [event.step for event in reward_data]
        rewards = [event.value for event in reward_data]
        
        # Plot reward time history
        ax.plot(steps, rewards, linewidth=1.5, label="1mil Iters")


    # Extract reward data
    if "rollout/ep_rew_mean" in ea_1mil_rb.Tags()["scalars"]:
        reward_data = ea_1mil_rb.Scalars("rollout/ep_rew_mean")
        steps = [event.step for event in reward_data]
        rewards = [event.value for event in reward_data]
        
        # Plot reward time history
        ax.plot(steps, rewards, linewidth=1.5, label="1mil + 200k PT Iters")

        
    # Save figure
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Reward per Episode")
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

    offset = 1000

    # Extract actor loss data
    if "train/actor_loss" in ea_1mil.Tags()["scalars"]:
        actor_loss_data = ea_1mil.Scalars("train/actor_loss")
        steps = [event.step for event in actor_loss_data]
        actor_losses = [event.value for event in actor_loss_data]
        steps = [s + offset for s in steps]
        # Plot actor loss time history on left y-axis
        ax1.plot(steps, actor_losses, linewidth=1.5, label="1mil Iters", color='C0')

    # Create second y-axis
    print("Event steps in 1mil:", len(steps))
    print("Min step:", min(steps), "Max step:", max(steps))
    # Extract actor loss data
    print("Events in 1mil_rb:", ea_1mil_rb.Tags()["scalars"])
    if "train/actor_loss" in ea_1mil_rb.Tags()["scalars"]:
        actor_loss_data = ea_1mil_rb.Scalars("train/actor_loss")
        steps = [event.step for event in actor_loss_data]
        actor_losses = [event.value for event in actor_loss_data]
        steps = [s + offset + 200_000 for s in steps]
        # Plot actor loss time history on right y-axis
        ax1.plot(steps, actor_losses, linewidth=1.5, label="1mil + 200k PT Iters", color='C1')

    ax1.plot(arr_iters, arr_actor_loss_pt, linewidth=1.5, label="Pre-Training", color='C2')
    print("Event steps in 1mil:", len(steps))
    print("Min step:", min(steps), "Max step:", max(steps))
    # Save figure
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Actor Network Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')

    #----------------------------------------------------------------------------------------------------

    #Critic losses
    df_critic_loss = pd.read_csv( os.path.join(path_pre_train_actor, "critic_losses.csv") )
    arr_critic_loss_pt = df_critic_loss["critic_loss"].tolist()
    arr_iters = list( range(1, len(arr_critic_loss_pt)+1) )
    arr_iters = [x * 1_000 for x in arr_iters]  # assuming batch size of 1_000 for pre-training


    offset = 1000

    # Extract actor loss data
    if "train/critic_loss" in ea_1mil.Tags()["scalars"]:
        actor_loss_data = ea_1mil.Scalars("train/critic_loss")
        steps = [event.step for event in actor_loss_data]
        actor_losses = [event.value for event in actor_loss_data]
        steps = [s + offset for s in steps]
        # Plot actor loss time history on left y-axis
        ax2.semilogy(steps, actor_losses, linewidth=1.5, label="1mil Iters", color='C0')

    # Create second y-axis
    print("Event steps in 1mil:", len(steps))
    print("Min step:", min(steps), "Max step:", max(steps))
    # Extract actor loss data
    print("Events in 1mil_rb:", ea_1mil_rb.Tags()["scalars"])
    if "train/critic_loss" in ea_1mil_rb.Tags()["scalars"]:
        actor_loss_data = ea_1mil_rb.Scalars("train/critic_loss")
        steps = [event.step for event in actor_loss_data]
        actor_losses = [event.value for event in actor_loss_data]
        steps = [s + offset + 200_000 for s in steps]
        # Plot actor loss time history on right y-axis
        ax2.semilogy(steps, actor_losses, linewidth=1.5, label="1mil + 200k PT Iters", color='C1')

    ax2.semilogy(arr_iters, arr_actor_loss_pt, linewidth=1.5, label="Pre-Training", color='C2')
    print("Event steps in 1mil:", len(steps))
    print("Min step:", min(steps), "Max step:", max(steps))
    # Save figure
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Critic Network Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    path_to_save = path_root
    fig.savefig(os.path.join(path_to_save, "actor_critic_loss_time_history.png"), dpi=300, bbox_inches="tight")
    print(f"\nCritic loss plot saved to {os.path.join(path_to_save, 'actor_critic_loss_time_history.png')}")





if __name__ == "__main__":
    misc_plotting()