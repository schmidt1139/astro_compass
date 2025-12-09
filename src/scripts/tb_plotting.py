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
    path_root = os.path.abspath(path_root)
    path_vanilla_100k = os.path.join(path_root, "100k")
    path_vanilla_1mil = os.path.join(path_root, "1mil")

    print(f"Path to tb data: {path_vanilla_100k}")

    # 100k run
    ea_100k = event_accumulator.EventAccumulator(
        path_vanilla_100k,
        size_guidance={
            event_accumulator.SCALARS: 0,   # 0 = load all
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
        },
    )
    ea_100k.Reload()  # actually loads the data from disk

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



    print("Loaded data from TB")
    
    # Print available scalars to see what data we have
    print("\nAvailable scalars:")
    for scalar_name in ea_100k.Tags()["scalars"]:
        print(f"  - {scalar_name}")

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract reward data
    if "rollout/ep_rew_mean" in ea_100k.Tags()["scalars"]:
        reward_data = ea_100k.Scalars("rollout/ep_rew_mean")
        steps = [event.step for event in reward_data]
        rewards = [event.value for event in reward_data]
        
        # Plot reward time history
        ax.plot(steps, rewards, linewidth=1.5, label="100k Mean Reward")


    # Extract reward data
    if "rollout/ep_rew_mean" in ea_1mil.Tags()["scalars"]:
        reward_data = ea_1mil.Scalars("rollout/ep_rew_mean")
        steps = [event.step for event in reward_data]
        rewards = [event.value for event in reward_data]
        
        # Plot reward time history
        ax.plot(steps, rewards, linewidth=1.5, label="1mil Mean Reward")

    
        
    # Save figure
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Reward per Episode")
    ax.set_title("Reward Time History")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path_to_save = path_root
    fig.savefig(os.path.join(path_to_save, "reward_time_history.png"), dpi=300, bbox_inches="tight")
    print(f"\nReward plot saved to {os.path.join(path_to_save, 'reward_time_history.png')}")
    plt.show()



if __name__ == "__main__":
    misc_plotting()