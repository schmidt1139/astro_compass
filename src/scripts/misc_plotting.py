import matplotlib.pyplot as plt
import os
import pandas as pd

def misc_plotting(path_plots=None):
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
    print(f"Path to plots: {path_plots}")

    plt.style.use("data/support_files/light_paper.mplstyle")

    #load actor loss dataframe
    df_actor_loss = pd.read_csv( os.path.join(path_plots, "actor_losses.csv") )
    arr_actor_loss_pt = df_actor_loss["actor_loss"].tolist()
    arr_iters = list( range(1, len(arr_actor_loss_pt)+1) )
    arr_iters = [x * 1_000 for x in arr_iters]  # assuming batch size of 1_000 for pre-training

    df_critic_loss = pd.read_csv( os.path.join(path_plots, "critic_losses.csv") )
    arr_critic_loss_pt = df_critic_loss["critic_loss"].tolist()
    arr_iters = list( range(1, len(arr_critic_loss_pt)+1) )
    arr_iters = [x * 1_000 for x in arr_iters]  # assuming batch size of 1_000 for pre-training

    df_eval_rewards = pd.read_csv( os.path.join(path_plots, "eval_rewards.csv") )
    arr_eval_rewards = df_eval_rewards["mean_reward"].tolist()
    arr_eval_rewards_std = df_eval_rewards["std_reward"].tolist()
    arr_iters_r = list( range(1, len(arr_eval_rewards)+1) )
    arr_iters_r = [x * 10_000 for x in arr_iters_r]  # assuming batch size of 1_000 for pre-training
    arr_eval_rewards_upper = [m + s for m, s in zip(arr_eval_rewards, arr_eval_rewards_std)]
    arr_eval_rewards_lower = [m - s for m, s in zip(arr_eval_rewards, arr_eval_rewards_std)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    ax1.plot(arr_iters, arr_actor_loss_pt, label="Actor Loss")
    ax1.set_xlabel("Pre-training Steps")
    ax1.set_ylabel("Actor Loss")
    ax1.set_ylim([-2, 1])
    ax1.legend()
    ax1.grid(True)

    ax2.semilogy(arr_iters, arr_critic_loss_pt, label="Critic Loss")
    ax2.set_xlabel("Pre-training Steps")
    ax2.set_ylabel("Critic Loss")
    ax2.legend()
    ax2.grid(True)
    plt.savefig(os.path.join(path_plots, "actor_critic_loss_plot_pretrain.png"))
    plt.close()

    plt.figure()
    plt.plot(arr_iters_r, arr_eval_rewards, label="Mean Evaluation Reward")
    plt.fill_between(arr_iters_r, arr_eval_rewards_lower, arr_eval_rewards_upper, color='b', alpha=0.2, label="+/- 1 Std Dev")
    
    # Mark new maximum rewards with red dots
    max_reward = float('-inf')
    max_indices = []
    max_rewards = []
    max_iters = []
    for i, reward in enumerate(arr_eval_rewards):
        if reward > max_reward:
            max_reward = reward
            max_indices.append(i)
            max_rewards.append(reward)
            max_iters.append(arr_iters_r[i])
    
    plt.scatter(max_iters, max_rewards, color='red', s=10, zorder=5, label="New Max Reward")
    
    plt.xlabel("Pre-training Steps")
    plt.ylabel("Evaluation Rewards")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path_plots, "eval_rewards_plot_pretrain.png"))
    plt.close()

if __name__ == "__main__":
    path_to_data = "C:\\Users\\micha\\MSI_Data\\Masters_Thesis\\z_script_output\\Dec16\\pre_train\\SAC_training_TBR_polar20251216_213750"
    path_to_data = os.path.abspath(path_to_data)
    misc_plotting(path_to_data)