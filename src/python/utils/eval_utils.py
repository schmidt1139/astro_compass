
import random
import numpy as np
from tqdm import tqdm
from utils.env_utils import gen_rl_environment
from stable_baselines3 import SAC as SB3_SAC
from utils.rl_utils import rollout_model
from concurrent.futures import as_completed, ProcessPoolExecutor
from utils.log_utils import log
import matplotlib.pyplot as plt

def run_single_rollout(seed, params):

    rollout_env = gen_rl_environment(params)
    rollout_env.seed(seed)
    params["seed_traj"] = seed
    params["flag_report_live"] = False

    #load model
    model = SB3_SAC.load(
        params["path_SAC_model_load"],
        env=rollout_env,
        device=params.get("eval_device", "cpu"),
        seed=seed,
        verbose=0,
    )  # Use path_output so SB3 creates SAC_1/ subdirectory

    return rollout_model(rollout_env, params, model, [])

def mc_evaluate_agent(params):

    # monte carlo eval
    num_rollouts = params.get("num_rollouts", 10)
    num_workers = params.get("cores", 4)
    seeds = [random.randint(0, 10000) for _ in range(num_rollouts)]

    results = []
    arr_episode_numbers = []
    arr_episode_rs = []
    arr_position_res = []
    arr_velocity_res = []
    arr_m = []
    list_pos_residuals = []
    list_vel_residuals = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_single_rollout, s, params) for s in seeds]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Parallel rollouts"):

            #do not store ephem data here to save memory
            temp = f.result()
            temp_rollout_data = temp[2]

            results.append(temp_rollout_data)

    for idx, (rollout_data) in enumerate(results):
        #get final stats
        total_reward = rollout_data.arr_reward_tot[-1]
        arr_episode_numbers.append(idx + 1)
        arr_episode_rs.append(total_reward)
        arr_position_res.append(rollout_data.arr_position_res[-1])
        arr_velocity_res.append(rollout_data.arr_velocity_res[-1])
        arr_m.append(rollout_data.arr_mass[-1])
        list_pos_residuals.append(rollout_data.arr_position_res)
        list_vel_residuals.append(rollout_data.arr_velocity_res)

    return arr_episode_numbers, arr_episode_rs, arr_position_res, arr_velocity_res, arr_m, list_pos_residuals, list_vel_residuals

def plot_log_mc_results(mc_results, test_log, params):

    arr_episode_numbers, arr_episode_rs, arr_position_res, arr_velocity_res, arr_m, list_pos_residuals, list_vel_residuals = mc_results
    mean_r = sum(arr_episode_rs) / len(arr_episode_rs)
    percentile_95_r = np.percentile(arr_episode_rs, 95)
    percentile_5_r = np.percentile(arr_episode_rs, 5)

    mean_position_res = sum(arr_position_res) / len(arr_position_res)
    percentile_95_position_res = np.percentile(arr_position_res, 95)
    percentile_5_position_res = np.percentile(arr_position_res, 5)
    
    mean_velocity_res = sum(arr_velocity_res) / len(arr_velocity_res)
    percentile_95_velocity_res = np.percentile(arr_velocity_res, 95)
    percentile_5_velocity_res = np.percentile(arr_velocity_res, 5)

    mean_m = sum(arr_m) / len(arr_m)
    percentile_95_m = np.percentile(arr_m, 95)
    percentile_5_m = np.percentile(arr_m, 5)


    test_log = log("\n\nMonte Carlo Evaluation Results:", test_log, True)
    test_log = log(
        "Number of rollouts: " + str(len(arr_episode_numbers)), test_log, True
    )
    test_log = log("", test_log, True)
    test_log = log("Mean reward: " + str(mean_r), test_log, True)
    test_log = log("95 percentile reward: " + str(percentile_95_r), test_log, True)
    test_log = log("5 percentile reward: " + str(percentile_5_r), test_log, True)
    test_log = log("", test_log, True)
    test_log = log("Mean position residual (nd): " + str(mean_position_res), test_log, True)
    test_log = log("95 percentile position residual (nd): " + str(percentile_95_position_res), test_log, True)
    test_log = log("5 percentile position residual (nd): " + str(percentile_5_position_res), test_log, True)
    test_log = log("", test_log, True)
    test_log = log("Mean velocity residual (nd): " + str(mean_velocity_res), test_log, True)
    test_log = log("95 percentile velocity residual (nd): " + str(percentile_95_velocity_res), test_log, True)
    test_log = log("5 percentile velocity residual (nd): " + str(percentile_5_velocity_res), test_log, True)
    test_log = log("", test_log, True)
    test_log = log("Mean m (nd): " + str(mean_m), test_log, True)
    test_log = log("95 percentile m (nd): " + str(percentile_95_m), test_log, True)
    test_log = log("5 percentile m (nd): " + str(percentile_5_m), test_log, True)
    test_log = log("", test_log, True)

    #generate reward histogram
    plt.figure(figsize=(10, 6))
    plt.hist(arr_episode_rs, bins=20, color='blue', alpha=0.7)
    plt.title('Histogram of Episode Rewards')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt_path = params["path_plots"]
    plt.savefig(f"{plt_path}/mc_episode_rewards_histogram.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(arr_position_res, bins=20, color='blue', alpha=0.7)
    plt.title('Histogram of Episode Position Residuals')
    plt.xlabel('Position Residual')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt_path = params["path_plots"]
    plt.savefig(f"{plt_path}/mc_episode_position_residuals_histogram.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(arr_velocity_res, bins=20, color='blue', alpha=0.7)
    plt.title('Histogram of Episode Velocity Residuals')
    plt.xlabel('Velocity Residual')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt_path = params["path_plots"]
    plt.savefig(f"{plt_path}/mc_episode_velocity_residuals_histogram.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(arr_m, bins=20, color='blue', alpha=0.7)
    plt.title('Histogram of Episode Final Mass')
    plt.xlabel('Final Mass')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt_path = params["path_plots"]
    plt.savefig(f"{plt_path}/mc_episode_final_mass_histogram.png")
    plt.close()

    #plot position residuals over time for all episodes
    plt.figure(figsize=(10, 6))
    for i, pos_residuals in enumerate(list_pos_residuals):
        plt.plot(pos_residuals, alpha=0.5)

    plt.title('Position Residuals Over Time for ' + str(len(list_pos_residuals)) + ' Episodes')
    plt.xlabel('Time Step')
    plt.ylabel('Position Residual (nd)')
    plt.grid(True)
    plt_path = params["path_plots"]
    plt.savefig(f"{plt_path}/mc_position_residuals_over_time.png")
    plt.close()

    #plot velocity residuals over time for all episodes
    plt.figure(figsize=(10, 6))
    for i, vel_residuals in enumerate(list_vel_residuals):
        plt.plot(vel_residuals, alpha=0.5)
        
    plt.title('Velocity Residuals Over Time for ' + str(len(list_vel_residuals)) + ' Episodes')
    plt.xlabel('Time Step')
    plt.ylabel('Velocity Residual (nd)')
    plt.grid(True)
    plt_path = params["path_plots"]
    plt.savefig(f"{plt_path}/mc_velocity_residuals_over_time.png")
    plt.close()

    return test_log