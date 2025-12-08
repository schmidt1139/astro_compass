import os
import random

import torch
from core.ephemeris_v2 import Ephemeris_v2 as Ephemeris
from core.process_single_trajectory import process_single_trajectory
from pretrain_utils import generate_env, generate_paths
from stable_baselines3 import SAC as SB3_SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from utils.env_utils import gen_rl_environment
from utils.eval_utils import mc_evaluate_agent, plot_log_mc_results
from utils.log_utils import (
    log,
    read_toml_config_file,
    write_config_file,
    write_log_to_file,
)
from utils.path_utils import PROJECT_ROOT
from utils.plotting_utils import plot_SAC_training_TBR_polar
from utils.rl_utils import (
    RewardLoggerCallback,
    rollout_model,
)


def main(params, seed_in=42):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    random.seed(seed_in)

    # initialize the training and evaluation environments
    env, eval_env, pre_train_env, single_env = generate_env(params, seed_in)

    # paths
    path_output, path_SAC_model, path_plots = generate_paths(params)
    params["path_plots"] = path_plots

    # reset the environment
    env.reset()

    model = SB3_SAC.load(
        params["path_SAC_model_load"],
        env=env,
        device=params.get("eval_device", "cpu"),
        seed=seed_in,
        tensorboard_log=path_output,
    )  # Use path_output so SB3 creates SAC_1/ subdirectory

    # report number of trainable parameters
    test_log = []
    num_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    test_log = log(
        f"Number of trainable parameters in the model: {num_params}",
        test_log,
        True,
    )

    callback = RewardLoggerCallback(print_freq=params["print_freq"])
    # Eval callback: saves best model by mean reward on eval_env
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=path_output,
        log_path=path_output,
        eval_freq=params["eval_freq"],  # adjust frequency
        n_eval_episodes=params["n_eval_episodes"],  # episodes per evaluation
        deterministic=True,
        render=False,
        verbose=0,
    )
    callback_list = CallbackList([eval_callback, callback])

    # optionally generate hamiltonian trajectory
    if params.get("flag_gen_H_traj", False):
        test_log = log(
            "Generating Hamiltonian trajectory for comparison...",
            test_log,
            True,
        )
        params["data_path"] = path_output
        params["scenario_index"] = 0
        params["flag_plot_traj"] = False
        results = process_single_trajectory(params)
        ephem_path = results[1]
        ephem_H = Ephemeris()
        try:
            ephem_H.read_from_file(ephem_path)
        except Exception as e:
            test_log = log(
                "Error generating Hamiltonian trajectory file: " + str(e),
                test_log,
                True,
            )
            params["flag_gen_H_traj"] = False

    rollout_env = gen_rl_environment(params)
    rollout_env.seed(seed_in)

    test_log, eph, rollout_data = rollout_model(rollout_env, params, model, test_log)

    # Monte Carlo evaluation
    mc_results = mc_evaluate_agent(params)

    plot_log_mc_results(mc_results, test_log, params)

    (
        arr_episode_numbers,
        arr_episode_rs,
        arr_position_res,
        arr_velocity_res,
        arr_m,
        list_pos_residuals,
        list_vel_residuals,
    ) = mc_results

    # render training plots
    test_log = log("Rendering training plots...", test_log, True)
    plot_SAC_training_TBR_polar(
        rollout_data,
        path_plots,
        eph,
        params,
        rollout_env,
        arr_episode_numbers=arr_episode_numbers,
        arr_episode_rs=arr_episode_rs,
        ephem_H=ephem_H if params.get("flag_gen_H_traj", False) else None,
    )

    env.close()

    # save ephemeris to file
    eph.write_to_file(
        os.path.join(path_output, "SAC_Test_Traj_Ephem.txt"),
        mod_vector_write_frequency=1,
    )

    test_log = log("Complete!", test_log, True)
    test_log = log("Plots saved to: " + path_output, test_log, True)

    # save log to file
    write_log_to_file(os.path.join(path_output, "SAC_Training_Log.txt"), test_log)

    # write config to output dir
    write_config_file(params, os.path.join(path_output, "SAC_Training_Config.txt"))

    print("\n\n\n")


if __name__ == "__main__":
    config_toml = "evaluate_agent_config.toml"
    path_config = os.path.join(PROJECT_ROOT, "data", "config", config_toml)
    params = read_toml_config_file(path_config)
    params["config_toml"] = config_toml

    main(params)
