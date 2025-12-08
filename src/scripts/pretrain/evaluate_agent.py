import os
import random
from pathlib import Path

# disable cuda
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from core.ephemeris_v2 import Ephemeris_v2 as Ephemeris
from core.process_single_trajectory import process_single_trajectory
from stable_baselines3 import SAC as SB3_SAC
from utils.config_utils import load_config, write_config_sources
from utils.env_utils import gen_rl_environment
from utils.eval_utils import mc_evaluate_agent, plot_log_mc_results
from utils.log_utils import (
    log,
    write_config_file,
    write_log_to_file,
)
from utils.plotting_utils import plot_SAC_training_TBR_polar
from utils.pretrain_utils import generate_env, generate_paths
from utils.rl_utils import _flatten_config_params, rollout_model


def main(config, seed_in=42, config_meta=None):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    random.seed(seed_in)

    paths_cfg = config["paths"]
    model_cfg = config["model"]
    general_cfg = config.get("general", {})

    env, _, _, _ = generate_env(config, seed_in)

    path_output, path_SAC_model, path_plots = generate_paths(paths_cfg)
    paths_cfg["path_plots"] = path_plots
    flat_params = _flatten_config_params(config)
    flat_params["path_plots"] = path_plots

    # reset the environment
    env.reset()

    model = SB3_SAC.load(
        paths_cfg["path_SAC_model_load"],
        env=env,
        device=model_cfg.get("eval_device", "cpu"),
        seed=seed_in,
        tensorboard_log=path_output,
    )

    # report number of trainable parameters
    test_log = []
    num_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    test_log = log(
        f"Number of trainable parameters in the model: {num_params}",
        test_log,
        True,
    )

    # optionally generate hamiltonian trajectory
    ephem_H = None

    if general_cfg.get("flag_gen_H_traj", False):
        test_log = log(
            "Generating Hamiltonian trajectory for comparison...",
            test_log,
            True,
        )
        flat_params["data_path"] = path_output
        flat_params["scenario_index"] = 0
        flat_params["flag_plot_traj"] = False
        results = process_single_trajectory(flat_params)
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
            general_cfg["flag_gen_H_traj"] = False

    rollout_env = gen_rl_environment(config)
    rollout_env.seed(seed_in)

    test_log, eph, rollout_data = rollout_model(rollout_env, config, model, test_log)

    # Monte Carlo evaluation
    mc_results = mc_evaluate_agent(config)

    plot_log_mc_results(mc_results, test_log, config)

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
        flat_params,
        rollout_env,
        arr_episode_numbers=arr_episode_numbers,
        arr_episode_rs=arr_episode_rs,
        ephem_H=ephem_H if general_cfg.get("flag_gen_H_traj", False) else None,
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
    write_config_file(flat_params, os.path.join(path_output, "SAC_Training_Config.txt"))

    if config_meta:
        write_config_sources(config_meta, Path(path_output))

    print("\n\n\n")


if __name__ == "__main__":
    base_files = [
        "common.toml",
        "envs.toml",
        "models.toml",
        "training.toml",
    ]
    experiment_file = "experiments/eval_default.toml"
    config, meta = load_config(base_files=base_files, experiment_file=experiment_file)

    main(config, seed_in=0, config_meta=meta)
