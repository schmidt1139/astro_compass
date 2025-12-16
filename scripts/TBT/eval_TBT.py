import os

import matplotlib.pyplot as plt
import torch

from astro_compass.constants.constants import Constants
from astro_compass.core.ephemeris import Ephemeris as Ephemeris
from astro_compass.core.hamiltonian_control import Hamiltonian_Controller_TBT
from astro_compass.utils.env_utils import gen_rl_environment
from astro_compass.utils.log_utils import (
    read_toml_config_file,
)
from astro_compass.utils.model_utils import get_model
from astro_compass.utils.path_utils import (
    CONFIG_ROOT,
    RUNS_ROOT,
    get_run_paths,
)
from astro_compass.vis.rollout_plotter import RolloutPlotter

plt.style.use("data/support_files/light_paper.mplstyle")
print("GPU available: ", torch.cuda.is_available())


def eval_TBT_agent(env, model, params, path_output, path_ephems):
    sma_t_i = Constants.SMA_EARTH

    # Optionally, test the trained agent
    obs, info = env.reset(seed=params.get("seed_traj", 42))

    sum_reward = 0.0

    # After training:

    print("Timesteps:", model.num_timesteps)
    print("Training complete")

    # optionally generate hamiltonian trajectory off of ephem
    ephem_H = generate_hamiltonian_trajectory(
        env,
        params,
        path_output,
        path_ephems,
        obs,
        info,
    )

    rollout_data, eph = rollout_policy(
        env,
        model,
        params,
        sma_t_i,
        obs,
        info,
    )

    # plot the results
    vis = RolloutPlotter(rollout_data, path_output)
    vis.plot(eph, ephem_H if ephem_H is not None else None)

    env.close()

    # save ephemeris to file
    eph.write_to_file(
        os.path.join(path_output, "SAC_Test_Traj_Ephem.txt"),
        mod_vector_write_frequency=1,
    )

    print("Complete!")
    print("Plots saved to: " + path_output)


def generate_hamiltonian_trajectory(env, params, path_output, path_ephems, obs, info):
    if params.get("flag_gen_H_traj", False):
        print("Generating Hamiltonian trajectory for comparison...")
        params["data_path"] = path_output
        params["scenario_index"] = 0
        params["flag_plot_traj"] = False

        init_observation = []
        init_observation.append(obs[0] * params["l_star"] / 1000)
        init_observation.append(obs[1] * params["l_star"] / 1000)
        init_observation.append(obs[2] * params["l_star"] / params["t_star"] / 1000)
        init_observation.append(obs[3] * params["l_star"] / params["t_star"] / 1000)
        init_observation.append(obs[4] * params["m_star"])
        init_observation.append(Constants.MU_SUN)
        init_observation.append(Constants.SMA_EARTH / 1000)

        input_TOF = 1.1 * 365.25 * 24 * 60 * 60

        unwrapped_env = env.unwrapped

        H_controller = Hamiltonian_Controller_TBT(
            unwrapped_env,
            init_observation,
            info,
            input_TOF,
        )

        # modify parameters
        H_controller.eps_threshold = params.get("eps_final", 0.0004)

        # compute solution
        flag_solved, h_sol, eps, sol, h_log = H_controller.hamiltonian_solution_finder()

        ephem_H = Ephemeris()
        ephem_path = os.path.join(path_ephems, "Hamiltonian_Traj_Ephem.txt")

        if flag_solved:
            # write output ephemeris
            eph_out, arr_time, arr_u, arr_rho, arr_alpha_x, arr_alpha_y = (
                H_controller.generate_output_ephemeris(eph)
            )
            eph_out.write_to_file(ephem_path, mod_vector_write_frequency=1)

        try:
            print("Generated Hamiltonian trajectory for comparison...")
            ephem_H.read_from_file(ephem_path)
        except Exception as e:
            print("Error generating Hamiltonian trajectory file: " + str(e))
            params["flag_gen_H_traj"] = False
    return ephem_H


def main():
    seed_in = 42
    params_path = os.path.join(CONFIG_ROOT, "SAC_training_TBT_config.toml")
    params = read_toml_config_file(params_path)
    env = gen_rl_environment(params)
    time_tag = "20251213_182606"
    paths = get_run_paths(RUNS_ROOT, time_tag)
    model = get_model(params, env, seed_in, paths["path_SAC_model"])
    eval_TBT_agent(env, model, params, paths["path_output"], paths["path_ephems"])


if __name__ == "__main__":
    main()
