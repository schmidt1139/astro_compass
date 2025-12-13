import os

import matplotlib.pyplot as plt
import torch

from astro_compass.constants.constants import Constants
from astro_compass.core.ephemeris import Ephemeris as Ephemeris
from astro_compass.core.hamiltonian_control import Hamiltonian_Controller_TBT
from astro_compass.core.spacecraft import Spacecraft
from astro_compass.utils.env_utils import gen_rl_environment
from astro_compass.utils.log_utils import (
    read_toml_config_file,
    write_config_file,
)
from astro_compass.utils.model_utils import get_model
from astro_compass.utils.path_utils import (
    CONFIG_ROOT,
    RUNS_ROOT,
    get_run_paths,
)
from astro_compass.utils.plotting_utils import SACRolloutData, plot_SAC_training
from astro_compass.utils.state_vector_utils import cartesian_to_polar

plt.style.use("data/support_files/light_paper.mplstyle")
print("GPU available: ", torch.cuda.is_available())


def eval_TBT_agent(env, model, params, path_output, path_ephems):
    sma_t_i = Constants.SMA_EARTH

    # Optionally, test the trained agent
    obs, info = env.reset(seed=params.get("seed_traj", 42))
    eph = Ephemeris()  # create new ephemeris object

    rollout_data1 = SACRolloutData()
    sum_reward = 0.0

    # After training:

    print("Timesteps:", model.num_timesteps)
    print("Training complete")

    print("Plotting test trajectory...")
    count_step = 0
    flag_continue = True
    terminated = False
    truncated = False

    # optionally generate hamiltonian trajectory off of ephem
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
            unwrapped_env, init_observation, info, input_TOF
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
            print(
                "Error generating Hamiltonian trajectory file: " + str(e),
                test_log,
                True,
            )
            params["flag_gen_H_traj"] = False

    while flag_continue:
        # step the env
        action, _states = model.predict(obs, deterministic=True)
        throttle = action[0]
        alpha_x = action[1]
        alpha_y = action[2]

        # dim state
        t_i = info["Elapsed time"]
        t_i_days = t_i / (3600 * 24)
        x_i = obs[0] * params["l_star"]
        y_i = obs[1] * params["l_star"]
        vx_i = obs[2] * params["l_star"] / params["t_star"]
        vy_i = obs[3] * params["l_star"] / params["t_star"]
        m_i = obs[4] * params["m_star"]

        # log data to ephemeris
        eph.add_data(t_i, x_i, y_i, vx_i, vy_i, m_i, alpha_x, alpha_y, throttle)

        # create polar state, create a temp SC object and calc OE
        r_i, theta_i, rdot_i, vtheta_i = cartesian_to_polar(x_i, y_i, vx_i, vy_i)
        SC = Spacecraft(
            r_i, theta_i, rdot_i, vtheta_i, m_i, params["max_T"], params["ISP"]
        )
        arr_OE = SC.calc_Planar_OE(0.0, 0.0, 0.0, 0.0, params["mu"])

        obs, reward, terminated, truncated, info = env.step(action)
        reward_mass_component = info.get("reward_mass_component", 0.0)
        reward_distance_component = info.get("reward_distance_component", 0.0)

        count_step = count_step + 1

        # log data
        rollout_data1.add_step(
            t_i_days,
            reward,
            throttle,
            alpha_x,
            alpha_y,
            obs[0],
            obs[1],
            obs[2],
            obs[3],
            arr_OE[0],
            sma_t_i,
            arr_OE[1],
            0.0,
            1.0,
            reward_mass_component,
            reward_distance_component,
        )

        if terminated or truncated:
            break

    print("Test trajectory complete")
    print("Steps taken: " + str(count_step))
    print("Total reward: " + str(rollout_data1.sum_reward))
    print("Final x: " + str(obs[0]) + " ")
    print("Final y: " + str(obs[1]) + " ")
    print("Final vx: " + str(obs[2]) + " ")
    print("Final vy: " + str(obs[3]) + " ")
    print("Final m: " + str(obs[4]) + " ")
    print("Final sma: " + str(arr_OE[0]) + " ")
    print("Final ecc: " + str(arr_OE[1]) + " ")
    print("terminated: " + str(terminated) + " ")
    print("truncated: " + str(truncated) + " ")

    # final env info
    for key, value in info.items():
        if key != "ODE Solution":
            print(f"{key}: {value}")

    # plot the results
    plot_SAC_training(
        rollout_data1,
        path_output,
        eph,
        ephem_H if ephem_H is not None else None,
    )

    env.close()

    # save ephemeris to file
    eph.write_to_file(
        os.path.join(path_output, "SAC_Test_Traj_Ephem.txt"),
        mod_vector_write_frequency=1,
    )

    print("Complete!")
    print("Plots saved to: " + path_output)

    # write config to output dir
    write_config_file(params, os.path.join(path_output, "SAC_Training_Config.txt"))


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
