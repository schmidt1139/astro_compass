import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from astro_compass.core.ephemeris import Ephemeris as Ephemeris
from astro_compass.core.rollouts import RolloutData
from astro_compass.core.spacecraft import Spacecraft
from astro_compass.utils.env_utils import gen_rl_environment
from astro_compass.utils.log_utils import (
    read_toml_config_file,
)
from astro_compass.utils.model_utils import get_model
from astro_compass.utils.path_utils import (
    CONFIG_ROOT,
    DATA_ROOT,
    RUNS_ROOT,
    get_run_paths,
)
from astro_compass.utils.state_vector_utils import (
    calc_OE_from_cart,
    cartesian_to_polar,
    polar_to_cartesian,
)

plt.style.use("data/support_files/light_paper.mplstyle")
print("GPU available: ", torch.cuda.is_available())


def main():
    seed_in = 42
    params_path = os.path.join(CONFIG_ROOT, "SAC_training_TBT_config.toml")
    params = read_toml_config_file(params_path)
    env = gen_rl_environment(params)

    model_id = "20251217_142819"
    paths = get_run_paths(RUNS_ROOT, model_id)
    model = get_model(params, env, seed_in, paths["path_SAC_model"])

    num_rollouts = 1

    # save the rollout to the
    # model_id/rollouts/rollout_data.pkl
    path_rollouts = os.path.join(paths["path_output"], "rollouts_H")
    os.makedirs(path_rollouts, exist_ok=True)

    ephems_dir = os.path.join(
        DATA_ROOT,
        "pre-training-data",
        "TBT",
        "pickle",
    )
    ephems = os.listdir(ephems_dir)

    for i in range(num_rollouts):
        # Load the hamiltonian trajectories

        with open(os.path.join(ephems_dir, ephems[i]), "rb") as f:
            ephem = pickle.load(f)

        X_target = np.array(
            [
                ephem.arr_x[-1],
                ephem.arr_y[-1],
                ephem.arr_vx[-1],
                ephem.arr_vy[-1],
            ]
        )
        mass_initial = 3366.0
        OE = calc_OE_from_cart(*X_target, params["mu"])
        a_target, e_target, w_target, theta = OE
        obs, info = env.reset()

        # update params based on hamiltonian IC and FC
        state = np.array(
            [
                ephem.arr_x[0],
                ephem.arr_y[0],
                ephem.arr_vx[0],
                ephem.arr_vy[0],
                mass_initial,
                a_target,
                e_target,
                w_target,
            ]
        )
        env.T_target = 2 * np.pi * (a_target**3 / params["mu"]) ** 0.5

        env.set_state(state)

        rollout_data, eph = rollout(
            env,
            model,
            params,
            obs,
            info,
        )

        env.close()

        path_rollout_data = os.path.join(path_rollouts, f"rollout_data_{i}.pkl")
        with open(path_rollout_data, "wb") as f:
            pickle.dump((rollout_data, eph), f)


def rollout(
    env,
    model,
    params,
    obs,
    info,
):
    rollout_data = RolloutData()
    eph = Ephemeris()
    count_step = 0
    flag_continue = True
    terminated = False
    truncated = False
    while flag_continue:
        # step the env
        action, hidden_state = model.predict(obs, deterministic=True)
        obs_dict = env.decode_observation(obs)

        # compute the cartesian state from observation
        theta = np.arctan2(obs_dict["sin_eta"], obs_dict["cos_eta"])

        x, y, vx, vy = polar_to_cartesian(
            obs_dict["r_nd"],
            theta,
            obs_dict["v_r_nd"],
            obs_dict["v_eta_nd"],
        )

        throttle = action[0]
        alpha_x = action[1]
        alpha_y = action[2]

        # dim state
        t_i = info["Elapsed time"]
        t_i_days = t_i / (3600 * 24)
        x_i = x * params["l_star"]
        y_i = y * params["l_star"]
        vx_i = vx * params["l_star"] / params["t_star"]
        vy_i = vy * params["l_star"] / params["t_star"]
        m_i = obs_dict["m_nd"] * params["m_star"]
        # log data to ephemeris
        eph.add_data(t_i, x_i, y_i, vx_i, vy_i, m_i, alpha_x, alpha_y, throttle)

        # create polar state, create a temp SC object and calc OE
        r_i, theta_i, rdot_i, vtheta_i = cartesian_to_polar(x_i, y_i, vx_i, vy_i)
        SC = Spacecraft(
            r_i,
            theta_i,
            rdot_i,
            vtheta_i,
            m_i,
            params["max_T"],
            params["ISP"],
        )
        arr_OE = SC.calc_Planar_OE(0.0, 0.0, 0.0, 0.0, params["mu"])

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs_dict = env.decode_observation(next_obs)
        reward_mass_component = info.get("reward_mass_component", 0.0)
        reward_distance_component = info.get("reward_distance_component", 0.0)

        count_step = count_step + 1

        # log data
        data = {
            "obs": obs_dict,
            "action": action,
            "reward": reward,
            "next_obs": next_obs_dict,
            "done": terminated or truncated,
            "info": info,
        }
        rollout_data.add_step(data)

        if terminated or truncated:
            break

    return rollout_data, eph


if __name__ == "__main__":
    main()
