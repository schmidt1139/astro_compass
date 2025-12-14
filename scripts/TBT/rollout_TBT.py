import os
import pickle

import matplotlib.pyplot as plt
import torch

from astro_compass.constants.constants import Constants
from astro_compass.core.ephemeris import Ephemeris as Ephemeris
from astro_compass.core.spacecraft import Spacecraft
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
from astro_compass.utils.plotting_utils import SACRolloutData
from astro_compass.utils.state_vector_utils import cartesian_to_polar

plt.style.use("data/support_files/light_paper.mplstyle")
print("GPU available: ", torch.cuda.is_available())


def rollout_agent_random_IC(env, model, params, num_rollouts, path_output):
    # save the rollout to the
    # model_id/rollouts/rollout_data.pkl
    path_rollouts = os.path.join(path_output, "rollouts")
    os.makedirs(path_rollouts, exist_ok=True)

    for i in range(num_rollouts):
        obs, info = env.reset()

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


def rollout_agent_pretrain_IC(env, model, params, num_rollouts, path_output):
    # save the rollout to the
    # model_id/rollouts/rollout_data.pkl
    path_rollouts = os.path.join(path_output, "rollouts")
    os.makedirs(path_rollouts, exist_ok=True)

    for i in range(num_rollouts):
        obs, info = env.reset()

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
    sma_t_i = Constants.SMA_EARTH

    rollout_data = SACRolloutData()
    eph = Ephemeris()
    count_step = 0
    flag_continue = True
    terminated = False
    truncated = False
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
        rollout_data.add_step(
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

    return rollout_data, eph


def main():
    seed_in = 42
    params_path = os.path.join(CONFIG_ROOT, "SAC_training_TBT_config.toml")
    params = read_toml_config_file(params_path)
    env = gen_rl_environment(params)
    time_tag = "20251213_182606"
    paths = get_run_paths(RUNS_ROOT, time_tag)
    model = get_model(params, env, seed_in, paths["path_SAC_model"])

    num_rollouts = 1
    rollout_agent_random_IC(env, model, params, num_rollouts, paths["path_output"])


if __name__ == "__main__":
    main()
