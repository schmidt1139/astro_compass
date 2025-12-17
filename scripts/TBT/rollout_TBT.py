import os
import pickle

import matplotlib.pyplot as plt
import torch

from astro_compass.constants.constants import Constants
from astro_compass.core.ephemeris import Ephemeris as Ephemeris
from astro_compass.core.rollouts import SACRolloutData
from astro_compass.core.spacecraft import Spacecraft
from astro_compass.utils.env_utils import gen_rl_environment
from astro_compass.utils.log_utils import (
    read_toml_config_file,
)
from astro_compass.utils.model_utils import get_model
from astro_compass.utils.path_utils import (
    CONFIG_ROOT,
    LOGS_ROOT,
    RUNS_ROOT,
    get_run_paths,
)
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
        state_dict = model.env.decode_state(obs)
        throttle = action[0]
        alpha_x = action[1]
        alpha_y = action[2]

        # dim state
        t_i = info["Elapsed time"]
        t_i_days = t_i / (3600 * 24)
        x_i = state_dict["x_m"] * params["l_star"]
        y_i = state_dict["y_m"] * params["l_star"]
        vx_i = state_dict["vx_m_s"] * params["l_star"] / params["t_star"]
        vy_i = state_dict["vy_m_s"] * params["l_star"] / params["t_star"]
        m_i = state_dict["m_kg"] * params["m_star"]
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

        obs, reward, terminated, truncated, info = env.step(action)
        obs_dict = env.decode_observation(obs)
        reward_mass_component = info.get("reward_mass_component", 0.0)
        reward_distance_component = info.get("reward_distance_component", 0.0)

        count_step = count_step + 1

        # log data
        rollout_data.add_step(
            time=t_i_days,
            reward=reward,
            throttle=throttle,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
            x=state_dict["x_m"],
            y=state_dict["y_m"],
            vx=state_dict["vx_m_s"],
            vy=state_dict["vy_m_s"],
            sma=arr_OE[0],
            sma_target=sma_t_i,
            ecc=arr_OE[1],
            ecc_target=0.0,
            ecc_max=1.0,
            reward_mass=reward_mass_component,
            reward_distance=reward_distance_component,
        )

        if terminated or truncated:
            break

    return rollout_data, eph


def main():
    seed_in = 42
    params_path = os.path.join(CONFIG_ROOT, "SAC_training_TBT_config.toml")
    params = read_toml_config_file(params_path)
    env = gen_rl_environment(params)
    time_tag = "20251217_142819"
    paths = get_run_paths(RUNS_ROOT, time_tag)
    model = get_model(params, env, seed_in, paths["path_SAC_model"])

    num_rollouts = 1
    rollout_agent_random_IC(
        env, model, params, num_rollouts, LOGS_ROOT, paths["path_output"]
    )


if __name__ == "__main__":
    main()
