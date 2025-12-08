import os
import numpy as np
from utils.log_utils import compare_logs, log, read_config_file, read_log_from_file, write_log_to_file
from envs.TwoBody_Orb2Orb_Transfer_Env_target import TwoBody_Orb2Orb_Transfer_Env_target
from constants.constants import Constants
from core.ephemeris_v3 import Ephemeris_v3
from utils.env_utils import gen_rl_environment
from utils.state_vector_utils import convert_attitude_from_radial_to_cartesian
from utils.plotting_utils import plot_overlay_ballistic_orbit
import matplotlib.pyplot as plot

def test_TBT_env(flag_report_live: bool = False):

    # define normalization parameters (for NN)
    params = read_config_file(os.path.join("data", "test_data", "test_TBT", "test_TBT_env_config.txt"))

    data_path = os.path.normpath(params["data_path"].strip('"'))

    test_log = []
    test_log = log(
        "Test Two-Body Transfer Env", test_log, flag_report_live
    )

    # generate environment
    env = gen_rl_environment(params)

    # reset environment
    obs, info = env.reset(seed=params["seed_env_init"])

    env_type = type(env).__name__

    test_log = log(f"Environment type: {env_type}\n", test_log, flag_report_live)
    test_log = log("Initial observation vector\n", test_log, flag_report_live)
    test_log = log("r_nd_0: " + str(obs[0]), test_log, flag_report_live)
    test_log = log("cos_theta_0: " + str(obs[1]), test_log, flag_report_live)
    test_log = log("sin_theta_0: " + str(obs[2]), test_log, flag_report_live)
    test_log = log("v_r_nd_0: " + str(obs[3]), test_log, flag_report_live)
    test_log = log("v_eta_nd_0: " + str(obs[4]), test_log, flag_report_live)
    test_log = log("mass_nd_0: " + str(obs[5]), test_log, flag_report_live)
    test_log = log("r_nd_target: " + str(obs[6]), test_log, flag_report_live)
    test_log = log("v_r_nd_target: " + str(obs[7]), test_log, flag_report_live)
    test_log = log("v_eta_nd_target: " + str(obs[8]), test_log, flag_report_live)
    test_log = log("", test_log, flag_report_live)

    items_of_interest = [
        "orbital_period_years",
        "target_period_years",
        "state_a_nd",
        "state_e_nd",
        "state_w_deg",
        "state_theta_deg",
        "state_aol_deg",
        "eta_check_deg",
        "state_x_nd",
        "state_y_nd",
        "state_vx_nd",
        "state_vy_nd",
        "target_a_nd",
        "target_e_nd",
        "target_w_deg",
        "target_theta_deg",
        "target_aol_deg",
    ]

    for item in info:
        if item in items_of_interest:
            test_log = log(f"{item}: {info[item]}", test_log, flag_report_live)

    step_size_yrs = info["step_size_yrs"]
    step_size_sec = info["step_size_s"]
    elapsed_time_yrs = 0.0
    elapsed_time_sec = 0.0
    steps = 0
    flag_continue = True
    # extract the cartesian
    state = env.get_cartesian_state()
    x_init = state[0]
    y_init = state[1]
    vx_init = state[2]
    vy_init = state[3]
    x_target_init = state[5]
    y_target_init = state[6]
    vx_target_init = state[7]
    vy_target_init = state[8]

    arr_reward = []

    eph = Ephemeris_v3()

    while flag_continue:

        # sample action
        action = [0.25, 0.0, 1.0]  # no thrust

        #step the environment
        obs, reward, done, truncated, info = env.step(action)

        arr_reward.append(reward)

        # extract the cartesian
        state = env.get_cartesian_state()
        ttg = obs[9] * params["t_star"]  # time to go in seconds

        # add data to ephemeris
        '''
        (self, et, x, y, vx, vy, m, target_x, target_y, target_vx, target_vy, TTG, 
                 alpha_x=0.0, alpha_y=0.0, u=0.0):
        '''

        x = state[0]
        y = state[1]
        u = action[0]
        alpha_r = action[1]
        alpha_theta = action[2]
        # convert to x and y components
        alpha_x, alpha_y = convert_attitude_from_radial_to_cartesian(
            x, y, alpha_r, alpha_theta
        )

        eph.add_data(elapsed_time_sec,
                     state[0], #x
                     state[1], #y
                     state[2], #vx
                     state[3], #vy
                     state[4], #m
                     state[5], #target_x
                     state[6], #target_y
                     state[7], #target_vx
                     state[8], #target_vy
                     ttg,
                     alpha_x=action[1],
                     alpha_y=action[2],
                     u=action[0])


        elapsed_time_yrs += step_size_yrs
        elapsed_time_sec += step_size_sec
        steps += 1

        if done or truncated:
            flag_continue = False

    test_log = log(f"\n\nEnv steps: {steps}", test_log, flag_report_live)

    fig_orb = eph.plot_xy(color_in="#7e03a8")
    fig_orb = plot_overlay_ballistic_orbit(
        x_init,
        y_init,
        vx_init,
        vy_init,
        env,
        fig_orb,
        params,
        eph,
        label_in="Initial Orbit",
        color_in="#0d0887",
    )
    fig_orb = plot_overlay_ballistic_orbit(
        x_target_init,
        y_target_init,
        vx_target_init,
        vy_target_init,
        env,
        fig_orb,
        params,
        eph,
        label_in="Target Orbit",
        color_in="#cc4778",
    )
    fig_orb = eph.adjust_plot_limits()
    fig_orb.savefig(os.path.join(data_path, "test_TBT_env_xy_plot.png"), dpi=300)

    # plot rewards
    fig_reward = plot.figure()
    plot.plot(arr_reward, label="Reward per Step")  
    plot.legend()
    plot.xlabel("Step")
    plot.ylabel("Reward")
    fig_reward.savefig(os.path.join(data_path, "test_TBT_env_reward_plot.png"), dpi=300)

    eph.write_to_file(os.path.join(data_path, "test_TBT_env_ephemeris.txt"))

    test_log = log(f"Environment type: {env_type}\n", test_log, flag_report_live)
    test_log = log("Final observation vector\n", test_log, flag_report_live)
    test_log = log("r_nd_0: " + str(obs[0]), test_log, flag_report_live)
    test_log = log("cos_theta_0: " + str(obs[1]), test_log, flag_report_live)
    test_log = log("sin_theta_0: " + str(obs[2]), test_log, flag_report_live)
    test_log = log("v_r_nd_0: " + str(obs[3]), test_log, flag_report_live)
    test_log = log("v_eta_nd_0: " + str(obs[4]), test_log, flag_report_live)
    test_log = log("mass_nd_0: " + str(obs[5]), test_log, flag_report_live)
    test_log = log("r_nd_target: " + str(obs[6]), test_log, flag_report_live)
    test_log = log("v_r_nd_target: " + str(obs[7]), test_log, flag_report_live)
    test_log = log("v_eta_nd_target: " + str(obs[8]), test_log, flag_report_live)
    test_log = log("", test_log, flag_report_live)

    items_of_interest = [
        "orbital_period_years",
        "target_period_years",
        "state_a_nd",
        "state_e_nd",
        "state_w_deg",
        "state_theta_deg",
        "state_aol_deg",
        "eta_check_deg",
        "state_x_nd",
        "state_y_nd",
        "state_vx_nd",
        "state_vy_nd",
        "target_a_nd",
        "target_e_nd",
        "target_w_deg",
        "target_theta_deg",
        "target_aol_deg",
    ]

    for item in info:
        if item in items_of_interest:
            test_log = log(f"{item}: {info[item]}", test_log, flag_report_live)

    write_log_to_file(os.path.join(data_path, "test_TBT_env_log.txt"), test_log)

    test_log = read_log_from_file(os.path.join(data_path, "test_TBT_env_log.txt"))
    truth_log = read_log_from_file(os.path.join(data_path, "truth_TBT_env_log.txt"))

    flag_pass = compare_logs(test_log, truth_log)

    if flag_pass:
        test_log = log("\nTEST PASSED: Test log matches truth log.", test_log, flag_report_live)
    else:
        test_log = log("\nTEST FAILED: Test log does not match truth log.", test_log, flag_report_live)

    return flag_pass


if __name__ == "__main__":
    test_TBT_env(flag_report_live=True)