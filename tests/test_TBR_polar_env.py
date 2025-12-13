import os
import tempfile

from matplotlib import pyplot as plt

from astro_compass.core.ephemeris_v2 import Ephemeris_v2
from astro_compass.utils.env_utils import gen_rl_environment
from astro_compass.utils.log_utils import (
    compare_logs,
    log,
    read_config_file,
    read_log_from_file,
    write_log_to_file,
)
from astro_compass.utils.path_utils import DATA_ROOT
from astro_compass.utils.plotting_utils import (
    SACRolloutData_TBR_polar,
    plot_rendezvous_traj,
    plot_SAC_training_TBR_polar,
)


def test_TBR_polar_env(flag_report_live: bool = False):
    plt.style.use(f"{DATA_ROOT}/support_files/light_paper.mplstyle")

    # config path
    path_test = os.path.join(DATA_ROOT, "test_data", "test_TBR_polar_env")
    path_config = os.path.join(path_test, "TBR_polar_config.txt")

    # define normalization parameters (for NN)
    params = read_config_file(path_config)

    test_log = []
    test_log = log("Test Two-Body Rendezvous Polar Env", test_log, flag_report_live)

    # generate the environment
    env = gen_rl_environment(params)

    count_traj = 0
    seed_traj = params["seed_env"]
    eph = Ephemeris_v2()
    rollout_data = SACRolloutData_TBR_polar()

    output_dir = tempfile.mkdtemp()

    flag_test_pass = True

    while count_traj < params["num_trajs"]:
        obs, info = env.reset(seed=seed_traj)
        state_cart = env.get_cartesian_state()

        test_log = log("Initial observation vector\n", test_log, flag_report_live)
        test_log = log("r_nd: " + str(obs[0]), test_log, flag_report_live)
        test_log = log("cos_eta: " + str(obs[1]), test_log, flag_report_live)
        test_log = log("sin_eta: " + str(obs[2]), test_log, flag_report_live)
        test_log = log("r_nd_target: " + str(obs[3]), test_log, flag_report_live)
        test_log = log("cos_eta_target: " + str(obs[4]), test_log, flag_report_live)
        test_log = log("sin_eta_target: " + str(obs[5]), test_log, flag_report_live)
        test_log = log("x_current_nd: " + str(obs[6]), test_log, flag_report_live)
        test_log = log("y_current_nd: " + str(obs[7]), test_log, flag_report_live)
        test_log = log("vx_current_nd: " + str(obs[8]), test_log, flag_report_live)
        test_log = log("vy_current_nd: " + str(obs[9]), test_log, flag_report_live)
        test_log = log("x_target_nd: " + str(obs[10]), test_log, flag_report_live)
        test_log = log("y_target_nd: " + str(obs[11]), test_log, flag_report_live)
        test_log = log("vx_target_nd: " + str(obs[12]), test_log, flag_report_live)
        test_log = log("vy_target_nd: " + str(obs[13]), test_log, flag_report_live)
        test_log = log("x_diff_nd: " + str(obs[14]), test_log, flag_report_live)
        test_log = log("y_diff_nd: " + str(obs[15]), test_log, flag_report_live)
        test_log = log("vx_diff_nd: " + str(obs[16]), test_log, flag_report_live)
        test_log = log("vy_diff_nd: " + str(obs[17]), test_log, flag_report_live)
        test_log = log("r_nd_diff: " + str(obs[18]), test_log, flag_report_live)
        test_log = log("v_comp_diff: " + str(obs[19]), test_log, flag_report_live)
        test_log = log("TTG_nd: " + str(obs[20]), test_log, flag_report_live)
        test_log = log("mass_current_nd: " + str(obs[21]), test_log, flag_report_live)
        test_log = log("\n", test_log, flag_report_live)

        for item in info:
            test_log = log(f"{item}: {info[item]}", test_log, flag_report_live)
        test_log = log("\n\n\n", test_log, flag_report_live)

        count_traj += 1
        seed_traj += 1
        steps = 0
        flag_continue = True

        eph.reset()

        eph.add_data(
            0.0,
            state_cart[0],
            state_cart[1],
            state_cart[2],
            state_cart[3],
            state_cart[4],
            state_cart[5],
            state_cart[6],
            state_cart[7],
            state_cart[8],
            state_cart[9],
            alpha_x=0.0,
            alpha_y=0.0,
            u=0.0,
        )

        while flag_continue == True:
            action = [0.5, 1.0, 1.0]  # action

            obs, reward, done, truncated, info = env.step(action)
            state_cart = env.get_cartesian_state()

            # get relevant information
            pos_reward = info["pos_reward"]
            vel_reward = info["vel_reward"]
            mass_reward = info["mass_reward"]
            throttle_reward = info["throttle_reward"]
            v_current_nd = info["v_current_nd"]
            v_target_nd = info["v_target_nd"]
            v_r_unit = info["v_r_unit"]
            v_t_unit = info["v_t_unit"]
            delta_cos_eta = obs[4] - obs[1]
            delta_sin_eta = obs[5] - obs[2]
            delta_target_v_nd = v_target_nd - v_current_nd
            d_v_r_unit = info["v_r_target_unit"] - info["v_r_unit"]
            d_v_t_unit = info["v_t_target_unit"] - info["v_t_unit"]
            pos_residual = info["pos_residual"]
            vel_residual = info["vel_residual"]

            eph.add_data(
                info["Elapsed time"],
                state_cart[0],
                state_cart[1],
                state_cart[2],
                state_cart[3],
                state_cart[4],
                state_cart[5],
                state_cart[6],
                state_cart[7],
                state_cart[8],
                state_cart[9],
                alpha_x=action[2],
                alpha_y=action[1],
                u=action[0],
            )

            """
            time, #1
            reward, #2
            throttle, #3
            alpha_r, #4
            alpha_theta, #5
            rad, #6
            cos_theta, #7
            sin_theta, #8
            v, #9
            cos_fpa, #10
            sin_fpa, #11
            m, #12
            d_rad_f, #13
            d_cos_theta_f, #14
            d_sin_theta_f, #15
            d_v_f, #16
            d_v_r_unit, #17
            d_v_t_unit, #18
            ttg, #19
            pos_reward, #20
            vel_reward, #21
            mass_reward, #22
            throttle_reward, #23
            """

            # store the results
            rollout_data.add_step(
                info["Elapsed time"] / 86400,  # elapsed time in days #1
                reward,  # reward #2
                action[0],  # throttle #3
                action[1],  # alpha_r #4
                action[2],  # alpha_theta #5
                obs[0],  # r_nd #6
                obs[1],  # eta_cos_nd #7
                obs[2],  # eta_sin_nd #8
                v_current_nd,  # v_nd #9
                v_r_unit,  # v_r_unit #10
                v_t_unit,  # v_t_unit #11
                obs[21],  # mass_nd #12
                obs[18],  # delta target_r_nd #13
                delta_cos_eta,  # delta target_eta_cos_nd #14
                delta_sin_eta,  # delta target_eta_sin_nd #15
                delta_target_v_nd,  # delta target_v_nd #16
                d_v_r_unit,  # delta v_r_unit #17
                d_v_t_unit,  # delta v_t_unit #18
                obs[20],  # TTG_nd #19
                pos_reward,  # position reward #20
                vel_reward,  # velocity reward #21
                mass_reward,  # mass reward #22
                throttle_reward,  # throttle reward #23
                pos_residual,
                vel_residual,
            )

            steps += 1

            if done or truncated:
                flag_continue = False

            if steps >= params["max_steps"]:
                flag_continue = False

        # fig = eph.plot_xy();
        # fig.savefig(os.path.join(DATA_ROOT, "test_data", "test_TBR", "test_traj_") + str(count_traj) + "_TBR_env.png")

        plot_SAC_training_TBR_polar(rollout_data, output_dir, eph, params, env)

        eph.write_to_file(
            os.path.join(output_dir, f"test_traj_ephemeris_{count_traj}_TBR_env.txt")
        )

        fig_orb = plot_rendezvous_traj(eph, env, params)
        fig_orb.savefig(
            os.path.join(output_dir, "SAC_Test_Traj.png"), dpi=300, bbox_inches="tight"
        )

        test_log = log("Final observation vector\n", test_log, flag_report_live)
        test_log = log("x_nd: " + str(obs[0]), test_log, flag_report_live)
        test_log = log("y_nd: " + str(obs[1]), test_log, flag_report_live)
        test_log = log("vx_nd: " + str(obs[2]), test_log, flag_report_live)
        test_log = log("vy_nd: " + str(obs[3]), test_log, flag_report_live)
        test_log = log("mass_nd: " + str(obs[4]) + "\n", test_log, flag_report_live)
        test_log = log("x_target_nd: " + str(obs[5]), test_log, flag_report_live)
        test_log = log("y_target_nd: " + str(obs[6]), test_log, flag_report_live)
        test_log = log("vx_target_nd: " + str(obs[7]), test_log, flag_report_live)
        test_log = log("vy_target_nd: " + str(obs[8]), test_log, flag_report_live)
        test_log = log("TTG: " + str(obs[9]) + "\n", test_log, flag_report_live)

        for item in info:
            test_log = log(f"{item}: {info[item]}", test_log, flag_report_live)

        test_log = log("\n\n\n", test_log, flag_report_live)

    path_log = os.path.join(output_dir, "test_TBR_polar_env_log.txt")
    write_log_to_file(path_log, test_log)

    path_truth_log = os.path.join(path_test, "truth_TBR_polar_env_log.txt")  # reward 0?
    truth_log = read_log_from_file(path_truth_log)

    log_compare = read_log_from_file(path_log)

    flag_logs_same = compare_logs(log_compare, truth_log)
    if not flag_logs_same:
        flag_test_pass = False
        test_log = log(
            "Log file does NOT match truth log file.", test_log, flag_report_live
        )
    else:
        test_log = log("Log file matches truth log file.", test_log, flag_report_live)

    assert flag_test_pass


if __name__ == "__main__":
    test_TBR_polar_env(flag_report_live=True)
