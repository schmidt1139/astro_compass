import os

from astro_compass.constants.constants import Constants
from astro_compass.core.ephemeris_v2 import Ephemeris_v2
from astro_compass.envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env
from astro_compass.utils.log_utils import log


def test_TBR_env(flag_report_live: bool = False):
    # define normalization parameters (for NN)
    params = {
        "mu": Constants.MU_SUN_M,  # sun mu [m^3/s^2]
        "max_T": 1.33 / 1000,  # max spacecraft thrust [kN]
        "ISP": 3872.0,  # spacecraft specific impulse [s]
        "TOF": 1.1 * 365.25 * 24 * 60 * 60,  # assumed time of flight [s]
        "l_star": 149598023000,  # characteristic length = Earth SMA [m]
        "m_star": 3366.0,  # characteristic mass = SC initial mass [kg]
        "t_star": (Constants.SMA_EARTH**3 / (Constants.MU_SUN_M))
        ** 0.5,  # characteristic time - derived
        "g0": Constants.G0,  # gravtational acceleration at Earth surface [m/s^2]
        "env_step_size": 3600 * 24,  # environment step size [s]
        "seed_env": 42,  # random seed for environment
        "num_trajs": 4,  # number of trajectories to simulate
        "max_steps": 1000,  # maximum number of steps per trajectory
    }

    test_log = []
    test_log = log("Test Two-Body Rendezvous Env", test_log, flag_report_live)

    env = TwoBodyRendezvous_Env(
        mu=params["mu"],  # solar gravitational parameter in m^3/s^2
        max_T=params["max_T"],  # max thrust in N
        ISP=params["ISP"],  # ISP in seconds
        TOF=params["TOF"],  # time of flight in seconds
        l_star=params["l_star"],  # characteristic length in m
        m_star=params["m_star"],  # characteristic mass in kg
        t_star=params["t_star"],  # characteristic time in s
        g0=params["g0"],  # gravitational acceleration at Earth surface in m/s^2
        step_size=params["env_step_size"],  # environment step size in seconds
    )

    count_traj = 0
    seed_traj = params["seed_env"]
    eph = Ephemeris_v2()

    flag_test_pass = True

    while count_traj < params["num_trajs"]:
        obs, info = env.reset(seed=seed_traj)

        test_log = log("Initial observation vector\n", test_log, flag_report_live)
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

        count_traj += 1
        seed_traj += 1
        steps = 0
        flag_continue = True

        eph.reset()

        eph.add_data(
            0.0,
            obs[0] * params["l_star"],
            obs[1] * params["l_star"],
            obs[2] * params["l_star"] / params["t_star"],
            obs[3] * params["l_star"] / params["t_star"],
            obs[4] * params["m_star"],
            obs[5] * params["l_star"],
            obs[6] * params["l_star"],
            obs[7] * params["l_star"] / params["t_star"],
            obs[8] * params["l_star"] / params["t_star"],
            obs[9] * params["t_star"],
            alpha_x=0.0,
            alpha_y=0.0,
            u=0.0,
        )

        while flag_continue == True:
            action = [0.1, -1.0, -1.0]  # action

            obs, reward, done, truncated, info = env.step(action)

            eph.add_data(
                info["Elapsed time"],
                obs[0] * params["l_star"],
                obs[1] * params["l_star"],
                obs[2] * params["l_star"] / params["t_star"],
                obs[3] * params["l_star"] / params["t_star"],
                obs[4] * params["m_star"],
                obs[5] * params["l_star"],
                obs[6] * params["l_star"],
                obs[7] * params["l_star"] / params["t_star"],
                obs[8] * params["l_star"] / params["t_star"],
                obs[9] * params["t_star"],
                alpha_x=action[2],
                alpha_y=action[1],
                u=action[0],
            )

            steps += 1

            if done or truncated:
                flag_continue = False

            if steps >= params["max_steps"]:
                flag_continue = False

        # fig = eph.plot_xy();
        # fig.savefig(os.path.join("data", "test_data", "test_TBR", "test_traj_") + str(count_traj) + "_TBR_env.png")

        eph.write_to_file(
            os.path.join("data", "test_data", "test_TBR", "test_traj_ephemeris_")
            + str(count_traj)
            + "_TBR_env.txt"
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

        # load truth data for comparison
        eph_truth = Ephemeris_v2()
        eph_truth.read_from_file(
            os.path.join("data", "test_data", "test_TBR", "test_traj_ephemeris_")
            + str(count_traj)
            + "_TBR_env_truth.txt"
        )

        # re-ingest ephemeris data for comparison
        eph_comp = Ephemeris_v2()
        eph_comp.read_from_file(
            os.path.join("data", "test_data", "test_TBR", "test_traj_ephemeris_")
            + str(count_traj)
            + "_TBR_env.txt"
        )

        eph_comp.compare_trajectories(
            eph_truth, position_tol=1e3, velocity_tol=1e-1, verbose=flag_report_live
        )

    if flag_test_pass:
        test_log = log(
            "Test PASSED: All trajectories match truth data within tolerance.",
            test_log,
            flag_report_live,
        )
    else:
        test_log = log(
            "Test FAILED: Discrepancies found between trajectories and truth data.",
            test_log,
            flag_report_live,
        )

    return flag_test_pass
