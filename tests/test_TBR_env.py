from utils.log_utils import log
from envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env
from constants.constants import Constants
from core.ephemeris import Ephemeris

def test_TBR_env(flag_report_live: bool = False):

    # define normalization parameters (for NN)
    params = {
        "mu": Constants.MU_SUN_M,  # sun mu [m^3/s^2]
        "max_T": 1.33 / 1000,  # max spacecraft thrust [kN]
        "ISP": 3872.0,  # spacecraft specific impulse [s]
        "TOF": 1.1 * 365.25 * 24 * 60 * 60,  # assumed time of flight [s]
        "l_star": 149598023000,  # characteristic length = Earth SMA [m]
        "m_star": 3366.0,  # characteristic mass = SC initial mass [kg]
        "t_star": (Constants.SMA_EARTH**3 / (Constants.MU_SUN_M)) ** 0.5,  # characteristic time - derived
        "g0": Constants.G0,  # gravtational acceleration at Earth surface [m/s^2]
        "env_step_size": 3600 * 24,  # environment step size [s]
        "seed_env": 42,  # random seed for environment
        "num_trajs": 10,  # number of trajectories to simulate
        "max_steps": 1000,  # maximum number of steps per trajectory
    }

    test_log = []
    test_log = log(
        "Test Two-Body Rendezvous Env", test_log, flag_report_live
    )

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
    eph = Ephemeris()

    flag_test_pass = True

    while count_traj < params["num_trajs"]:

        obs, info = env.reset(seed=seed_traj)

        test_log = log("Initial observation vector\n", test_log, flag_report_live)
        test_log = log("x_nd: " + str(obs[0]), test_log, flag_report_live)
        test_log = log("y_nd: " + str(obs[1]), test_log, flag_report_live)
        test_log = log("vx_nd: " + str(obs[2]), test_log, flag_report_live)
        test_log = log("vy_nd: " + str(obs[3]), test_log, flag_report_live)
        test_log = log("mass_nd: " + str(obs[4]) + "\n", test_log, flag_report_live)

        for item in info:
            test_log = log(f"{item}: {info[item]}", test_log, flag_report_live)

        count_traj += 1
        seed_traj += 1
        steps = 0
        flag_continue = True

        eph.reset()
        
        eph.add_data(0.0, 
                     obs[0]*params["l_star"], 
                     obs[1]*params["l_star"], 
                     obs[2]*params["l_star"]/params["t_star"], 
                     obs[3]*params["l_star"]/params["t_star"], 
                     obs[4]*params["m_star"], 
                     alpha_x=0.0, 
                     alpha_y=0.0, 
                     u=0.0)

        while flag_continue == True:

            action = [ 0.1, -1.0, -1.0 ]  # action

            obs, reward, done, truncated, info = env.step(action)

            eph.add_data(info["Elapsed time"], 
                obs[0]*params["l_star"], 
                obs[1]*params["l_star"], 
                obs[2]*params["l_star"]/params["t_star"], 
                obs[3]*params["l_star"]/params["t_star"], 
                obs[4]*params["m_star"], 
                alpha_x=action[2], 
                alpha_y=action[1], 
                u=action[0])

            steps += 1

            if done or truncated:
                flag_continue = False

            if steps >= params["max_steps"]:
                flag_continue = False

        fig = eph.plot_xy();
        fig.savefig("data\\test_data\\test_TBR\\test_traj_" + str(count_traj) + "_TBR_env.png")

        eph.write_to_file("data\\test_data\\test_TBR\\test_traj_ephemeris_" + str(count_traj) + "_TBR_env.txt")
        
        test_log = log("Final observation vector\n", test_log, flag_report_live)
        test_log = log("x_nd: " + str(obs[0]), test_log, flag_report_live)
        test_log = log("y_nd: " + str(obs[1]), test_log, flag_report_live)
        test_log = log("vx_nd: " + str(obs[2]), test_log, flag_report_live)
        test_log = log("vy_nd: " + str(obs[3]), test_log, flag_report_live)
        test_log = log("mass_nd: " + str(obs[4]) + "\n", test_log, flag_report_live)

        # load truth data for comparison
        eph_truth = Ephemeris()
        eph_truth.read_from_file("data\\test_data\\test_TBR\\test_traj_ephemeris_" + str(count_traj) + "_TBR_env_truth.txt")

        # re-ingest ephemeris data for comparison
        eph_comp = Ephemeris()
        eph_comp.read_from_file("data\\test_data\\test_TBR\\test_traj_ephemeris_" + str(count_traj) + "_TBR_env.txt")

        for i in range(eph_comp.num_vectors):

            vec_compare = eph_comp.get_vector_at_index(i)
            vec_truth = eph_truth.get_vector_at_index(i)

            if (vec_compare[0] - vec_truth[0]) > 1e-12:
                test_log = log(f"Discrepancy in t at index {i}: {vec_compare[0]} vs {vec_truth[0]}", test_log, flag_report_live)
                flag_test_pass = False

            if (vec_compare[1] - vec_truth[1]) > 1e-12:
                test_log = log(f"Discrepancy in x at index {i}: {vec_compare[1]} vs {vec_truth[1]}", test_log, flag_report_live)
                flag_test_pass = False

            if (vec_compare[2] - vec_truth[2]) > 1e-12:
                test_log = log(f"Discrepancy in y at index {i}: {vec_compare[2]} vs {vec_truth[2]}", test_log, flag_report_live)
                flag_test_pass = False

            if (vec_compare[3] - vec_truth[3]) > 1e-12:
                test_log = log(f"Discrepancy in vx at index {i}: {vec_compare[3]} vs {vec_truth[3]}", test_log, flag_report_live)
                flag_test_pass = False

            if (vec_compare[4] - vec_truth[4]) > 1e-12:
                test_log = log(f"Discrepancy in vy at index {i}: {vec_compare[4]} vs {vec_truth[4]}", test_log, flag_report_live)
                flag_test_pass = False

            if (vec_compare[5] - vec_truth[5]) > 1e-12:
                test_log = log(f"Discrepancy in m at index {i}: {vec_compare[5]} vs {vec_truth[5]}", test_log, flag_report_live)
                flag_test_pass = False

            if (vec_compare[6] - vec_truth[6]) > 1e-12:
                test_log = log(f"Discrepancy in ax at index {i}: {vec_compare[6]} vs {vec_truth[6]}", test_log, flag_report_live)
                flag_test_pass = False

            if (vec_compare[7] - vec_truth[7]) > 1e-12:
                test_log = log(f"Discrepancy in ay at index {i}: {vec_compare[7]} vs {vec_truth[7]}", test_log, flag_report_live)
                flag_test_pass = False

            if (vec_compare[8] - vec_truth[8]) > 1e-12:
                test_log = log(f"Discrepancy in u at index {i}: {vec_compare[8]} vs {vec_truth[8]}", test_log, flag_report_live)
                flag_test_pass = False

    if flag_test_pass:
        test_log = log("Test PASSED: All trajectories match truth data within tolerance.", test_log, flag_report_live)
    else:
        test_log = log("Test FAILED: Discrepancies found between trajectories and truth data.", test_log, flag_report_live)

    return flag_test_pass

test_TBR_env(True)