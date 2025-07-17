import gymnasium as gym
import sys
import os
import torch
import filecmp

from gymnasium import envs
from gymnasium.envs.registration import register

# Adding python src code directory
current_dir = os.path.dirname(__file__)
python_src_dir = os.path.abspath(os.path.join(current_dir, "..", "python"))
sys.path.append(python_src_dir)

from NN_Utils import query_NN_at_state
from Constants import Constants
from Neural_Net_Controllers import NN_TBT_Controller


# register the environment if it isn't registered
if "TwoBody_Orb2Orb_Transfer_Env-v0" not in envs.registry.keys():
    register(
        id="TwoBody_Orb2Orb_Transfer_Env-v0",
        entry_point="TwoBody_Orb2Orb_Transfer_Env:TwoBody_Orb2Orb_Transfer_Env",
    )


# initialize the environment
env = gym.make("TwoBody_Orb2Orb_Transfer_Env-v0")
seed_in = 42


def log(info, log, flag_report_to_console=False):

    log.append(info)

    if flag_report_to_console:
        print(info)

    return log


def test_env_step_no_action(env, seed_in):

    test_log = []
    test_log = log(
        "Test Environment Step with Neural Network Thrust Action", test_log, True
    )

    # paths
    path_test_dir = os.path.normpath(
        os.path.join(os.getcwd(), "data\\test_data\\test_env_step_with_nn_action\\")
    )
    path_test_report = os.path.normpath(
        os.path.join(path_test_dir, "output_test_env_step_with_nn_action_log.txt")
    )
    path_test_truth = os.path.normpath(
        os.path.join(path_test_dir, "truth_test_env_step_with_nn_action_log.txt")
    )
    path_input_nn = os.path.normpath(
        os.path.join(
            path_test_dir, "nn_controller_weights_smoothed_full_10e3_epochs.pth"
        )
    )

    observation, info = env.reset(seed=seed_in)
    test_log = log("Environment has been reset", test_log, True)
    test_log = log("Seed: " + str(seed_in), test_log, True)

    # load neural network from file
    nn_controller = NN_TBT_Controller()  # instantiate NN object
    nn_control_param_dict = torch.load(
        path_input_nn
    )  # load parameter dictionary from file
    nn_controller.load_state_dict(
        nn_control_param_dict
    )  # load the state parameter dictionary

    test_log = log("Neural Net loaded from: " + str(path_input_nn), test_log, True)

    test_log = log("Observation: ", test_log, True)
    test_log = log("X: " + str(observation[0]), test_log, True)
    test_log = log("Y: " + str(observation[1]), test_log, True)
    test_log = log("VX: " + str(observation[2]), test_log, True)
    test_log = log("VY: " + str(observation[3]), test_log, True)
    test_log = log("m: " + str(observation[4]), test_log, True)
    test_log = log("mu: " + str(observation[5]), test_log, True)
    test_log = log("sma_target: " + str(observation[6]), test_log, True)
    test_log = log("\n", test_log, True)

    test_log = log("Info: ", test_log, True)
    for key in info.keys():
        test_log = log(str(key) + ": " + str(info[key]), test_log, True)

    test_log = log("\n", test_log, True)

    # pack NN state
    x = observation[0]
    y = observation[1]
    vx = observation[2]
    vy = observation[3]
    m = observation[4]
    state = [x, y, vx, vy, m]

    # define normalization parameters
    params = {
        "mu": Constants.MU_SUN * 10 ** (9),  # sun mu [m^3/s^2]
        "max_T": 1.33,  # max spacecraft thrust [N]
        "ISP": 3872.0,  # spacecraft specific impulse [s]
        "TOF": 1.1 * 365.25 * 24 * 60 * 60,  # assumed time of flight [s]
        "l_star": 149598023000,  # characteristic length = Earth SMA [m]
        "m_star": 3366.0,  # characteristic mass = SC initial mass [kg]
        "t_star": (149598023000**3 / (Constants.MU_SUN * 10 ** (9)))
        ** 0.5,  # characteristic time - derived
        "g0": Constants.G0,  # gravtational acceleration at Earth surface [m/s^2]
    }

    # get action from NN
    action = query_NN_at_state(nn_controller, state, params)

    test_log = log("Action", test_log, True)
    test_log = log("u: " + str(action[0]), test_log, True)
    test_log = log("alpha_x: " + str(action[1]), test_log, True)
    test_log = log("alpha_y: " + str(action[2]), test_log, True)
    test_log = log("\n", test_log, True)

    # step the environment
    observation, reward, terminated, truncated, info_2 = env.step(action)

    # update observation
    test_log = log("Observation: ", test_log, True)
    test_log = log("X: " + str(observation[0]), test_log, True)
    test_log = log("Y: " + str(observation[1]), test_log, True)
    test_log = log("VX: " + str(observation[2]), test_log, True)
    test_log = log("VY: " + str(observation[3]), test_log, True)
    test_log = log("m: " + str(observation[4]), test_log, True)
    test_log = log("mu: " + str(observation[5]), test_log, True)
    test_log = log("sma_target: " + str(observation[6]), test_log, True)
    test_log = log("\n", test_log, True)

    # Report the reward
    test_log = log("reward: " + str(reward), test_log, True)
    test_log = log("terminated: " + str(terminated), test_log, True)
    test_log = log("truncated: " + str(truncated), test_log, True)
    test_log = log("\n", test_log, True)

    test_log = log("Info post-step: ", test_log, True)
    for key in info_2.keys():
        test_log = log(str(key) + ": " + str(info_2[key]), test_log, True)

    test_log = log("\n", test_log, True)

    # write the log to a text file
    with open(path_test_report, "w", encoding="utf-8") as f:
        for line in test_log:
            f.write(line + "\n")

    # compare the two files
    print("output log: ", path_test_report)
    print("truth log: ", path_test_truth)
    are_same = filecmp.cmp(path_test_report, path_test_truth, shallow=False)
    print("Test passed? ", are_same)


test_env_step_no_action(env, seed_in)
