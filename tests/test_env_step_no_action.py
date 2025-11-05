import numpy as np
import gymnasium as gym
import sys
import os
import filecmp

from gymnasium import envs
from gymnasium.envs.registration import register
from utils.log_utils import log
from envs.TwoBody_Orb2Orb_Transfer_Env import TwoBody_Orb2Orb_Transfer_Env

# Adding python src code directory
current_dir = os.path.dirname(__file__)
python_src_dir = os.path.abspath(os.path.join(current_dir, "..", "python"))
sys.path.append(python_src_dir)


# register the environment if it isn't registered
if "TwoBody_Orb2Orb_Transfer_Env-v0" not in envs.registry.keys():
    register(
        id="TwoBody_Orb2Orb_Transfer_Env-v0",
        entry_point="TwoBody_Orb2Orb_Transfer_Env:TwoBody_Orb2Orb_Transfer_Env",
    )


def test_env_step_no_action(flag_report_live=False):

    # initialize the environment
    env = TwoBody_Orb2Orb_Transfer_Env()
    seed_in = 42

    test_log = []
    test_log = log("Test Environment Step with No Action", test_log, flag_report_live)

    observation_2, info = env.reset(seed=seed_in)
    test_log = log("Environment has been reset", test_log, flag_report_live)
    test_log = log("Seed: " + str(seed_in), test_log, flag_report_live)

    test_log = log("Observation: ", test_log, flag_report_live)
    test_log = log("X: " + str(observation_2[0]), test_log, flag_report_live)
    test_log = log("Y: " + str(observation_2[1]), test_log, flag_report_live)
    test_log = log("VX: " + str(observation_2[2]), test_log, flag_report_live)
    test_log = log("VY: " + str(observation_2[3]), test_log, flag_report_live)
    test_log = log("m: " + str(observation_2[4]), test_log, flag_report_live)
    test_log = log("mu: " + str(observation_2[5]), test_log, flag_report_live)
    test_log = log("sma_target: " + str(observation_2[6]), test_log, flag_report_live)
    test_log = log("\n", test_log, flag_report_live)

    test_log = log("Info: ", test_log, flag_report_live)
    for key in info.keys():
        test_log = log(str(key) + ": " + str(info[key]), test_log, flag_report_live)

    test_log = log("\n", test_log, flag_report_live)

    # zero action
    action = np.array([0.0, 0.0, 0.0])

    test_log = log("Action", test_log, flag_report_live)
    test_log = log("u: " + str(action[0]), test_log, flag_report_live)
    test_log = log("alpha_x: " + str(action[1]), test_log, flag_report_live)
    test_log = log("alpha_y: " + str(action[2]), test_log, flag_report_live)
    test_log = log("\n", test_log, flag_report_live)

    # step the environment
    observation_2, reward, terminated, truncated, info_2 = env.step(action)

    # update observation_2
    test_log = log("Observation: ", test_log, flag_report_live)
    test_log = log("X: " + str(observation_2[0]), test_log, flag_report_live)
    test_log = log("Y: " + str(observation_2[1]), test_log, flag_report_live)
    test_log = log("VX: " + str(observation_2[2]), test_log, flag_report_live)
    test_log = log("VY: " + str(observation_2[3]), test_log, flag_report_live)
    test_log = log("m: " + str(observation_2[4]), test_log, flag_report_live)
    test_log = log("mu: " + str(observation_2[5]), test_log, flag_report_live)
    test_log = log("sma_target: " + str(observation_2[6]), test_log, flag_report_live)
    test_log = log("\n", test_log, flag_report_live)

    # Report the reward
    test_log = log("reward: " + str(reward), test_log, flag_report_live)
    test_log = log("terminated: " + str(terminated), test_log, flag_report_live)
    test_log = log("truncated: " + str(truncated), test_log, flag_report_live)
    test_log = log("\n", test_log, flag_report_live)

    test_log = log("Info post-step: ", test_log, flag_report_live)
    for key in info_2.keys():
        test_log = log(str(key) + ": " + str(info_2[key]), test_log, flag_report_live)

    test_log = log("\n", test_log, flag_report_live)

    # write the log to a text file
    dir_test = os.path.normpath(
        os.path.join(os.getcwd(), "data", "test_data", "test_env_step_no_action")
    )
    path_test_report = os.path.normpath(
        os.path.join(dir_test, "output_test_env_step_no_action_log.txt")
    )
    path_test_truth = os.path.normpath(
        os.path.join(dir_test, "truth_test_env_step_no_action_log.txt")
    )
    with open(path_test_report, "w", encoding="utf-8") as f:
        for line in test_log:
            f.write(line + "\n")

    # compare the two files
    are_same = filecmp.cmp(path_test_report, path_test_truth, shallow=False)
    return are_same
