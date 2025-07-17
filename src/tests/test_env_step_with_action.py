import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plot
import sys
import os
import filecmp

from gymnasium import envs
from gymnasium.envs.registration import register

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


# initialize the environment
env = gym.make("TwoBody_Orb2Orb_Transfer_Env-v0")
seed_in = 42

def log(info, log, flag_report_to_console=False):

    log.append(info)

    if flag_report_to_console:
        print(info)

    return log


def test_env_step_no_action(env,seed_in):

    test_log = []
    test_log = log("Test Environment Step with Thrust Action", test_log, True)

    total_steps_in_env = 0

    observation_2, info = env.reset(seed=seed_in)
    test_log = log("Environment has been reset", test_log, True)
    test_log = log("Seed: " + str(seed_in), test_log, True)

    test_log = log("Observation: ", test_log, True)
    test_log = log("X: " + str(observation_2[0]), test_log, True)
    test_log = log("Y: " + str(observation_2[1]), test_log, True)
    test_log = log("VX: " + str(observation_2[2]), test_log, True)
    test_log = log("VY: " + str(observation_2[3]), test_log, True)
    test_log = log("m: " + str(observation_2[4]), test_log, True)
    test_log = log("mu: " + str(observation_2[5]), test_log, True)
    test_log = log("sma_target: " + str(observation_2[6]), test_log, True)
    test_log = log("\n", test_log, True)

    test_log = log("Info: ", test_log, True)
    for key in info.keys():
        test_log = log(str(key) + ": " + str(info[key]), test_log, True)

    test_log = log("\n", test_log, True)

    #throttle is 1 and unit vector components are 1 and 1 (vector is normalized in "step")
    action = np.array([1.0, 1.0, 1.0])

    test_log = log("Action", test_log, True)
    test_log = log("u: " + str(action[0]), test_log, True)
    test_log = log("alpha_x: " + str(action[1]), test_log, True)
    test_log = log("alpha_y: " + str(action[2]), test_log, True)
    test_log = log("\n", test_log, True)

    #step the environment
    observation_2, reward, terminated, truncated, info_2 = env.step(action)

    #update observation_2
    test_log = log("Observation: ", test_log, True)
    test_log = log("X: " + str(observation_2[0]), test_log, True)
    test_log = log("Y: " + str(observation_2[1]), test_log, True)
    test_log = log("VX: " + str(observation_2[2]), test_log, True)
    test_log = log("VY: " + str(observation_2[3]), test_log, True)
    test_log = log("m: " + str(observation_2[4]), test_log, True)
    test_log = log("mu: " + str(observation_2[5]), test_log, True)
    test_log = log("sma_target: " + str(observation_2[6]), test_log, True)
    test_log = log("\n", test_log, True)

    #Report the reward
    test_log = log("reward: " + str(reward), test_log, True)
    test_log = log("terminated: " + str(terminated), test_log, True)
    test_log = log("truncated: " + str(truncated), test_log, True)
    test_log = log("\n", test_log, True)

    test_log = log("Info post-step: ", test_log, True)
    for key in info_2.keys():
        test_log = log(str(key) + ": " + str(info_2[key]), test_log, True)

    test_log = log("\n", test_log, True)

    #write the log to a text file
    dir_test = os.path.normpath(os.path.join(os.getcwd(), "data\\test_data\\test_env_step_with_action\\"))
    path_test_report = os.path.normpath(os.path.join(dir_test, "output_test_env_step_with_action_log.txt"))
    path_test_truth = os.path.normpath(os.path.join(dir_test, "truth_test_env_step_with_action_log.txt"))
    with open(path_test_report, "w", encoding="utf-8") as f:
        for line in test_log:
            f.write(line + "\n")

    #compare the two files
    print("output log: ", path_test_report)
    print("truth log: ", path_test_truth)
    are_same = filecmp.cmp(path_test_report, path_test_truth, shallow=False)
    print("Test passed? ", are_same)

    



test_env_step_no_action(env, seed_in)


