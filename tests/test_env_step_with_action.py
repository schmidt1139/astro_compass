import os
import tempfile

import numpy as np

from astro_compass.envs.TwoBody_Orb2Orb_Transfer_Env import TwoBody_Orb2Orb_Transfer_Env
from astro_compass.utils.log_utils import log
from astro_compass.utils.path_utils import DATA_ROOT
from astro_compass.utils.test_utils import compare_log_files_with_tolerance


def test_env_step_with_action(flag_report_live=False):
    # initialize the environment
    env = TwoBody_Orb2Orb_Transfer_Env()
    seed_in = 42

    test_log = []
    test_log = log(
        "Test Environment Step with Thrust Action", test_log, flag_report_live
    )

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

    # throttle is 1 and unit vector components are 1 and 1 (vector is normalized in "step")
    action = np.array([1.0, 1.0, 1.0])

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
        os.path.join(DATA_ROOT, "test_data", "test_env_step_with_action")
    )
    path_test_report = tempfile.NamedTemporaryFile().name

    path_test_truth = os.path.normpath(
        os.path.join(dir_test, "truth_test_env_step_with_action_log.txt")
    )
    with open(path_test_report, "w", encoding="utf-8") as f:
        for line in test_log:
            f.write(line + "\n")

    # Compare log files with numerical tolerance for cross-platform compatibility
    are_same = compare_log_files_with_tolerance(
        path_test_report, path_test_truth, flag_report_live=False
    )

    return are_same


if __name__ == "__main__":
    test_env_step_with_action(flag_report_live=True)
