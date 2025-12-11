import os

import torch.nn as nn
from matplotlib import pyplot as plt
from stable_baselines3 import SAC as SB3_SAC

from astro_compass.core.training_data_generation import read_ephems_from_dir
from astro_compass.utils.env_utils import gen_rl_environment
from astro_compass.utils.log_utils import log, read_config_file
from astro_compass.utils.rl_utils import import_training_into_replay_buffer_v3
from astro_compass.utils.test_utils import binary_compare


def test_fast_replay_buffer_seeding(flag_report_live: bool = False):
    # seed
    seed_in = 42

    plt.style.use("data/support_files/light_paper.mplstyle")

    # config path
    path_test = os.path.join("data", "test_data", "test_fast_replay_buffer_seeding")
    path_config = os.path.join(path_test, "test_fast_replay_buffer_seeding_config.txt")

    # define normalization parameters (for NN)
    params = read_config_file(path_config)

    test_log = []
    test_log = log("Test Fast Replay Buffer Seeding", test_log, flag_report_live)

    # buffer size
    buffer_size = int(params.get("input_buffer_size", 100000))

    test_log = log(
        "Importing ephems from: " + str(params["path_ephems"]),
        test_log,
        flag_report_live,
    )
    test_log = log(
        "Number of ephems to use: " + str(params["num_ephems_to_use"]),
        test_log,
        flag_report_live,
    )
    test_log = log(
        "Input buffer capacity: " + str(buffer_size), test_log, flag_report_live
    )

    # read the ephems from directory
    set_ephems, filenames = read_ephems_from_dir(
        params["path_ephems"],
        params["num_ephems_to_use"],
        version=params["ephem_version"],
        flag_return_filenames=True,
        params=params,
    )

    test_log = log(
        f"Number of ephems read: {len(set_ephems)}", test_log, flag_report_live
    )

    # generate the environment
    env = gen_rl_environment(params)

    # load SB3 model

    # define the policy architecture
    policy_kwargs = dict(
        activation_fn=nn.LeakyReLU,  # LeakyReLU activation function
    )
    model = SB3_SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed_in,
        tensorboard_log=path_test,  # Use path_output so SB3 creates SAC_1/ subdirectory
        buffer_size=buffer_size,
        policy_kwargs=policy_kwargs,
    )

    test_log = log("Buffer capacity: " + str(buffer_size), test_log, flag_report_live)
    test_log = log(
        "Starting size: " + str(model.replay_buffer.size()), test_log, flag_report_live
    )

    import_training_into_replay_buffer_v3(set_ephems, test_log, model, env, params)

    path_replay_buffer = os.path.join(
        path_test, "test_fast_replay_buffer_seeding_replay_buffer.pkl"
    )
    model.save_replay_buffer(path_replay_buffer)

    test_log = log("Buffer capacity: " + str(buffer_size), test_log, flag_report_live)
    test_log = log(
        "Ending buffer size: " + str(model.replay_buffer.size()),
        test_log,
        flag_report_live,
    )

    # compare to truth
    path_replay_buffer_truth = os.path.join(
        path_test, "truth_fast_replay_buffer_seeding_replay_buffer.pkl"
    )

    flag_pass = binary_compare(path_replay_buffer, path_replay_buffer_truth)
    test_log = log(
        "Comparing generated replay buffer to truth buffer.", test_log, flag_report_live
    )
    test_log = log(
        f"Generated replay buffer path: {path_replay_buffer}",
        test_log,
        flag_report_live,
    )
    test_log = log(
        f"Truth replay buffer path: {path_replay_buffer_truth}",
        test_log,
        flag_report_live,
    )

    if flag_pass:
        test_log = log(
            "Fast replay buffer seeding test PASSED.", test_log, flag_report_live
        )
    else:
        test_log = log(
            "Fast replay buffer seeding test FAILED.", test_log, flag_report_live
        )

    return True


if __name__ == "__main__":
    test_fast_replay_buffer_seeding(True)
