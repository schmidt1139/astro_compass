import os
import pickle

import matplotlib.pyplot as plt

from astro_compass.core.ephem_converter import generate_rollouts
from astro_compass.utils.log_utils import read_toml_config_file
from astro_compass.utils.path_utils import CONFIG_ROOT, DATA_ROOT, PLOT_ROOT
from astro_compass.vis.rollout_plotter import RolloutPlotter


def main():
    # Generate rollout data from the Hamiltonian Trajectories
    path_config = os.path.join(CONFIG_ROOT, "SAC_training_TBT_config.toml")
    params = read_toml_config_file(path_config)

    # Load ephem pickles
    ephem_path = os.path.join(
        DATA_ROOT,
        "pre-training-data",
        "TBT",
        "pickle",
        "477753.pkl",
    )
    with open(ephem_path, "rb") as f:
        ephem = pickle.load(f)

    rollouts = generate_rollouts(ephem, params)

    # Plot the rollout metrics vs time

    vis = RolloutPlotter(rollouts[0], PLOT_ROOT)
    vis.plot()
    plt.show()


if __name__ == "__main__":
    main()
