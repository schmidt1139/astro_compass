import os

import matplotlib.pyplot as plt

from astro_compass.core.ephemeris_v3 import Ephemeris_v3
from astro_compass.utils.path_utils import DATA_ROOT, RUNS_ROOT
from astro_compass.vis.ephem_plotter import EphemPlotterExtend


def main():
    model_id = "20251217_142819"
    true_ephems_dir = os.path.join(DATA_ROOT, "pre-training-data", "TBT", "pickle")
    model_ephems_dir = os.path.join(RUNS_ROOT, model_id, "rollouts_H")

    # Need to convert rollouts to ephems first
    ephem = Ephemeris_v3()
    # Load the rollout data pickles and convert to ephemeris
    rollouts_files = os.listdir(model_ephems_dir)
    for file in rollouts_files:
        if file.endswith(".pkl"):
            path_rollout = os.path.join(model_ephems_dir, file)
            rollout_data, _ = Ephemeris_v3.load_rollout_data(path_rollout)
            ephem_from_rollout = Ephemeris_v3.from_rollout_data(rollout_data)
            # Save the ephem to the same directory
            path_ephem = os.path.join(
                model_ephems_dir,
                file.replace("rollout_data", "ephem").replace(".pkl", "_ephem.pkl"),
            )
            with open(path_ephem, "wb") as f:
                Ephemeris_v3.save_ephem(ephem_from_rollout, f)

    vis = EphemPlotterExtend(
        model_ephems_dir,
        num_ephems=3,
        directory_H=true_ephems_dir,
    )

    vis.plot()
    plt.show()


if __name__ == "__main__":
    main()
