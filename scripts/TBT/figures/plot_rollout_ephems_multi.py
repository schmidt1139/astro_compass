import os

import matplotlib.pyplot as plt

from astro_compass.utils.path_utils import RUNS_ROOT

from ....scripts.visualization.plot_ephems_multi import EphemPlotterExtend


def main():
    model_id = "20251217_142819"
    model_rollouts_dir = os.path.join(RUNS_ROOT, model_id, "rollouts", "ephems")

    vis = EphemPlotterExtend(model_rollouts_dir, num_ephems=3)

    vis.plot()
    plt.show()


if __name__ == "__main__":
    main()
