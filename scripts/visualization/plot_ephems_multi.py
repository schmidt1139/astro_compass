import os

import matplotlib.pyplot as plt

from astro_compass.utils.path_utils import DATA_ROOT
from astro_compass.vis.ephem_plotter import EphemPlotterExtend


def main():
    directory = os.path.join(DATA_ROOT, "pre-training-data", "TBT", "pickle")

    vis = EphemPlotterExtend(directory, num_ephems=5)

    vis.plot()
    plt.show()


if __name__ == "__main__":
    main()
