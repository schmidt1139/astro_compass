import os
import pickle

from astro_compass.utils.path_utils import DATA_ROOT
from astro_compass.vis.ephem_plotter import EphemPlotter


def main(path_to_ephemeris):
    with open(path_to_ephemeris, "rb") as f:
        eph = pickle.load(f)

    # Plot the ephemeris
    vis = EphemPlotter(eph)
    figs = vis.plot_all_ephemeris_data()

    # report the final vector
    index = eph.num_vectors - 1
    state_vector = eph.get_vector_at_index(index)
    print("Final vector: ", state_vector)


if __name__ == "__main__":
    directory = os.path.join(DATA_ROOT, "pre-training-data", "TBT", "pickle")
    path_to_ephemeris = os.path.join(directory, "8505947.pkl")

    main(path_to_ephemeris)
