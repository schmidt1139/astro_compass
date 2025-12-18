import os

from astro_compass.utils.plot_utils import RUNS_ROOT

from astro_compass.core.ephem_converter import generate_rollouts
from astro_compass.vis.rollout_plotter import RolloutPlotter


def main():
    rollout_data = None
    path = None

    # Generate rollout data from the Hamiltonian Trajectories

    # Load ephem pickles
    directory = os.path.join(RUNS_ROOT, "20251214_175440")
    ephem = import_ephem(directory, idx=0)
    rollout = generate_rollouts(ephem)

    # Plot the rollout metrics vs time

    vis = RolloutPlotter(rollout_data, path)
    vis.plot()
    pass


if __name__ == "__main__":
    main()
