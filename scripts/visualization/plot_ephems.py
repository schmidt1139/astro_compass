import os

from astro_compass.core.ephemeris import Ephemeris
from astro_compass.utils.path_utils import PROJECT_ROOT
from astro_compass.vis.ephem_plotter import EphemPlotter


def ingest_ephem_and_plot(path_to_ephemeris):
    print("Ingest Ephemeris and Plot Script")
    print("Ephemeris path: ", path_to_ephemeris)

    # create an ephemeris object
    eph = Ephemeris()

    # read in data from the specified file
    eph.read(path_to_ephemeris)

    print("Successfully ingested ephemeris")
    print("Number of vectors: ", eph.num_vectors)

    # Plot the ephemeris
    vis = EphemPlotter(eph)
    figs = vis.plot_all_ephemeris_data()

    # report the final vector
    index = eph.num_vectors - 1
    state_vector = eph.get_vector_at_index(index)
    print("Final vector: ", state_vector)


directory = os.path.join(PROJECT_ROOT, "z_script_output", "temp_out")
ephem_file_name = "test_TBR_ephem_traj_seed_2002693622_tof_1p0_scenario_0_extended.txt"
path_to_ephemeris = directory + ephem_file_name

ingest_ephem_and_plot(path_to_ephemeris)
