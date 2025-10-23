import sys
import os
from core.ephemeris import Ephemeris

def ingest_ephem_and_plot(path_to_ephemeris):
    print("Ingest Ephemeris and Plot Script")
    print("Ephemeris path: ", path_to_ephemeris)

    # create an ephemeris object
    eph = Ephemeris()

    # read in data from the specified file
    eph.read_from_file(path_to_ephemeris)

    print("Successfully ingested ephemeris")
    print("Number of vectors: ", eph.num_vectors)

    # Plot the ephemeris
    eph.plot_xy()
    sma_Earth = 149598023 * 1000  # m
    sma_Mars = 2.32495e8 * 1000  # m
    eph.plot_xy_ref_orbit(sma_Mars, "Mars","#b7410e")
    eph.plot_xy_ref_orbit(sma_Earth, "Earth")
    fig_orb = eph.plot_xy()
    eph.plot_all_ephemeris_data()

    # report the final vector
    index = eph.num_vectors - 1
    state_vector = eph.get_vector_at_index(index)
    print("Final vector: ", state_vector)


directory = "data\\test_data\\"
ephem_file_name = "ephemeris_nn_propagation.txt"
path_to_ephemeris = directory + ephem_file_name

ingest_ephem_and_plot(path_to_ephemeris)
