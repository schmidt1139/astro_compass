import sys
import os

# Adding python src code directory
sys.path.append(os.path.abspath("../python"))

from Ephemeris import Ephemeris


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

    # report the final vector
    index = eph.num_vectors - 1
    state_vector = eph.get_vector_at_index(index)
    print("Final vector: ", state_vector)


directory = "..\\..\\data\\training_ephems\\"
ephem_file_name = "test_ephemeris.txt"
path_to_ephemeris = directory + ephem_file_name

ingest_ephem_and_plot(path_to_ephemeris)
