import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import cpu_count

from tqdm import tqdm

from astro_compass.core.ephemeris import Ephemeris
from astro_compass.core.ephemeris_v2 import Ephemeris_v2
from astro_compass.core.ephemeris_v3 import Ephemeris_v3
from astro_compass.core.hamiltonian_control import Hamiltonian_Controller_TBT

# Adding python src code directory
sys.path.append(os.path.abspath("../python"))


def generate_nn_training_data(env, args, thread_id, output_dir):
    # reset the environment
    init_observation, init_info = env.reset()

    # The prescribed time of flight for the transfer trajectory [s]
    input_TOF = args["TOF"]

    # ephemeris
    eph = Ephemeris()

    # create H controller object
    H_controller = Hamiltonian_Controller_TBT(
        env, init_observation, init_info, input_TOF
    )

    # modify parameters
    H_controller.eps_threshold = args["eps_final"]

    # compute solution
    flag_solved, h_sol, eps, sol, log = H_controller.hamiltonian_solution_finder()

    if flag_solved:
        # write output ephemeris
        eph_out, arr_time, arr_u, arr_rho, arr_alpha_x, arr_alpha_y = (
            H_controller.generate_output_ephemeris(eph)
        )

        # writing ephemeris
        utc_now = datetime.utcnow()

        # Format: YYYY_MM_DD_hh_mm_ss_fs (fs = fractional seconds = microseconds)
        timestamp_str = utc_now.strftime("%Y_%m_%d_%H_%M_%S_%f")
        thread_id_str = str(thread_id)
        ephem_name = "ephemeris_t" + thread_id_str + "_" + timestamp_str + ".txt"

        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, ephem_name)

        eph_out.write_to_file(filename, mod_vector_write_frequency=10)

    else:
        filename = ""

    return thread_id, flag_solved, ephem_name


def generate_nn_training_data_parallel(env, args):
    print("Generating Parallel Neural Network Training Data")
    print("Two Body Transfer")
    print("Number of trajectories to generate: ", args["num_trajs"])
    print("CPU Count: ", cpu_count())
    print("Number of threads: ", args["num_threads"])
    print("Beginning data generation...\n")

    counter = 0

    with ProcessPoolExecutor(max_workers=args["num_threads"]) as executor:
        futures = []
        for traj_idx in range(args["num_trajs"]):
            futures.append(
                executor.submit(
                    generate_nn_training_data, env, args, traj_idx, args["output_dir"]
                )
            )

        for future in as_completed(futures):
            thread_id, result, filename = future.result()
            if result:
                counter = counter + 1
            print(
                f"Thread {thread_id}    Success: {result}   Counter: {counter}   Ephem: {filename}"
            )


def _read_single_ephem(path, version):
    if version == 1.0:
        eph = Ephemeris()
    elif version == 2.0:
        eph = Ephemeris_v2()
    elif version == 3.0:
        eph = Ephemeris_v3()
    else:
        raise ValueError("Unsupported ephemeris version: " + str(version))
    eph.read_from_file(path)
    return eph


def read_ephems_from_dir(
    directory,
    num_ephems_to_use=None,
    version=1.0,
    flag_return_filenames=False,
    params=None,
):
    filenames = os.listdir(directory)
    end_i = len(filenames)
    if num_ephems_to_use is not None:
        end_i = min(num_ephems_to_use, len(filenames))
    filenames = filenames[:end_i]
    paths = [os.path.join(directory, file) for file in filenames]

    num_workers = params.get("cores", 1) if params is not None else 1

    list_ephems = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_read_single_ephem, path, version) for path in paths]
        for f in tqdm(as_completed(futures), total=len(futures)):
            list_ephems.append(f.result())

    if flag_return_filenames:
        return list_ephems, filenames
    else:
        return list_ephems
