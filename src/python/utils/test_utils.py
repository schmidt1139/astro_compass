from core.ephemeris import Ephemeris
from core.ephemeris_v2 import Ephemeris_v2
from utils.log_utils import log


def compare_trajectories(sa_output_ephems, sa_truth_ephems, test_log, version, flag_report_live):

    flag_all_match = True

    # Compare the trajectories of the successful runs
    index = 0
    for ephem_file in sa_output_ephems:

        if version == 1:
            eph = Ephemeris()
            eph.read_from_file(ephem_file)

            eph_truth = Ephemeris()
            eph_truth.read_from_file(sa_truth_ephems[index])
        elif version == 2:
            eph = Ephemeris_v2()
            eph.read_from_file(ephem_file)

            eph_truth = Ephemeris_v2()
            eph_truth.read_from_file(sa_truth_ephems[index])
        else:
            raise ValueError("Invalid version specified for ephemeris comparison.")

        test_log = log(f"Comparing trajectory {index} to truth file: {sa_truth_ephems[index]}", test_log, flag_report_live)

        flag_same = eph.compare_trajectories(eph_truth, position_tol=1.0, velocity_tol=1e-3)

        if not flag_same:
            test_log = log(f"Trajectory {index} does not match.", test_log, flag_report_live)
            flag_all_match = False

        index += 1

    return test_log, flag_all_match
