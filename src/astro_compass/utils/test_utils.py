import re

import numpy as np

from astro_compass.core.ephemeris import Ephemeris
from astro_compass.core.ephemeris_v2 import Ephemeris_v2
from astro_compass.utils.log_utils import log


def compare_trajectories(
    sa_output_ephems, sa_truth_ephems, test_log, version, flag_report_live
):
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

        test_log = log("Comparing trajectory files", test_log, flag_report_live)
        test_log = log(f"Output: {ephem_file}", test_log, flag_report_live)
        test_log = log(f"Truth: {sa_truth_ephems[index]}", test_log, flag_report_live)

        flag_same = eph.compare_trajectories(
            eph_truth, position_tol=1_000_000.0, velocity_tol=1.0
        )

        if not flag_same:
            test_log = log(
                f"Trajectory {index} does not match.", test_log, flag_report_live
            )
            flag_all_match = False

        index += 1

    return test_log, flag_all_match


def compare_log_files_with_tolerance(
    path_output, path_truth, rtol=1e-5, atol=1e-8, flag_report_live=False
):
    """
    Compare two log files with numerical tolerance for cross-platform compatibility.

    Args:
        path_output: Path to the output log file
        path_truth: Path to the truth log file
        rtol: Relative tolerance for numerical comparison (default 1e-5 = 0.001%)
        atol: Absolute tolerance for numerical comparison (default 1e-8)
        flag_report_live: Whether to print detailed differences

    Returns:
        bool: True if files match within tolerance, False otherwise
    """
    try:
        with open(path_output, "r") as f1, open(path_truth, "r") as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()

        if len(lines1) != len(lines2):
            if flag_report_live:
                print(
                    f"Log files have different lengths: {len(lines1)} vs {len(lines2)}"
                )
            return False

        are_same = True
        for i, (line1, line2) in enumerate(zip(lines1, lines2)):
            # Extract all numbers from each line (including scientific notation)
            nums1 = [
                float(x) for x in re.findall(r"-?\d+\.?\d*[eE]?[+-]?\d*", line1) if x
            ]
            nums2 = [
                float(x) for x in re.findall(r"-?\d+\.?\d*[eE]?[+-]?\d*", line2) if x
            ]

            # If different number of numerical values, compare as strings
            if len(nums1) != len(nums2):
                if line1.strip() != line2.strip():
                    are_same = False
                    if flag_report_live:
                        print(
                            f"Line {i + 1} differs (non-numerical):\n  {line1.strip()}\n  {line2.strip()}"
                        )
            else:
                # Compare numerical values with tolerance
                for n1, n2 in zip(nums1, nums2):
                    if not np.isclose(n1, n2, rtol=rtol, atol=atol):
                        are_same = False
                        if flag_report_live:
                            print(
                                f"Line {i + 1} numerical difference: {n1} vs {n2} (diff: {abs(n1 - n2)})"
                            )
                        break

        return are_same

    except Exception as e:
        print(f"Error comparing log files: {e}")
        return False


def binary_compare(file1, file2):
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        return f1.read() == f2.read()
