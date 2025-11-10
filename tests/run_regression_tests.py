import sys
import os

# CRITICAL: Set thread limits BEFORE any other imports to prevent resource exhaustion on shared systems
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['PYTHONUNBUFFERED'] = '1'  # Unbuffered output for real-time console feedback

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter threading issues
import gymnasium as gym

# Get the project root directory and change to it
# This ensures relative paths in tests work correctly
tests_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(tests_dir)
os.chdir(project_root)
print(f"Working directory set to: {os.getcwd()}")

# Add the tests directory to the path so we can import test modules
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

# Import test functions WITHOUT the 'tests.' prefix
from test_Hamiltonians import test_Hamiltonians
from test_env_step_with_action import test_env_step_with_action
from test_env_step_no_action import test_env_step_no_action
from test_env_step_with_nn_action import test_env_step_with_nn_action
from test_SAC_training import test_SAC_training
from test_seeded_SAC_training import test_seeded_SAC_training
from test_TBR_env import test_TBR_env
from test_Hamiltonian_TBR_controller import test_Hamiltonian_TBR_Controller
from test_datagen_Hamiltonian_TBR_controller_parallel import test_datagen_Hamiltonian_TBR_parallel
from test_datagen_Hamiltonian_TBR_controller_parallel_hard import test_datagen_Hamiltonian_TBR_parallel_hard


def run_regression_tests(flag_report_live=False):

    print("Running Regression Tests...\n")

    arr_test_pass_bools = []
    arr_test_names = []

    test_name = "test_datagen_Hamiltonian_TBR_parallel_hard"
    arr_test_names.append(test_name)
    test_num = len(arr_test_names)
    print(f"\n\nRunning Test {test_num}: {test_name}")
    flag_test_pass = test_datagen_Hamiltonian_TBR_parallel_hard(flag_report_live)
    print(f"\n\n{test_name} passed:  ", flag_test_pass)
    arr_test_pass_bools.append(flag_test_pass)

    test_name = "test_datagen_Hamiltonian_TBR_parallel"
    arr_test_names.append(test_name)
    test_num = len(arr_test_names)
    print(f"\n\nRunning Test {test_num}: {test_name}")
    flag_test_pass = test_datagen_Hamiltonian_TBR_parallel(flag_report_live)
    print(f"\n\n{test_name} passed:  ", flag_test_pass)
    arr_test_pass_bools.append(flag_test_pass)

    test_name = "test_TBR_env"
    arr_test_names.append(test_name)
    test_num = len(arr_test_names)
    print(f"\n\nRunning Test {test_num}: {test_name}")
    flag_test_pass = test_TBR_env(flag_report_live)
    print(f"\n\n{test_name} passed:  ", flag_test_pass)
    arr_test_pass_bools.append(flag_test_pass)

    test_name = "test_Hamiltonian_TBR_controller"
    arr_test_names.append(test_name)
    test_num = len(arr_test_names)
    print(f"\n\nRunning Test {test_num}: {test_name}")
    flag_test_pass = test_Hamiltonian_TBR_Controller(flag_report_live)
    print(f"\n\n{test_name} passed:  ", flag_test_pass)
    arr_test_pass_bools.append(flag_test_pass)
                            
    arr_test_names.append("test_Hamiltonians")
    test_num = len(arr_test_names)
    print(f"\n\nRunning Test {test_num}: {arr_test_names[test_num-1]}")
    flag_test_pass = test_Hamiltonians(flag_report_live)
    print(f"\n\n{arr_test_names[test_num-1]} passed: ", flag_test_pass)
    arr_test_pass_bools.append(flag_test_pass)

    test_name = "test_env_step_with_action"
    arr_test_names.append(test_name)
    test_num = len(arr_test_names)
    print(f"\n\nRunning Test {test_num}: {test_name}")
    flag_test_pass = test_env_step_with_action(flag_report_live)
    print(f"\n\n{test_name} passed:  ", flag_test_pass)
    arr_test_pass_bools.append(flag_test_pass)

    test_name = "test_env_step_no_action"
    arr_test_names.append(test_name)
    test_num = len(arr_test_names)
    print(f"\n\nRunning Test {test_num}: {test_name}")
    flag_test_pass = test_env_step_no_action(flag_report_live)
    print(f"\n\n{test_name} passed:  ", flag_test_pass)
    arr_test_pass_bools.append(flag_test_pass)

    test_name = "test_env_step_with_nn_action"
    arr_test_names.append(test_name)
    test_num = len(arr_test_names)
    print(f"\n\nRunning Test {test_num}: {test_name}")
    flag_test_pass = test_env_step_with_nn_action(flag_report_live)
    print(f"\n\n{test_name} passed:  ", flag_test_pass)
    arr_test_pass_bools.append(flag_test_pass)

    test_name = "test_SAC_training"
    arr_test_names.append(test_name)
    test_num = len(arr_test_names)
    print(f"\n\nRunning Test {test_num}: {test_name}")
    flag_test_pass = test_SAC_training(flag_report_live)
    print(f"\n\n{test_name} passed:  ", flag_test_pass)
    arr_test_pass_bools.append(flag_test_pass)

    test_name = "test_seeded_SAC_training"
    arr_test_names.append(test_name)
    test_num = len(arr_test_names)
    print(f"\n\nRunning Test {test_num}: {test_name}")
    flag_test_pass = test_seeded_SAC_training(flag_report_live)
    print(f"\n\n{test_name} passed:  ", flag_test_pass)
    arr_test_pass_bools.append(flag_test_pass)


    print("\n\nSummary of Regression Tests:")
    num_tests = len(arr_test_names)
    flag_all_passed = True
    for i in range(num_tests):

        line = f"{arr_test_names[i]}"
        len_line = len(line)
        while len_line < 40:
            line += " "
            len_line += 1

        print(f"{line} passed: {arr_test_pass_bools[i]}")

        if not arr_test_pass_bools[i]:
            flag_all_passed = False

    print("\nAll tests passed: ", flag_all_passed)
    print("\n\nRegression tests complete")


if __name__ == "__main__":
    run_regression_tests(False)
