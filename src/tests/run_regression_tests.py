from test_Hamiltonians import test_Hamiltonians
from test_env_step_with_action import test_env_step_with_action
from test_env_step_no_action import test_env_step_no_action
from test_env_step_with_nn_action import test_env_step_with_nn_action


def run_regression_tests(flag_report_live=False):

    print("Running Regression Tests...\n")

    arr_test_pass_bools = []
    arr_test_names = []

    arr_test_names.append("Hamiltonians Test")
    test_num = len(arr_test_names)
    print(f"Running Test {test_num}: {arr_test_names[test_num-1]}")
    flag_test_pass = test_Hamiltonians(flag_report_live)
    print("Test Hamiltonians passed: ", flag_test_pass)
    arr_test_pass_bools.append(flag_test_pass)

    test_name = "test_env_step_with_action"
    arr_test_names.append(test_name)
    test_num = len(arr_test_names)
    print(f"Running Test {test_num}: {test_name}")
    flag_test_pass = test_env_step_with_action(flag_report_live)
    print(f"{test_name} passed:  ", flag_test_pass)
    arr_test_pass_bools.append(flag_test_pass)

    test_name = "test_env_step_no_action"
    arr_test_names.append(test_name)
    test_num = len(arr_test_names)
    print(f"Running Test {test_num}: {test_name}")
    flag_test_pass = test_env_step_no_action(flag_report_live)
    print(f"{test_name} passed:  ", flag_test_pass)
    arr_test_pass_bools.append(flag_test_pass)

    test_name = "test_env_step_with_nn_action"
    arr_test_names.append(test_name)
    test_num = len(arr_test_names)
    print(f"Running Test {test_num}: {test_name}")
    flag_test_pass = test_env_step_with_nn_action(flag_report_live)
    print(f"{test_name} passed:  ", flag_test_pass)
    arr_test_pass_bools.append(flag_test_pass)

    print("\n\nSummary of Regression Tests:")
    num_tests = len(arr_test_names)
    flag_all_passed = True
    for i in range(num_tests):
        print(f"{arr_test_names[i]} passed: {arr_test_pass_bools[i]}")

        if not arr_test_pass_bools[i]:
            flag_all_passed = False

    print("\nAll tests passed: ", flag_all_passed)
    print("\n\nRegression tests complete")


run_regression_tests()
