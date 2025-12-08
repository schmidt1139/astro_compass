"""Pytest-based regression harness aggregating key smoke and env tests.

This keeps the legacy regression coverage but runs via pytest with minimal boilerplate.
"""

import os
import sys
from pathlib import Path

import matplotlib
import pytest  # type: ignore
from utils.path_utils import PROJECT_ROOT, ensure_repo_paths_on_sys_path

# Keep these thread caps to avoid oversubscription in CI
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["PYTHONUNBUFFERED"] = "1"
matplotlib.use("Agg")

# Ensure repo imports work when running directly under pytest
ensure_repo_paths_on_sys_path()
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
os.chdir(PROJECT_ROOT)

# Import target test functions
from test_generate_buffer_smoke import test_generate_buffer_smoke  # noqa: E402
from test_pre_train_agent_smoke import test_pre_train_agent_smoke  # noqa: E402
from test_train_agent_smoke import test_train_agent_smoke  # noqa: E402

# Each entry: (name, callable, needs_flag_report_live)
_TEST_MATRIX = [
    # ("test_fast_replay_buffer_seeding", test_fast_replay_buffer_seeding, True),
    # ("test_TBR_polar_env", test_TBR_polar_env, True),
    # ("test_TBR_env", test_TBR_env, True),
    # ("test_Hamiltonian_TBR_controller", test_Hamiltonian_TBR_Controller, True),
    # ("test_Hamiltonians", test_Hamiltonians, True),
    # ("test_env_step_with_action", test_env_step_with_action, True),
    # ("test_env_step_no_action", test_env_step_no_action, True),
    # ("test_env_step_with_nn_action", test_env_step_with_nn_action, True),
    # ("test_SAC_training", test_SAC_training, True),
    # ("test_seeded_SAC_training", test_seeded_SAC_training, True),
    # (
    #     "test_datagen_Hamiltonian_TBR_parallel_hard",
    #     test_datagen_Hamiltonian_TBR_parallel_hard,
    #     True,
    # ),
    # (
    #     "test_datagen_Hamiltonian_TBR_parallel",
    #     test_datagen_Hamiltonian_TBR_parallel,
    #     True,
    # ),
    ("test_generate_buffer_smoke", test_generate_buffer_smoke, False),
    ("test_pre_train_agent_smoke", test_pre_train_agent_smoke, False),
    ("test_train_agent_smoke", test_train_agent_smoke, False),
    # ("test_evaluate_agent_smoke", test_evaluate_agent_smoke, False),
]


@pytest.mark.regression
@pytest.mark.parametrize("name,fn,needs_flag", _TEST_MATRIX)
def test_regression_suite(name, fn, needs_flag):
    """Run a curated regression test list via pytest."""
    if needs_flag:
        assert (
            fn(False) is not False
        )  # Many legacy tests return bool; treat truthy as pass
    else:
        fn()
