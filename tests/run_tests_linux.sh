#!/bin/bash
# Wrapper script to run regression tests on Linux with thread limits
# This prevents "Resource temporarily unavailable" errors on shared systems

# Set thread limits to prevent resource exhaustion
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# Run the tests
python run_regression_tests.py "$@"
