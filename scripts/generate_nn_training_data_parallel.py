import os
import sys

from astro_compass.utils.path_utils import PROJECT_ROOT

# Add project root to Python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from astro_compass.core.training_data_generation import (
    generate_nn_training_data_parallel,
)
from astro_compass.envs.TwoBody_Orb2Orb_Transfer_Env import TwoBody_Orb2Orb_Transfer_Env

if __name__ == "__main__":
    # Initialize dictionary with input parameters to function
    args = {
        "num_trajs": 12,  # number of trajectories to generate
        "num_threads": 6,  # number of threads
        "TOF": 1.1 * 365.25 * 24 * 60 * 60,  # s
        "r0": 2.32495e08,  # Initial state/radius [km]
        "vr0": 0.0,  # Initial state/radial velocity [km/s]
        "vtheta0": 23.6464,  # Initial state/transpose velocity [km/s]
        "m0": 3366.0,  # Initial state/mass [kg]
        "mu": 1.3e11,  # Central body gravitational parameter [km^3/s^2]
        "sma_target": 1.49598e08,  # Target sma of final circular orbit [km]
        "max_thrust": 1.33,  # Max thrust of the spacecraft engine [N]
        "ISP": 3872.0,  # Specific impulse of the spacecraft engine [s]
        "eps_final": 0.0001,  # Final smoothing parameter to achieve
        "output_dir": os.path.join("data", "training_ephems", "test_set3"),
    }

    # initialize the environment
    env = TwoBody_Orb2Orb_Transfer_Env()

    generate_nn_training_data_parallel(env, args)
