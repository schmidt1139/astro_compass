
import gymnasium as gym
import sys
import os

from gymnasium import envs
from gymnasium.envs.registration import register

# Adding python src code directory
sys.path.append(os.path.abspath("../python"))

from solve_two_body_transfer import solve_two_body_transfer_and_write_ephem

# USER INPUTS ------------------------------------------------------------------

# Initialize dictionary with input parameters to function
args = {
    "TOF": 1.1 * 365.25 * 24 * 60 * 60,  # s
    "r0": 2.32495e08,  # Initial state/radius [km]
    "theta0": 0.0,  # Initial state/true anomaly [rad]
    "vr0": 0.0,  # Initial state/radial velocity [km/s]
    "vtheta0": 23.6464,  # Initial state/transpose velocity [km/s]
    "m0": 3366.0,  # Initial state/mass [kg]
    "mu": 1.3e11,  # Central body gravitational parameter [km^3/s^2]
    "sma_target": 1.49598e08,  # Target sma of final circular orbit [km]
    "max_thrust": 1.33,  # Max thrust of the spacecraft engine [N]
    "ISP": 3872.0,  # Specific impulse of the spacecraft engine [s]
    "filename_ephemeris_out": "..\\..\\data\\training_ephems\\solution_ephemeris.txt" #output path
}

# END USER INPUTS --------------------------------------------------------------

# register the environment if it isn't registered
if "TwoBody_Orb2Orb_Transfer_Env-v0" not in envs.registry.keys():
    register(
        id="TwoBody_Orb2Orb_Transfer_Env-v0",
        entry_point="TwoBody_Orb2Orb_Transfer_Env:TwoBody_Orb2Orb_Transfer_Env",
    )


# initialize the environment
env = gym.make("TwoBody_Orb2Orb_Transfer_Env-v0")

solve_two_body_transfer_and_write_ephem(env, args)
