import os

from constants.constants import Constants
from envs.TwoBody_Orb2Orb_Transfer_Env_nd import TwoBody_Orb2Orb_Transfer_Env_nd

from astro_compass.core.solve_two_body_transfer import (
    solve_two_body_transfer_and_write_ephem,
)

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
    "filename_ephemeris_out": os.path.join(
        "data", "training_ephems", "solution_ephemeris.txt"
    ),  # output path
}

# END USER INPUTS --------------------------------------------------------------

# initialize the environment
env = TwoBody_Orb2Orb_Transfer_Env_nd(
    mu=args["mu"],
    max_T=args["max_thrust"],
    ISP=args["ISP"],
    TOF=args["TOF"],
    l_star=args["sma_target"],
    m_star=args["m0"],
    t_star=args["TOF"],
    g0=Constants.G0,
    step_size=3600,
)

solve_two_body_transfer_and_write_ephem(env, args)
