import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plot
import sys
import os

from gymnasium import envs
from gymnasium.envs.registration import register

# Adding python src code directory
current_dir = os.path.dirname(__file__)
python_src_dir = os.path.abspath(os.path.join(current_dir, "..", "python"))
sys.path.append(python_src_dir)

from Ephemeris import Ephemeris
from Hamiltonian_Control import Hamiltonian_Controller_TBT


# register the environment if it isn't registered
if "TwoBody_Orb2Orb_Transfer_Env-v0" not in envs.registry.keys():
    register(
        id="TwoBody_Orb2Orb_Transfer_Env-v0",
        entry_point="TwoBody_Orb2Orb_Transfer_Env:TwoBody_Orb2Orb_Transfer_Env",
    )


# initialize the environment
env = gym.make("TwoBody_Orb2Orb_Transfer_Env-v0")

#plotting setup
matplotlib.rcParams.update({
    "text.usetex": False,                      # Use LaTeX for all text
    "font.family": "serif",                   # Use serif font
    "font.size": 10,                          # Match AIAA body font size
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "figure.figsize": (3.5, 2.5),             # Single-column figure
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": False,                       # No gridlines in AIAA style
})


num_traj = 10

# The prescribed time of flight for the transfer trajectory [s]
input_TOF = 1.1 * 365.25 * 24 * 60 * 60
steps_per_traj = np.ceil(input_TOF / env.unwrapped.step_size)

np.set_printoptions(precision=3)  # Limit to 3 decimal places

# reset the TBT env
seed = 111
init_observation, init_info = env.reset(seed=seed)

# ephemeris
eph = Ephemeris()

print(init_observation)

# extract some parameters of interest
sun_rad = env.unwrapped.planet_radii[0]
C1 = init_info["max_thrust"] * 1000  # max thrust in N
C2 = init_info["ISP"]  # spacecraft specific impulse in seconds

# compute Hamiltonian Solution
H_controller = Hamiltonian_Controller_TBT(env, init_observation, init_info, input_TOF)


# compute solution
flag_solved, h_sol, eps, sol, log = H_controller.hamiltonian_solution_finder()

# write output ephemeris
eph_out, arr_time, arr_u, arr_rho, arr_alpha_x, arr_alpha_y = (
    H_controller.generate_output_ephemeris(eph)
)

fig, ax = plot.subplots(figsize=(6, 6))
ax.plot(arr_time, arr_rho)
ax.set_title("Switching Function")

fig, ax = plot.subplots(figsize=(6, 6))
ax.plot(arr_time, arr_u)
ax.set_title("Spacecraft Thrust Throttle over Time")

fig, ax = plot.subplots(figsize=(6, 6))
ax.plot(arr_time, arr_alpha_x, label="alpha_x")
ax.plot(arr_time, arr_alpha_y, label="alpha_y")
ax.set_title("Alpha Vector (Maneuver Direction) over Time")
ax.legend()

# Ephemeris plotting
sun_rad = 6.957e8
sma_Earth = 149598023 * 1000  # m
sma_Mars = 2.32495e8 * 1000  # m
eph_out.plot_xy(sun_rad)
eph_out.plot_xy_ref_orbit(sma_Earth, "Earth Orbit")
eph_out.plot_xy_ref_orbit(sma_Mars, "Mars Orbit")


np.set_printoptions(precision=16)
print("Solution for initial co-states: ", h_sol)
print("Final smoothing parameter used in solution generation: ", eps)
print("Printing targeter log...\n")

for item in log:
    print(item)

print(sol)
