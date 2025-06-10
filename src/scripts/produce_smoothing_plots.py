import numpy as np
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plot
import sys
import os

from gymnasium import envs
from gymnasium.envs.registration import register

# Adding python src code directory
sys.path.append(os.path.abspath("../python"))

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


# The prescribed time of flight for the transfer trajectory [s]
input_TOF = 1.1 * 365.25 * 24 * 60 * 60
steps_per_traj = np.ceil(input_TOF / env.unwrapped.step_size)

np.set_printoptions(precision=3)  # Limit to 3 decimal places

# reset the TBT env
seed = 1
init_observation, init_info = env.reset(seed=seed)

# ephemeris
eph = Ephemeris()

print(init_observation)

# extract some parameters of interest
sun_rad = env.unwrapped.planet_radii[0]
C1 = init_info["max_thrust"] * 1000  # max thrust in N
C2 = init_info["ISP"]  # spacecraft specific impulse in seconds

print("Targeter eps 1.0")
# compute Hamiltonian Solution
H_controller = Hamiltonian_Controller_TBT(env, init_observation, init_info, input_TOF)

#alter parameters
H_controller.gamma = 1 - (1 / 2) ** (6)
H_controller.eps_0 = 1.0 / H_controller.gamma
H_controller.eps = H_controller.eps_0
H_controller.eps_threshold = 1.0

# compute solution
flag_solved, h_sol, eps, sol, log = H_controller.hamiltonian_solution_finder()

for item in log:
    print(item)
print(h_sol)
print("\n\n\n\n")

# write output ephemeris
eph_out, arr_time, arr_u, arr_rho, arr_alpha_x, arr_alpha_y = (
    H_controller.generate_output_ephemeris(eph)
)

figs = eph_out.plot_all_ephemeris_data()

i = 0
eph_num = 1
for fig in figs:
    i = i + 1
    fig.savefig("..\\..\\data\\plots\\eph_" + str(eph_num) + "_plot" + str(i) + ".pdf")
    
# Ephemeris plotting
sun_rad = 6.957e8
sma_Earth = 149598023 * 1000  # m
sma_Mars = 2.32495e8 * 1000  # m
eph_out.plot_xy(sun_rad)
eph_out.plot_xy_ref_orbit(sma_Earth, "Earth Orbit")
traj_plot = eph_out.plot_xy_ref_orbit(sma_Mars, "Mars Orbit")

traj_plot.savefig("..\\..\\data\\plots\\eph_" + str(eph_num) + "_traj_plot.pdf")

#-----------------------------------------------------------------------------


# eps 0.1
eph2 = Ephemeris()

print("Targeter eps 0.1")
# compute Hamiltonian Solution
H_controller2 = Hamiltonian_Controller_TBT(env, init_observation, init_info, input_TOF)

#alter parameters
H_controller2.eps_threshold = 0.1
H_controller2.arr_lam_0 = np.array([-0.575, -0.261, -0.407, -1.047,  0.139])

# compute solution
flag_solved, h_sol, eps, sol, log = H_controller2.hamiltonian_solution_finder()

for item in log:
    print(item)
print(h_sol)
print("\n\n\n\n")

# write output ephemeris
eph_out2, arr_time2, arr_u2, arr_rho2, arr_alpha_x2, arr_alpha_y2 = (
    H_controller2.generate_output_ephemeris(eph2)
)


# -----------------------------------------------------------------------------


# eps 0.01
eph3 = Ephemeris()

print("Targeter eps 0.01")
# compute Hamiltonian Solution
H_controller3 = Hamiltonian_Controller_TBT(env, init_observation, init_info, input_TOF)

#alter parameters
H_controller3.eps_threshold = 0.01
H_controller3.arr_lam_0 = np.array([-0.575, -0.261, -0.407, -1.047,  0.139])

# compute solution
flag_solved, h_sol, eps, sol, log = H_controller3.hamiltonian_solution_finder()

for item in log:
    print(item)
print(h_sol)
print("\n\n\n\n")

# write output ephemeris
eph_out3, arr_time3, arr_u3, arr_rho3, arr_alpha_x3, arr_alpha_y3 = (
    H_controller3.generate_output_ephemeris(eph3)
)

# -----------------------------------------------------------------------------


# eps 0.001
eph4 = Ephemeris()

print("Targeter eps 0.001")
# compute Hamiltonian Solution
H_controller4 = Hamiltonian_Controller_TBT(env, init_observation, init_info, input_TOF)

#alter parameters
H_controller4.arr_lam_0 = np.array([-0.575, -0.261, -0.407, -1.047,  0.139])

# compute solution
flag_solved, h_sol, eps, sol, log = H_controller4.hamiltonian_solution_finder()

for item in log:
    print(item)
print(h_sol)
print("\n\n\n\n")

# write output ephemeris
eph_out4, arr_time4, arr_u4, arr_rho4, arr_alpha_x4, arr_alpha_y4 = (
    H_controller4.generate_output_ephemeris(eph4)
)


# -----------------------------------------------------------------------------


# eps 0.0001
eph5 = Ephemeris()

print("Targeter eps 0.0001")
# compute Hamiltonian Solution
H_controller5 = Hamiltonian_Controller_TBT(env, init_observation, init_info, input_TOF)

H_controller5.eps_threshold = 0.0001

#alter parameters
H_controller5.arr_lam_0 = np.array([-0.575, -0.261, -0.407, -1.047,  0.139])

# compute solution
flag_solved, h_sol, eps, sol, log = H_controller5.hamiltonian_solution_finder()

for item in log:
    print(item)
print(h_sol)
print("\n\n\n\n")

# write output ephemeris
eph_out5, arr_time5, arr_u5, arr_rho5, arr_alpha_x5, arr_alpha_y5 = (
    H_controller5.generate_output_ephemeris(eph5)
)


#plotting-----------------------------------------------------------------------

fig, ax = plot.subplots(figsize=(6, 6))
ax.plot(eph_out.arr_et/86400, arr_rho, label=r"$\epsilon = 1$")
# ax.plot(eph_out2.arr_et/86400, arr_rho2, label=r"$\epsilon = 0.1$")
# ax.plot(eph_out3.arr_et/86400, arr_rho3, label=r"$\epsilon = 0.01$")
# ax.plot(eph_out4.arr_et/86400, arr_rho4, label=r"$\epsilon = 0.001$")
ax.plot(eph_out5.arr_et/86400, arr_rho5, label=r"$\epsilon = 0.0001$")
ax.set_xlabel(r"Elapsed Time (days)")
ax.set_ylabel(r"Switching Function $\rho$")
fig.tight_layout()
ax.set_title(r"Switching Function $\rho$")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("..\\..\\data\\plots\\rho.pdf")  # Vector format
plot.show()

fig, ax = plot.subplots(figsize=(6, 6))
ax.plot(eph_out.arr_et/86400, arr_u, label=r"$\epsilon = 1$")
# ax.plot(eph_out2.arr_et/86400, arr_u2, label=r"$\epsilon = 0.1$")
# ax.plot(eph_out3.arr_et/86400, arr_u3, label=r"$\epsilon = 0.01$")
# ax.plot(eph_out4.arr_et/86400, arr_u4, label=r"$\epsilon = 0.001$")
ax.plot(eph_out5.arr_et/86400, arr_u5, label=r"$\epsilon = 0.0001$")
ax.set_xlabel(r"Elapsed Time (days)")
ax.set_ylabel(r"$u$")
fig.tight_layout()
ax.set_title(r"Spacecraft Thrust Throttle $u$ over Time")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("..\\..\\data\plots\\throttle.pdf")  # Vector format
plot.show()

fig, ax = plot.subplots(figsize=(6, 6))
ax.plot(eph_out.arr_et/86400, eph_out.arr_m, label=r"$\epsilon = 1$")
ax.plot(eph_out2.arr_et/86400, eph_out2.arr_m, label=r"$\epsilon = 0.1$")
ax.plot(eph_out3.arr_et/86400, eph_out3.arr_m, label=r"$\epsilon = 0.01$")
ax.plot(eph_out4.arr_et/86400, eph_out4.arr_m, label=r"$\epsilon = 0.001$")
ax.plot(eph_out5.arr_et/86400, eph_out5.arr_m, label=r"$\epsilon = 0.0001$")
ax.set_xlabel(r"Elapsed Time (days)")
ax.set_ylabel(r"$u$")
fig.tight_layout()
ax.set_title(r"Spacecraft Mass over Time")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("..\\..\\data\plots\\mass.pdf")  # Vector format
plot.show()

fig, ax = plot.subplots(figsize=(6, 6))
ax.plot(eph_out.arr_et, arr_alpha_x, label=r"$\alpha_x$")
# ax.plot(eph_out2.arr_et, arr_alpha_y, label=r"$\alpha_y$")
ax.set_title(r"$alpha$ Vector (Maneuver Direction) over Time")
fig.tight_layout()
ax.legend()

print("e=1 final mass: ", eph_out.arr_m[-1] )
print("e=0.0001 final mass: ", eph_out5.arr_m[-1] )

sun_rad = 6.957e8
sma_Earth = 149598023 * 1000  # m
sma_Mars = 2.32495e8 * 1000  # m
eph_out5.plot_xy(sun_rad)
eph_out5.plot_xy_ref_orbit(sma_Earth, "Earth Orbit")
eph_out5.plot_xy_ref_orbit(sma_Mars, "Mars Orbit")


np.set_printoptions(precision=16)
print("Solution for initial co-states: ", h_sol)
print("Final smoothing parameter used in solution generation: ", eps)
print("Printing targeter log...\n")

for item in log:
    print(item)

print("\n\n\n\n")

figs = eph_out5.plot_all_ephemeris_data()

i = 0
eph_num = 5
for fig in figs:
    i = i + 1
    fig.savefig("..\\..\\data\\plots\\eph_" + str(eph_num) + "_plot" + str(i) + ".pdf")
    
# Ephemeris plotting
sun_rad = 6.957e8
sma_Earth = 149598023 * 1000  # m
sma_Mars = 2.32495e8 * 1000  # m
eph_out5.plot_xy(sun_rad)
eph_out5.plot_xy_ref_orbit(sma_Earth, "Earth Orbit")
traj_plot5 = eph_out5.plot_xy_ref_orbit(sma_Mars, "Mars Orbit")

traj_plot5.savefig("..\\..\\data\\plots\\eph_" + str(eph_num) + "_traj_plot.pdf")

filename="..\\..\\data\\training_ephems\\ephemris_report_20250510.txt"
eph_out.write_to_file(filename, mod_vector_write_frequency=10)