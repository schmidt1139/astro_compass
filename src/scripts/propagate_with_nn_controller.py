import gymnasium as gym
import sys
import os
import torch
import filecmp
import matplotlib.pyplot as plt

from gymnasium import envs
from gymnasium.envs.registration import register

# Adding python src code directory
# Adding python src code directory
os.chdir("C:/Users/micha/MSI_Data/Masters_Thesis/astro_compass")
print("Now working in:", os.getcwd())

sys.path.append(os.path.relpath("src/python/"))
sys.path.append(os.path.relpath("src/scripts/"))

from NN_Utils import query_NN_at_state
from Constants import Constants
from Neural_Net_Controllers import NN_TBT_Controller
from Log_Utils import log
from Ephemeris import Ephemeris

# register the environment if it isn't registered
if "TwoBody_Orb2Orb_Transfer_Env-v0" not in envs.registry.keys():
    register(
        id="TwoBody_Orb2Orb_Transfer_Env-v0",
        entry_point="TwoBody_Orb2Orb_Transfer_Env:TwoBody_Orb2Orb_Transfer_Env",
    )

# initialize the environment
env = gym.make("TwoBody_Orb2Orb_Transfer_Env-v0")
seed_in = 42

plt.style.use("data/support_files/dark_scientific.mplstyle")

def prop_with_nn_controller(env, seed_in):

    test_log = []
    test_log = log(
        "Test Environment Step with Neural Network Thrust Action", test_log, True
    )

    # paths
    path_test_dir = os.path.normpath(
        os.path.join(os.getcwd(), "data\\test_data\\")
    )
    path_nns = os.path.normpath( os.path.join(os.getcwd(), "data\\neural_networks\\"))
    path_plots = os.path.normpath(
        os.path.join(os.getcwd(), "data\\plots\\")
    )
    path_input_nn = os.path.normpath(
        os.path.join(
            path_nns, "nn_controller_weights_smoothed_full_10e3_epochs.pth"
        )
    )

    #reset the environment
    observation, info = env.reset(seed=seed_in)
    test_log = log("Environment has been reset", test_log, True)
    test_log = log("Seed: " + str(seed_in), test_log, True)

    #Generate an ephemeris object
    eph = Ephemeris()

    # define normalization parameters (for NN)
    params = {
        "mu": Constants.MU_SUN * 10 ** (9),  # sun mu [m^3/s^2]
        "max_T": 1.33,  # max spacecraft thrust [N]
        "ISP": 3872.0,  # spacecraft specific impulse [s]
        "TOF": 1.1 * 365.25 * 24 * 60 * 60,  # assumed time of flight [s]
        "l_star": 149598023000,  # characteristic length = Earth SMA [m]
        "m_star": 3366.0,  # characteristic mass = SC initial mass [kg]
        "t_star": (149598023000**3 / (Constants.MU_SUN * 10 ** (9)))
        ** 0.5,  # characteristic time - derived
        "g0": Constants.G0,  # gravtational acceleration at Earth surface [m/s^2]
    }

    # load neural network from file
    nn_controller = NN_TBT_Controller()  # instantiate NN object
    nn_control_param_dict = torch.load(
        path_input_nn
    )  # load parameter dictionary from file
    nn_controller.load_state_dict(
        nn_control_param_dict
    )  # load the state parameter dictionary

    test_log = log("Neural Net loaded from: " + str(path_input_nn), test_log, True)

    test_log = log("Info: ", test_log, True)
    for key in info.keys():
        test_log = log(str(key) + ": " + str(info[key]), test_log, True)

    test_log = log("\n", test_log, True)

    # pack NN state
    t_i = 0.0
    x_i = observation[0]*1000
    y_i = observation[1]*1000
    vx_i = observation[2]*1000
    vy_i = observation[3]*1000
    m_i = observation[4]
    state_i = [x_i, y_i, vx_i, vy_i, m_i]

    while ( t_i <= params['TOF'] ):

        # get action from NN
        action = query_NN_at_state(nn_controller, state_i, params)
        u_i = action[0]
        alpha_x_i = action[1]
        alpha_y_i = action[2]

        #add initial state
        eph.add_data(
                    t_i, x_i, y_i, vx_i, vy_i, m_i, alpha_x_i, alpha_y_i, u_i
                )

        # step the environment
        observation, reward, terminated, truncated, info_2 = env.step(action)

        # transition state vector (t = i + 1)
        t_i = info_2['Elapsed time']
        x_i = observation[0]*1000
        y_i = observation[1]*1000
        vx_i = observation[2]*1000
        vy_i = observation[3]*1000
        m_i = observation[4]
        state_i = [x_i, y_i, vx_i, vy_i, m_i]

    test_log = log("Final Info: ", test_log, True)
    for key in info_2.keys():
        test_log = log(str(key) + ": " + str(info_2[key]), test_log, True)

    test_log = log("\n", test_log, True)
    sma_achieved = info_2["a"]*1000
    sma_target = Constants.SMA_EARTH
    pct_diff = (sma_achieved - sma_target)/sma_target*100
    test_log = log("Achieved SMA (m): " + str(sma_achieved), test_log, True)
    test_log = log("Target SMA (m): " + str(sma_target), test_log, True)
    test_log = log("Percent error (%): " + str(pct_diff), test_log, True)


    # generate and save figures
    fig_orb = eph.plot_xy()
    eph.plot_xy_ref_orbit(Constants.SMA_MARS, "Mars", "#b7410e")
    eph.plot_xy_ref_orbit(Constants.SMA_EARTH, "Earth")
    figs = eph.plot_all_ephemeris_data(False)

    path_traj_plot = os.path.join(path_plots,"traj_nn_propagation.jpg")
    path_fuel_use_plot = os.path.join(path_plots,"fuel_use_nn_propagation.jpg")
    path_thrust_dir_plot = os.path.join(path_plots,"thrust_dir_nn_propagation.jpg")
    path_throttle_plot = os.path.join(path_plots,"throttle_nn_propagation.jpg")
    path_ephem = os.path.join(path_test_dir,"ephemeris_nn_propagation.txt")

    fig_orb.savefig(path_traj_plot, bbox_inches="tight")
    figs[0].savefig(path_fuel_use_plot, bbox_inches="tight")
    figs[1].savefig(path_thrust_dir_plot, bbox_inches="tight")
    figs[2].savefig(path_throttle_plot, bbox_inches="tight")

    # save resultant ephemeris
    eph.write_to_file(path_ephem)

    test_log = log("\n", test_log, True)


prop_with_nn_controller(env, seed_in)
