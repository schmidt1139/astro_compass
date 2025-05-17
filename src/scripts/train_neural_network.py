# %%
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# Adding module path from src
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, module_path)

from gymnasium import envs
from gymnasium.envs.registration import register
from torch.utils.data import DataLoader, TensorDataset
from Neural_Net_Controller import NN_TBT_Controller
from Training_Data_Generation import read_ephems_from_dir
from StateVectorUtilities import non_dimensionalize
from Constants import Constants
from Ephemeris import Ephemeris
from Hamiltonian_Control import Hamiltonian_Controller_TBT


# plotting setup
matplotlib.rcParams.update(
    {
        "text.usetex": False,  # Use LaTeX for all text
        "font.family": "serif",  # Use serif font
        "font.size": 10,  # Match AIAA body font size
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "figure.figsize": (3.5, 2.5),  # Single-column figure
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": False,  # No gridlines in AIAA style
    }
)

# register the environment if it isn't registered
if "TwoBody_Orb2Orb_Transfer_Env-v0" not in envs.registry.keys():
    register(
        id="TwoBody_Orb2Orb_Transfer_Env-v0",
        entry_point="TwoBody_Orb2Orb_Transfer_Env:TwoBody_Orb2Orb_Transfer_Env",
    )


# initialize the environment
env = gym.make("TwoBody_Orb2Orb_Transfer_Env-v0")


def train_neural_network():
    # Setup
    # --------------------------------------------------------------------------------------

    # parameters
    training_data_pts = 10000  # training data batch size
    training_epochs = 1000  # number of training epochs to run
    min_mse = 999999  # min mse init value
    patience = 10000  # If the number of iterations since latest min is greater than this number - training ends
    learning_rate = 0.01  # Parameter learning rate
    plot_update = 100  # Number of epochs before plot is updated

    # paths
    dir_training_dir = "data\\training_ephems\\test_set2\\"  # path to training data
    dir_plots = "data\\plots\\"  # path for storing plot data
    dir_nn = "data\\neural_networks\\"  # path for saving trained nn

    # plotting structure init
    arr_epochs = []
    arr_loss = []

    # establish nn controller
    NN_TBT = NN_TBT_Controller()
    num_p = sum(p.numel() for p in NN_TBT.parameters() if p.requires_grad)

    # mse loss function
    criterion = nn.MSELoss()

    # establish optimizer
    optimizer = torch.optim.Adam(NN_TBT.parameters(), lr=learning_rate)

    # define a LR scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # read ephemeris files
    print("Reading ephems from " + dir_training_dir)
    set_ephems = read_ephems_from_dir(dir_training_dir)
    num_ephems = len(set_ephems)
    print(str(num_ephems) + " ephems loaded")
    print(str(num_ephems * set_ephems[1].num_vectors) + " training data points")
    print("Number of Neural Network Parameters: " + str(num_p))

    # collect all training data
    matrix_training = []
    ref_matrix_training = []
    count = 0

    # Loading Training Data
    # -------------------------------------------------------------------------------------
    # step through all ephems and store data into one data structure for training
    for i, traj in enumerate(set_ephems):
        n_vecs = traj.num_vectors

        for j in range(0, n_vecs):
            vector = traj.get_vector_at_index(j)

            # unpack components of interest
            x = vector[1]
            y = vector[2]
            vx = vector[3]
            vy = vector[4]
            m = vector[5]
            alpha_x = vector[6]
            alpha_y = vector[7]
            u = vector[8]

            state = [x, y, vx, vy, m]
            max_T = 1.33
            mu = Constants.MU_SUN * 10 ** (9)
            ISP = 3872.0
            TOF = 1.1 * 365.25 * 24 * 60 * 60
            l_star = 149598023000
            m_star = 3366.0
            t_star = (l_star**3 / mu) ** 0.5

            # non-dimensionalize the state vector
            outputs = non_dimensionalize(
                state, Constants.G0, mu, max_T, ISP, TOF, l_star, m_star, t_star
            )

            # nn input (i.e. X vector)
            state_nd = np.array(outputs[0][0:4])

            # control reference data (i.e. Y vector)
            control_vec = np.array([alpha_x, alpha_y, u])

            # stack vectors
            matrix_training.append(state_nd)
            ref_matrix_training.append(control_vec)

            count = count + 1

    print("Total training data length: ", len(matrix_training))
    print("Count: ", count)

    # format into stacked tensor form
    matrix_training = [
        torch.tensor(vec, dtype=torch.float32) for vec in matrix_training
    ]
    ref_matrix_training = [
        torch.tensor(vec, dtype=torch.float32) for vec in ref_matrix_training
    ]
    matrix_training = torch.vstack(matrix_training)
    ref_matrix_training = torch.vstack(ref_matrix_training)

    # check the data arrays are the same shape
    assert matrix_training.shape[0] == ref_matrix_training.shape[0], (
        "Mismatch in number of samples"
    )

    # create combined dataset
    dataset = TensorDataset(matrix_training, ref_matrix_training)

    # Training
    # --------------------------------------------------------------------------------------------------------

    arr_epochs = []
    arr_loss = []

    epoch = 1
    iters = 0
    i_at_min = 0
    flag_exit = False

    # using torch loader object
    loader = DataLoader(dataset, batch_size=training_data_pts, shuffle=True)

    while epoch <= training_epochs:
        for batch_X, batch_y in loader:
            outputs = NN_TBT(batch_X)
            loss = criterion(outputs, batch_y)  # Calc loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # Clear gradients

            # check min loss
            if loss.item() < min_mse:
                min_mse = loss.item()
                i_at_min = iters

            # exit condition
            if iters > patience + i_at_min:
                print("Patience Criterion reached, exiting training")
                flag_exit = True
                break

            iters = iters + 1

        if flag_exit:
            break

        print(
            f"Epoch [{epoch}/{training_epochs}], Loss: {loss.item():.4f}   Min loss: {min_mse:.4f}   last min: {iters - i_at_min}   lr: {scheduler.get_last_lr()[0]:.4e}"
        )

        arr_epochs.append(epoch)
        arr_loss.append(loss.item())

        if epoch % plot_update == 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(arr_epochs, arr_loss)
            ax.set_xlabel(r"Training Epochs")
            ax.set_ylabel(r"Training Loss (MSE)")
            ax.set_ylim([0, 0.2])
            fig.tight_layout()
            fig.savefig(dir_plots + "nn_training.pdf")  # Vector format

        epoch = epoch + 1

        # lr scheduler step
        scheduler.step()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(arr_epochs, arr_loss)
    ax.set_xlabel(r"Training Epochs")
    ax.set_ylabel(r"Loss (MSE)")
    ax.set_ylim([0, 0.2])
    fig.tight_layout()
    fig.savefig(dir_plots + "nn_training.pdf")  # Vector format

    # save NN to file
    torch.save(NN_TBT.state_dict(), dir_nn + "nn_controller_weights.pth")

    # Compare with Hamiltonian control
    # --------------------------------------------------------------------------------------------------------
    eph = Ephemeris()

    # reset the TBT env
    seed = 1
    input_TOF = 1.1 * 365.25 * 24 * 60 * 60
    init_observation, init_info = env.reset(seed=seed)

    # compute Hamiltonian Solution
    H_controller = Hamiltonian_Controller_TBT(
        env, init_observation, init_info, input_TOF
    )

    # alter parameters
    H_controller.eps_threshold = 0.0001
    H_controller.arr_lam_0 = np.array([-0.575, -0.261, -0.407, -1.047, 0.139])

    # compute solution
    flag_solved, h_sol, eps, sol, log = H_controller.hamiltonian_solution_finder()

    # write output ephemeris
    eph_out, arr_time, arr_u, arr_rho, arr_alpha_x, arr_alpha_y = (
        H_controller.generate_output_ephemeris(eph)
    )

    arr_time = arr_time * H_controller.t_star / 86400  # convert to days

    arr_u_nn = []
    arr_a_x = []
    arr_a_y = []
    NN_TBT.eval()

    # compute nn control
    for index, t in enumerate(arr_time):
        state_vec = eph_out.get_vector_at_index(index)

        outputs = non_dimensionalize(
            state_vec, Constants.G0, mu, max_T, ISP, TOF, l_star, m_star, t_star
        )

        state_nd = np.array(outputs[0][0:4], dtype=np.float32)
        state_tensor = torch.from_numpy(state_nd).unsqueeze(0)

        with torch.no_grad():
            control_nn = NN_TBT(state_tensor)

        arr_u_nn.append(control_nn[0][0])
        arr_a_x.append(control_nn[0][1])
        arr_a_y.append(control_nn[0][2])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(eph_out.arr_et, arr_u, label="Hamiltonian Control")
    ax.plot(eph_out.arr_et, arr_u_nn, label="Neural Network")
    ax.set_xlabel(r"Elapsed Days")
    ax.set_ylabel(r"Throttle Input (u)")
    fig.tight_layout()
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(dir_plots + "u_compare.pdf")  # Vector format

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(eph_out.arr_et, arr_alpha_x, label=r"H $\alpha_x$")
    ax.plot(eph_out.arr_et, arr_alpha_y, label=r"H $\alpha_y$")
    ax.plot(eph_out.arr_et, arr_a_x, label=r"NN $\alpha_x$")
    ax.plot(eph_out.arr_et, arr_a_y, label=r"NN $\alpha_y$")
    ax.set_xlabel(r"Elapsed Days")
    ax.set_ylabel(r"Thrust Direction ($\alpha_x,\alpha_y$)")
    fig.tight_layout()
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(dir_plots + "a_compare.pdf")  # Vector format


train_neural_network()
# %%
