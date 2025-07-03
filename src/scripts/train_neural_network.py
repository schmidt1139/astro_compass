import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

# Adding python src code directory
current_dir = os.path.dirname(__file__)
python_src_dir = os.path.abspath(os.path.join(current_dir, "..", "python"))
sys.path.append(python_src_dir)

from gymnasium import envs
from gymnasium.envs.registration import register
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from Neural_Net_Controller import NN_TBT_Controller
from Training_Data_Generation import read_ephems_from_dir
from StateVectorUtilities import non_dimensionalize
from Constants import Constants
from Plotting_Utils import format_plots, plot_training_loss
from NN_Utils import evaluate_neural_network, pre_process_training_data, training_epoch

# register the environment if it isn't registered
if "TwoBody_Orb2Orb_Transfer_Env-v0" not in envs.registry.keys():
    register(
        id="TwoBody_Orb2Orb_Transfer_Env-v0",
        entry_point="TwoBody_Orb2Orb_Transfer_Env:TwoBody_Orb2Orb_Transfer_Env",
    )


# initialize the environment
env = gym.make("TwoBody_Orb2Orb_Transfer_Env-v0")

#format plotting
format_plots()

def train_neural_network():

    # parameters
    training_data_pts = 1000  # training data batch size
    training_epochs = 1000  # number of training epochs to run
    min_mse = 999999  # min mse init value
    patience = 200000  # If the number of iterations since  min is greater than this number - training ends
    learning_rate_i = 0.1 # Initial Parameter learning rate
    learning_rate_f = 0.1  # Final Parameter learning rate
    plot_update = training_epochs  # Number of epochs before plot is updated
    report_update = 10 #Number of epochs between reporting training status
    train_fraction = 0.8 #Fraction of data to use for training
    eval_fraction = 0.2 #Fraction of data to use for eval
    gamma_Steps = 1000 #Number of steps needed to reduce LR
    lr_gamma = 0.5 #Reduction factor for learning rate
    annealing_tmax = 1000

    params = {
        "mu":Constants.MU_SUN * 10 ** (9), #sun mu [m^3/s^2]
        "max_T":1.33, #max spacecraft thrust [N]
        "ISP":3872.0, #spacecraft specific impulse [s]
        "TOF":1.1 * 365.25 * 24 * 60 * 60, #assumed time of flight [s]
        "l_star":149598023000, #characteristic length = Earth SMA [m]
        "m_star":3366.0, #characteristic mass = SC initial mass [kg]
        "t_star":(149598023000**3 / (Constants.MU_SUN * 10 ** (9) ) ) ** 0.5, #characteristic time - derived
        "g0":Constants.G0 #gravtational acceleration at Earth surface [m/s^2]
    }

    # paths
    dir_training_dir = "\\data\\training_ephems\\test_set_smoothed_0.5\\"  # path to training data
    dir_plots = "\\data\\plots\\"  # path for storing plot data
    dir_nn = "\\data\\neural_networks\\"  # path for saving trained nn
    path_training_dir = os.getcwd() + dir_training_dir
    path_plots = os.getcwd() + dir_plots
    path_nn = os.getcwd() + dir_nn
    path_plot_nn_training = path_plots + "nn_training.jpg"

    # plotting structure init
    arr_epochs = []
    arr_loss = []

    # establish nn controller
    NN_TBT = NN_TBT_Controller()
    num_p = sum(p.numel() for p in NN_TBT.parameters() if p.requires_grad)

    # mse loss function
    criterion = nn.MSELoss()

    # establish optimizer
    # optimizer = torch.optim.Adam(NN_TBT.parameters(), lr=learning_rate_i)
    optimizer = torch.optim.SGD( NN_TBT.parameters(), lr=learning_rate_i )

    # define a LR scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=annealing_tmax, eta_min=learning_rate_f)

    # read ephemeris files
    set_ephems = read_ephems_from_dir(path_training_dir)
    num_ephems = len(set_ephems)
    print("Current wd: " + os.getcwd() )
    print("Reading ephems from " + path_training_dir)
    print(str(num_ephems) + " ephems loaded")
    print(str(num_ephems * set_ephems[0].num_vectors) + " training data points")
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

    #split into training and eval
    train_size = int(train_fraction * len(dataset))
    val_size = int(eval_fraction * len(dataset))

    train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # for reproducibility
    )

    # Training
    # --------------------------------------------------------------------------------------------------------

    arr_epochs = []
    arr_loss = []

    epoch = 1
    iters = 0
    i_at_min = 0
    min_mse = np.inf
    flag_exit = False

    #load parameters from file


    #set to training mode
    NN_TBT.train()

    # using torch loader object to load training and eval data
    train_loader = DataLoader(train_dataset, batch_size=training_data_pts, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_data_pts, shuffle=False)

    while epoch <= training_epochs:
        train_loss = 0
        num_pts = 0
        for batch_X, batch_y in train_loader:
            outputs = NN_TBT(batch_X)
            loss = criterion(outputs, batch_y)  # Calc loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # Clear gradients
            num_pts = num_pts + len(batch_X)
            train_loss += loss.item()*len(batch_X)

            iters = iters + 1

        avg_train_loss = train_loss / num_pts

        # check min loss
        if avg_train_loss < min_mse:
            min_mse = avg_train_loss
            i_at_min = iters

        # exit condition
        if iters > patience + i_at_min:
            print("Patience Criterion reached, exiting training")
            flag_exit = True
            break

        if flag_exit:
            break

        #eval NN
        if epoch % plot_update == 0:
            params['flag_plot'] = True
        else:
            params['flag_plot'] = False

        avg_loss_val = evaluate_neural_network(NN_TBT, val_loader, criterion, set_ephems[1], params )
        NN_TBT.train()

        print(
            f"Epoch [{epoch}/{training_epochs}], Training Loss: {avg_train_loss:.4e}, Eval loss: {avg_loss_val:.4e}   Min loss: {min_mse:.4e}   last min: {iters - i_at_min}   lr: {scheduler.get_last_lr()[0]:.4e}"
        )

        arr_epochs.append(epoch)
        arr_loss.append(avg_loss_val)

        if epoch % plot_update == 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(arr_epochs, arr_loss)
            ax.set_xlabel(r"Training Epochs")
            ax.set_ylabel(r"Eval Loss (MSE)")
            fig.tight_layout()
            fig.savefig(dir_plots + "nn_training.pdf")  # Vector format
            plt.show()

        epoch = epoch + 1

        # lr scheduler step
        if (epoch<annealing_tmax):
            scheduler.step()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(arr_epochs, arr_loss)
    ax.set_xlabel(r"Training Epochs")
    ax.set_ylabel(r"Loss (MSE)")
    fig.tight_layout()
    fig.savefig(dir_plots + "nn_training.pdf")  # Vector format

    # save NN to file
    torch.save(NN_TBT.state_dict(), dir_nn + "nn_controller_weights.pth")

train_neural_network()
# %%
