import sys
import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

# Adding python src code directory
current_dir = os.path.dirname(__file__)
python_src_dir = os.path.abspath(os.path.join(current_dir, "..", "python"))
sys.path.append(python_src_dir)

from gymnasium import envs
from gymnasium.envs.registration import register
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from Neural_Net_Controllers import NN_TBT_Controller_alpha
from Training_Data_Generation import read_ephems_from_dir
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

# format plotting
format_plots()


def train_u_network():

    # parameters
    params = {
        "training_data_pts": 1000, # training data batch size
        "training_epochs": 1000, # number of training epochs to run
        "patience": 200000, # Epochs without training loss improvement to stop training
        "learning_rate_i": 0.1, # Initial Parameter learning rate
        "learning_rate_f": 0.1, # Final Parameter learning rate
        "plot_update": 1000, # Number of epochs before plot is updated
        "report_update": 1, # Number of epochs between reporting training status
        "train_fraction": 0.8, # Fraction of data to use for training
        "eval_fraction": 0.2, # Fraction of data to use for eval
        "annealing_tmax": 1000, # Cosine annealing max iters
        "loss": "MSE", #MSE, BCEWithLogitsLoss
        "control_data_set": "alpha", # Control data sets to train (all, u, alpha)
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

    # paths
    print("Current wd: " + os.getcwd())
    dir_training_dir = (
        "..\\data\\training_ephems\\test_set_bang_bang\\"  # path to training data
    )
    dir_plots = "..\\data\\plots\\"  # path for storing plot data
    dir_nn = "..\\data\\neural_networks\\"  # path for saving trained nn
    path_training_dir = os.path.normpath(os.path.join(os.getcwd(), dir_training_dir))
    path_plots = os.path.normpath(os.path.join(os.getcwd(), dir_plots))
    path_nn = os.path.normpath(os.path.join(os.getcwd(), dir_nn))
    path_plot_nn_training = path_plots + "nn_training.jpg"

    # plotting structure init
    arr_epochs = []
    arr_loss = []

    # establish nn controller (throttle only)
    NN_TBT = NN_TBT_Controller_alpha()
    num_p = sum(p.numel() for p in NN_TBT.parameters() if p.requires_grad)

    # mse loss function
    if ( params["loss"] == "BCEWithLogitsLoss" ):
        criterion = nn.BCEWithLogitsLoss()
    elif( params["loss"] == "MSE" ):
        criterion = nn.MSELoss()
    
    min_loss = np.inf  # min mse init value

    # establish optimizer
    # optimizer = torch.optim.Adam(NN_TBT.parameters(), lr=learning_rate_i)
    optimizer = torch.optim.SGD(NN_TBT.parameters(), lr=params["learning_rate_i"])

    # define a LR scheduler
    scheduler = CosineAnnealingLR(
        optimizer, T_max=params["annealing_tmax"], eta_min=params["learning_rate_f"]
    )

    # read ephemeris files
    set_ephems = read_ephems_from_dir(path_training_dir)
    num_ephems = len(set_ephems)
    print("Reading ephems from " + path_training_dir)
    print(str(num_ephems) + " ephems loaded")
    print(str(num_ephems * set_ephems[0].num_vectors) + " training data points")
    print("Number of Neural Network Parameters: " + str(num_p))

    train_dataset, val_dataset = pre_process_training_data(
        set_ephems, params
    )

    # Training
    # --------------------------------------------------------------------------------------------------------

    arr_epochs = []
    arr_loss = []
    arr_loss_train = []

    epoch = 1
    i_at_min = 0
    min_loss = np.inf
    flag_exit = False

    # set to training mode
    NN_TBT.train()

    # using torch loader object to load training and eval data
    train_loader = DataLoader(train_dataset, batch_size=params["training_data_pts"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params["training_data_pts"], shuffle=False)

    while epoch <= params["training_epochs"]:

        # perform training epoch
        NN_TBT, avg_train_loss = training_epoch(
            NN_TBT, train_loader, val_loader, criterion, optimizer
        )

        # check min loss
        if avg_train_loss < min_loss:
            min_loss = avg_train_loss
            i_at_min = epoch

        # exit condition
        if epoch > params["patience"] + i_at_min:
            print("Patience Criterion reached, exiting training")
            flag_exit = True
            break

        if flag_exit:
            break

        # eval NN
        if epoch % params["plot_update"] == 0:
            params["flag_plot"] = True
        else:
            params["flag_plot"] = False

        avg_loss_val = evaluate_neural_network(
            NN_TBT, val_loader, criterion, params, path_plots, set_ephems[0]
        )
        NN_TBT.train()

        arr_epochs.append(epoch)
        arr_loss_train.append(avg_train_loss)
        arr_loss.append(avg_loss_val)

        if epoch % params["report_update"] == 0:
            print(
                f"Epoch [{epoch}/{params["training_epochs"]}], Training Loss: {avg_train_loss:.4e}, Eval loss: {avg_loss_val:.4e}   Min loss: {min_loss:.4e}   last min: {epoch - i_at_min}   lr: {scheduler.get_last_lr()[0]:.4e}"
            )

        if epoch % params["plot_update"] == 0:
            plot_training_loss(
                arr_epochs, arr_loss_train, arr_loss, path_plot_nn_training, params
            )

        epoch = epoch + 1

    # final training plot update
    plot_training_loss(arr_epochs, arr_loss_train, arr_loss, path_plot_nn_training, params)

    # save NN to file
    torch.save(NN_TBT.state_dict(), path_nn + "nn_controller_weights.pth")


train_u_network()
