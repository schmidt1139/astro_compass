import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from StateVectorUtilities import non_dimensionalize
from Constants import Constants
from torch.utils.data import DataLoader, TensorDataset, random_split

def evaluate_neural_network(NN_TBT, val_loader, criterion, params, dir_plots, ephem_compare ):

    #evalualte the nn
    NN_TBT.eval()
    num_samples = 0

    with torch.no_grad():
        val_loss = 0
        for val_inputs, val_targets in val_loader:
            outputs = NN_TBT(val_inputs)
            batch_size = val_inputs.size(0)
            loss = criterion(outputs, val_targets)
            val_loss += loss.item()*batch_size
            num_samples += batch_size

        if ( num_samples > 0 ):
            avg_loss = val_loss / num_samples
        else:
            avg_loss = 0

    if ( (params["flag_plot"] == True ) and num_samples > 0 ):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(outputs[:,0], val_targets[:,0], label="Training Targets")
        ax.scatter(outputs[:,0], outputs[:,0], label="NN Values")
        ax.set_xlabel(r"Control ref u ")
        ax.set_ylabel(r"Control deltas u")
        ax.legend()
        fig.tight_layout()
        fig.savefig(dir_plots + "nn_val_compare_u.jpg")  # Vector format

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(outputs[:,1], val_targets[:,1], label=r"Training Targets $\alpha_y$")
        ax.scatter(outputs[:,1], outputs[:,1], label=r"NN Values $\alpha_x$")
        ax.set_xlabel(r"Control ref $\alpha_x$")
        ax.set_ylabel(r"Control deltas $\alpha_y$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(dir_plots + "nn_val_compare_alpha_x.jpg")  # Vector format

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(outputs[:,2], val_targets[:,2], label=r"Training Targets $\alpha_y$")
        ax.scatter(outputs[:,2], outputs[:,2], label=r"NN Values $\alpha_y$")
        ax.set_xlabel(r"Control ref $\alpha_y$") 
        ax.set_ylabel(r"Control deltas $\alpha_y$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(dir_plots + "nn_val_compare_alpha_y.jpg")  # Vector format

    #compare NN outputs with ephemeris control
    if ( params["flag_plot"] == True ):
        compare_NN_with_ephem(NN_TBT, ephem_compare, dir_plots, params )

    return avg_loss

def compare_NN_with_ephem( NN_TBT, sample_ephem_compare, dir_plots, params_ND ):

    arr_u_nn = []
    arr_ax_nn = []
    arr_ay_nn = []

    for i, t in enumerate(sample_ephem_compare.arr_et):

        vector = sample_ephem_compare.get_vector_at_index(i)

        arr_control = query_NN_at_ephem_state(NN_TBT,vector,params_ND)

        arr_u_nn.append( arr_control[0] )
        arr_ax_nn.append( arr_control[1] )
        arr_ay_nn.append( arr_control[2] )



    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot( sample_ephem_compare.arr_et/86400, sample_ephem_compare.arr_u, label="Ephemeris" )
    ax.plot( sample_ephem_compare.arr_et/86400, arr_u_nn, label="Neural Network" )
    ax.set_xlabel("Elapsed time [days]") 
    ax.set_ylabel("Ephemeris Throttle u")
    ax.legend()
    ax.legend()
    fig.tight_layout()
    fig.savefig(dir_plots + "nn_ephem_compare_u.jpg")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot( sample_ephem_compare.arr_et/86400, sample_ephem_compare.arr_alpha_x, label=r"Ephemeris $\alpha_x$", color = "red" )
    ax.plot( sample_ephem_compare.arr_et/86400, arr_ax_nn, label=r"Neural Network $\alpha_x$", color = "orange" )
    ax.plot( sample_ephem_compare.arr_et/86400, sample_ephem_compare.arr_alpha_y, label=r"Ephemeris $\alpha_y$", color = "blue" )
    ax.plot( sample_ephem_compare.arr_et/86400, arr_ay_nn, label=r"Neural Network $\alpha_y$", color = "green" )
    ax.set_xlabel("Elapsed time [days]") 
    ax.set_ylabel("Ephemeris Thrust Direction")
    ax.legend()
    ax.legend()
    fig.tight_layout()
    fig.savefig(dir_plots + "nn_ephem_compare_alpha.jpg")

def query_NN_at_ephem_state(NN_TBT, vector, params_ND):

    # unpack components of interest
    x = vector[1]
    y = vector[2]
    vx = vector[3]
    vy = vector[4]
    m = vector[5]
    alpha_x = vector[6]
    alpha_y = vector[7]
    u = vector[8]

    #pack into an array
    state = [x, y, vx, vy, m]

    # non-dimensionalize the state vector
    state_nd = non_dimensionalize(
        state, params_ND["g0"], params_ND["mu"], params_ND["max_T"], params_ND["ISP"], 
        params_ND["TOF"], params_ND["l_star"], params_ND["m_star"], params_ND["t_star"]
    )

    #pack state as a tensor
    nn_input = torch.tensor( np.array(state_nd[0][0:5]), dtype=torch.float32 )

    #eval NN
    with torch.no_grad():
        nn_control = NN_TBT( nn_input )

    #convert to an array
    nn_control = np.array(nn_control)

    return nn_control

def pre_process_training_data(set_ephems, train_fraction, eval_fraction, params):

    # training data collections
    matrix_training = []
    ref_matrix_training = []
    count = 0

    # step through all ephems and store data into one data structure for training
    for traj in set_ephems:
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

            # non-dimensionalize the state vector
            outputs = non_dimensionalize(
                state, params["g0"], params["mu"], params["max_T"], params["ISP"], params["TOF"], params["l_star"], params["m_star"], params["t_star"]
            )

            # nn input (i.e. X vector)
            state_nd = np.array(outputs[0][0:5])

            # control reference data (i.e. Y vector)
            control_vec = np.array([u, alpha_x, alpha_y])

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

    #random splitting of eval data
    train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # for reproducibility
    )

    return train_dataset, val_dataset

def training_epoch( NN_TBT, train_loader, val_loader, criterion, optimizer ):

    train_loss = 0
    num_pts = 0
    iters = 0

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

    return NN_TBT, avg_train_loss


