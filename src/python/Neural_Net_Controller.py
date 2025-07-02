import torch
import torch.nn as nn


class NN_TBT_Controller(nn.Module):
    def __init__(self):
        super(NN_TBT_Controller, self).__init__()
        print("Neural Network Controller Created")

        # LeakyReLU activation function
        self.lrelu = nn.LeakyReLU( inplace=True )

        self.fc1 = nn.Linear(5, 32)  # input layer
        self.fc2 = nn.Linear(32, 32)  # hidden layer 1
        self.fc3 = nn.Linear(32, 32)  # hidden layer 2
        self.fc4 = nn.Linear(32, 32)  # hidden layer 3
        self.fc5 = nn.Linear(32, 32)  # hidden layer 4
        self.fc6 = nn.Linear(32, 3)  # output layer

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.lrelu(self.fc3(x))
        x = self.lrelu(self.fc4(x))
        x = self.lrelu(self.fc5(x))
        x = self.fc6(x)

        # Apply sigmoid to throttle output (u), tanh to direction
        u = torch.sigmoid(x[:, 0:1])     # throttle in [0, 1]
        thrust_dir = torch.tanh(x[:, 1:])  # direction in [-1, 1], can be normalized later

        return torch.cat([u, thrust_dir], dim=1)
