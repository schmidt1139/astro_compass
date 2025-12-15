import torch
import torch.nn as nn


class NN_TBT_Controller(nn.Module):
    def __init__(self):
        super(NN_TBT_Controller, self).__init__()
        print("Neural Network Controller Created")

        self.fc1 = nn.Linear(5, 32)  # input layer
        self.fc2 = nn.Linear(32, 32)  # hidden layer 1
        self.fc3 = nn.Linear(32, 32)  # hidden layer 2
        self.fc4 = nn.Linear(32, 32)  # hidden layer 3
        self.fc5 = nn.Linear(32, 32)  # hidden layer 4
        self.fc6 = nn.Linear(32, 3)  # output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))

        return x
