import torch
from torch import nn
from torch.nn import functional as F
from andps import ANDP

# For this experiment we need to define a custom Weighting function
# Which means that we need to override the forward as well

# First we define the custom weighting function
# we decided to define the state as the controllable part of the state and then the image
# so our curent control state is x[:3] and the image is x[3:]
class CNN_weight(nn.Module):
    def __init__(self, ds_dim, N):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 5, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(5, 10, kernel_size=(5, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(1690+ds_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 3)
        self.fc6 = nn.Linear(64, N)

    def forward(self, state):
        # this expects the images as well as the current state
        x = self.pool1(
            F.relu(self.conv1(state[:, 3:].reshape(state.shape[0], 1, 64, 64))))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.cat((x, state[:, :3]), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.softmax(self.fc6(x), dim=1)

        return x


# let's define a class that inherits from ANDP
class cnn_ANDP(ANDP):
    def __init__(self, ds_dim, N, target, custom_weighting_function, device):
        super().__init__(ds_dim, N, target, all_weights=custom_weighting_function, device=device)


    def forward(self, x):
        batch_size = x.shape[0]
        s_all = torch.zeros((1, self.ds_dim)).to(x.device)
        w_all = self.all_weights(x)

        for i in range(self.N):
            A = self.all_params_B_A[i].weight + self.all_params_C_A[i].weight
            s_all = s_all + torch.mul(w_all[:, i].view(batch_size, 1), torch.mm(
                A, (self.x_target-x[:, :3]).transpose(0, 1)).transpose(0, 1))
        return s_all
