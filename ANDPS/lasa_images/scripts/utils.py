from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
import geotorch
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    """ Custom Linear Movement Dataset. """

    def __init__(self, x, y):
        """
        Args:
            x (numpy array): # of demos x controllable vec size numpy array that contains x_c (INPUTS)
            y (numpy array): # of demos x controllable dvec/dt  size numpy array that contains x_cdot (OUTPUTS)
        """
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CNN_lasa_image(nn.Module):
    def __init__(self, classes, dim, N, images):
        super().__init__()
        self.images = [img.detach() for img in images]
        self.conv1 = nn.Conv2d(1, 5, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(5, 10, kernel_size=(5, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(1960+dim, 500)

        self.fc2 = nn.Linear(500, N)

    def forward(self, state):
        # batch of images
        img = torch.empty(size=(state.shape[0], 1, 68, 68)).to(state.device)
        for i in range(state.shape[0]):
            img[i] = self.images[int(state[i, 2])].clone().detach()

        x = self.pool1(F.relu(self.conv1(img)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # print(x.shape)
        # print(x)
        x = torch.cat((x, state[:, :2]), dim=1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class andps_images(nn.Module):
    def __init__(self, ds_dim, N, target, classes, images, device='cpu'):
        super().__init__()
        self.N = N
        self.ds_dim = ds_dim
        self.n_params = ds_dim
        self.all_weights = CNN_lasa_image(
            classes, self.ds_dim, N, images).to(device)
        self.all_params_B_A = nn.ModuleList(
            [nn.Linear(self.n_params, self.n_params, bias=False) for i in range(N)])
        # self.images = images
        for i in range(N):
            geotorch.positive_semidefinite(self.all_params_B_A[i])

        self.all_params_C_A = nn.ModuleList(
            [nn.Linear(self.n_params, self.n_params, bias=False) for i in range(N)])

        for i in range(N):
            geotorch.skew(self.all_params_C_A[i])
        self.x_tar = torch.Tensor(target).view(-1, ds_dim).to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        s_all = torch.zeros((1, self.ds_dim)).to(x.device)
        w_all = self.all_weights(x)

        for i in range(self.N):
            A = (self.all_params_B_A[i].weight + self.all_params_C_A[i].weight)
            s_all = s_all + torch.mul(w_all[:, i].view(batch_size, 1), torch.mm(A, (self.x_tar-x[:, :2]).transpose(0, 1)).transpose(0, 1))
        return s_all
